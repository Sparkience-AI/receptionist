"""Buffered and streaming requests to Anthropic and Gemini."""
import json
import re
from contextlib import contextmanager

import httpx

from .errors import ProviderError


def _request(provider, model, key, prompt, system_prompt):
    name = "Anthropic" if provider == "anthropic" else "Gemini"
    if not key:
        raise ProviderError(f"Enter a {name} API key in the sidebar or set it in the environment.")
    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
        if system_prompt:
            body["system"] = system_prompt
    else:
        model = model.removeprefix("models/")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", model):
            raise ProviderError("Enter a valid Gemini model ID.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": key}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    return name, url, headers, body


@contextmanager
def _provider_errors(name):
    try:
        yield
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        message = {
            401: "Check your API key.",
            403: "Check your API key and model permissions.",
            404: "Check the model ID and access.",
            429: "Rate limit or quota reached. Try again later.",
        }.get(status, "Check the configured model and try again.")
        raise ProviderError(f"{name} returned HTTP {status}. {message}") from None
    except httpx.TimeoutException:
        raise ProviderError(f"{name} timed out. Try again.") from None
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        raise ProviderError(f"{name} returned an invalid response or could not connect.") from None


def generate(provider, model, key, prompt, system_prompt, on_usage):
    name, url, headers, body = _request(provider, model, key, prompt, system_prompt)
    with _provider_errors(name):
        response = httpx.post(url, headers=headers, json=body, timeout=45.0)
        response.raise_for_status()
        data = response.json()
        if provider == "anthropic":
            usage = data.get("usage", {})
            inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
            parts = [part for part in data.get("content", []) if part.get("type") == "text"]
            incomplete = data.get("stop_reason") == "max_tokens"
        else:
            usage = data.get("usageMetadata", {})
            inputs = usage.get("promptTokenCount")
            outputs = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            candidates = data.get("candidates") or [{}]
            parts = [part for part in candidates[0].get("content", {}).get("parts", []) if not part.get("thought")]
            incomplete = candidates[0].get("finishReason") == "MAX_TOKENS"
        if on_usage is not None and inputs is not None and outputs is not None:
            on_usage(inputs, outputs)
        answer = "".join(part.get("text", "") for part in parts)
    if not answer.strip():
        raise ProviderError(f"{name} returned no text. Try a different prompt or model.")
    if incomplete:
        raise ProviderError(f"{name} reached its output limit.", partial=answer)
    return answer



def _events(lines):
    """Decode the JSON data in each SSE event."""
    payload = []
    for line in lines:
        if line.startswith("data:"):
            payload.append(line[5:].lstrip(" "))
        elif not line and payload:
            yield json.loads("\n".join(payload))
            payload.clear()
    if payload:
        yield json.loads("\n".join(payload))


def _anthropic_event(event, usage):
    kind = event.get("type")
    if kind == "error":
        raise ProviderError("Anthropic interrupted the stream. Try again.")
    if kind == "message_start":
        usage.update(event.get("message", {}).get("usage") or {})
    if kind == "message_delta":
        usage.update(event.get("usage") or {})
    delta = event.get("delta") or {}
    block = event.get("content_block") or {}
    if kind == "content_block_delta" and delta.get("type") == "text_delta":
        return delta.get("text", ""), delta.get("stop_reason"), False
    if kind == "content_block_start" and block.get("type") == "text":
        return block.get("text", ""), None, False
    return "", delta.get("stop_reason"), kind == "message_stop"


def _gemini_event(event, usage):
    if "error" in event:
        raise ProviderError("Gemini interrupted the stream. Try again.")
    usage.update(event.get("usageMetadata") or {})
    candidate = (event.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    reason = candidate.get("finishReason")
    return text, reason, reason not in (None, "FINISH_REASON_UNSPECIFIED")


def stream(provider, model, key, prompt, system_prompt, on_usage):
    name, url, headers, body = _request(provider, model, key, prompt, system_prompt)
    anthropic = provider == "anthropic"
    if anthropic:
        body["stream"] = True
    else:
        url = url.removesuffix(":generateContent") + ":streamGenerateContent?alt=sse"

    usage = {}
    finished = False
    reason = None
    has_text = False
    parse_event = _anthropic_event if anthropic else _gemini_event
    try:
        with _provider_errors(name), httpx.stream(
            "POST", url, headers=headers, json=body, timeout=45.0,
        ) as response:
            response.raise_for_status()
            for event in _events(response.iter_lines()):
                delta, stop_reason, terminal = parse_event(event, usage)
                reason = stop_reason or reason
                finished = finished or terminal
                if delta:
                    has_text = has_text or bool(delta.strip())
                    yield delta
    finally:
        if anthropic:
            inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
        else:
            inputs = usage.get("promptTokenCount")
            outputs = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        if on_usage is not None and inputs is not None and outputs is not None:
            on_usage(inputs, outputs)

    if not finished:
        raise ProviderError(f"The {name} stream ended early. Try again.")
    if reason not in (None, "end_turn", "stop_sequence", "STOP"):
        raise ProviderError(f"{name} stopped before completing the response.")
    if not has_text:
        raise ProviderError(f"{name} returned no text. Try a different prompt or model.")
