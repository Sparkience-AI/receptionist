"""Buffered generation through the Anthropic and Gemini APIs."""
import re

import httpx

from .errors import ProviderError


def generate(provider, model, key, prompt, system_prompt, on_usage):
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

    try:
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

    if not answer.strip():
        raise ProviderError(f"{name} returned no text. Try a different prompt or model.")
    if incomplete:
        raise ProviderError(f"{name} reached its output limit.", partial=answer)
    return answer
