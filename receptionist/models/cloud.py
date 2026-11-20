"""OpenAI generation and streaming with optional usage reporting."""
from __future__ import annotations

import os
from contextlib import contextmanager

from httpx import HTTPError
from openai import APITimeoutError, AuthenticationError, NotFoundError, OpenAIError, RateLimitError
from openai import OpenAI as OpenAIClient

from .byok import generate as generate_byok
from .errors import ProviderError


class CloudModel:
    def __init__(self, config, api_keys=None):
        self.name = config.get('name', 'gpt-4o-mini')
        self.provider = config.get('provider', 'openai')
        self.api_key = (config.get('api_key') or (api_keys or {}).get(self.provider)
                        or os.getenv(f"{self.provider.upper()}_API_KEY")
                        or (os.getenv("GEMINI_API_KEY") if self.provider == "google" else None))

    @contextmanager
    def _completion(self, prompt, system_prompt, **options):
        if self.provider != 'openai':
            raise ProviderError("Select OpenAI, Anthropic, or Gemini as the cloud provider.")
        if not self.api_key:
            raise ProviderError("Enter an OpenAI API key in the sidebar or set OPENAI_API_KEY.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            client = OpenAIClient(api_key=self.api_key, timeout=45.0, max_retries=0)
            yield client.chat.completions.create(model=self.name, messages=messages, **options)
        except AuthenticationError:
            raise ProviderError("OpenAI rejected the credentials. Check OPENAI_API_KEY.") from None
        except NotFoundError:
            raise ProviderError("OpenAI could not find the configured model. Check its name and access.") from None
        except RateLimitError:
            raise ProviderError("OpenAI rate limit or quota reached. Check your quota or try again later.") from None
        except APITimeoutError:
            raise ProviderError("OpenAI timed out. Try again.") from None
        except (OpenAIError, HTTPError, OSError):
            raise ProviderError("The OpenAI request failed. Check the connection or try again.") from None

    @staticmethod
    def _report_usage(response, on_usage):
        usage = getattr(response, 'usage', None)
        if on_usage is not None and usage is not None:
            on_usage(usage.prompt_tokens, usage.completion_tokens)

    def generate(self, prompt, system_prompt=None, *, on_usage=None):
        if self.provider in ("anthropic", "google"):
            return generate_byok(self.provider, self.name, self.api_key, prompt, system_prompt, on_usage)
        with self._completion(prompt, system_prompt) as response:
            self._report_usage(response, on_usage)
            answer = response.choices[0].message.content if response.choices else None
            if not answer or not answer.strip():
                raise ProviderError("OpenAI returned an empty response. Try again.")
            if response.choices[0].finish_reason in ('length', 'content_filter'):
                raise ProviderError('OpenAI stopped before completing the response.', partial=answer)
            return answer

    def stream(self, prompt, system_prompt=None, *, on_usage=None):
        if self.provider in ("anthropic", "google"):
            yield self.generate(prompt, system_prompt, on_usage=on_usage)
            return
        options = {"stream_options": {"include_usage": True}} if on_usage is not None else {}
        finish_reason = None
        has_text = False
        with self._completion(prompt, system_prompt, stream=True, **options) as stream:
            for chunk in stream:
                self._report_usage(chunk, on_usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                if choice.delta.content:
                    has_text = has_text or bool(choice.delta.content.strip())
                    yield choice.delta.content
        if finish_reason is None:
            raise ProviderError("The OpenAI stream ended early. Try again.")
        if finish_reason in ('length', 'content_filter'):
            raise ProviderError('OpenAI stopped before completing the response.')
        if not has_text:
            raise ProviderError("OpenAI returned an empty response. Try again.")
