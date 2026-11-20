"""Cloud model abstraction with basic multi-provider support & streaming.

Currently implemented providers: openai (others return stub message).
Add keys via environment variables or config api_keys.
"""
from __future__ import annotations
import os
from typing import Iterable, Generator, Dict, Any
from openai import OpenAI as OpenAIClient


SUPPORTED_PREVIEW_MODELS = {"gpt-5-preview"}

class CloudModel:
	def __init__(self, config: Dict[str, Any], api_keys: Dict[str,str] | None = None):
		self.name = config.get('name', 'gpt-4o-mini')
		self.provider = config.get('provider', 'openai')
		self.api_keys = api_keys or {}
		# Resolve key precedence: explicit config api_key -> api_keys mapping -> env var
		explicit = config.get('api_key')
		if explicit:
			self.api_key = explicit
		else:
			self.api_key = self.api_keys.get(self.provider) or os.getenv(f"{self.provider.upper()}_API_KEY")

	def generate(self, prompt: str, system_prompt: str | None = None) -> str:
		if self.provider == 'openai':
			if OpenAIClient is None:
				return "[OpenAI client not installed]"
			client = OpenAIClient(api_key=self.api_key)
			messages = []
			if system_prompt:
				messages.append({"role":"system","content":system_prompt})
			messages.append({"role":"user","content":prompt})
			try:
				resp = client.chat.completions.create(model=self.name, messages=messages)
				return resp.choices[0].message.content.strip()
			except Exception as e:
				return f"[OpenAI error: {e}]"
		return f"[Provider {self.provider} not yet implemented]"

	def stream(self, prompt: str, system_prompt: str | None = None) -> Generator[str,None,None]:
		if self.provider == 'openai':
			if OpenAIClient is None:
				yield "[OpenAI client not installed]"
				return
			client = OpenAIClient(api_key=self.api_key)
			messages = []
			if system_prompt:
				messages.append({"role":"system","content":system_prompt})
			messages.append({"role":"user","content":prompt})
			try:
				stream = client.chat.completions.create(model=self.name, messages=messages, stream=True)
				for chunk in stream:
					delta = chunk.choices[0].delta.content or ""
					if delta:
						yield delta
			except Exception as e:
				yield f"[OpenAI stream error: {e}]"
			return
		yield f"[Streaming not supported for provider {self.provider}]"
