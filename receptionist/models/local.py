"""Local model interface (Ollama) with safer subprocess handling.

Future: add native python API if exposed; currently shells out.
"""
from __future__ import annotations
import subprocess, json, shlex
from typing import Generator

class LocalModel:
	def __init__(self, config):
		self.name = config.get('name', 'gemma2:2b')
		self.provider = config.get('provider', 'ollama')

	def generate(self, prompt: str) -> str:
		if self.provider != 'ollama':
			return f"[Local provider {self.provider} not supported]"
		try:
			# Use stdin to avoid command-line length / quoting issues
			proc = subprocess.run([
				"ollama","run", self.name
			], input=prompt.encode('utf-8'), capture_output=True, timeout=45)
			if proc.returncode != 0:
				return f"[Local model error: {proc.stderr.decode('utf-8', 'ignore')[:200]}]"
			return proc.stdout.decode('utf-8','replace').strip()
		except FileNotFoundError:
			return "[Ollama not installed or not in PATH]"
		except Exception as e:
			return f"[Local model exception: {e}]"

	def stream(self, prompt: str) -> Generator[str,None,None]:
		# Basic chunked yield from full result (Ollama CLI streaming not implemented here)
		text = self.generate(prompt)
		for chunk in text.split():
			yield chunk + ' '
