"""Local generation through the Ollama CLI."""
import subprocess

from .errors import ProviderError


class LocalModel:
    def __init__(self, config):
        self.name = config.get('name', 'gemma2:2b')
        self.provider = config.get('provider', 'ollama')

    def generate(self, prompt: str) -> str:
        if self.provider != 'ollama':
            raise ProviderError("Select Ollama as the local provider.")
        try:
            proc = subprocess.run(
                ["ollama", "run", self.name], input=prompt.encode('utf-8'),
                capture_output=True, timeout=45,
            )
        except FileNotFoundError:
            raise ProviderError("Install Ollama and make sure it is on PATH.") from None
        except subprocess.TimeoutExpired:
            raise ProviderError("Ollama timed out. Try again or select a smaller model.") from None
        except OSError:
            raise ProviderError("Ollama could not start. Check your installation.") from None
        if proc.returncode != 0:
            raise ProviderError("Ollama could not run the model. Check that Ollama is running and the configured model is available.")
        answer = proc.stdout.decode('utf-8', 'replace')
        if not answer.strip():
            raise ProviderError("Ollama returned an empty response. Try again.")
        return answer

    def stream(self, prompt: str):
        # The CLI buffers output here. Yield the original text with whitespace intact.
        yield self.generate(prompt)
