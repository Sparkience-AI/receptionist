import unittest
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

from receptionist.models.cloud import CloudModel
from receptionist.models.errors import ProviderError


class BringYourOwnKeyTests(unittest.TestCase):
    def test_provider_request_and_usage(self):
        cases = (
            ("anthropic", "api.anthropic.com", "x-api-key", {
                "content": [{"type": "text", "text": "OK"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }),
            ("google", "generativelanguage.googleapis.com", "x-goog-api-key", {
                "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1, "thoughtsTokenCount": 1},
            }),
        )
        for provider, host, header, body in cases:
            with self.subTest(provider=provider), patch("receptionist.models.byok.httpx.post") as post:
                post.return_value.json.return_value = body
                usage = []
                model = CloudModel({"provider": provider, "name": "test-model", "api_key": "test-key"})
                result = "".join(model.stream("prompt", "system", on_usage=lambda *counts: usage.append(counts)))
                self.assertEqual(result, "OK")
                self.assertEqual(usage, [(4, 2)])
                self.assertEqual(httpx.URL(post.call_args.args[0]).host, host)
                self.assertEqual(post.call_args.kwargs["headers"][header], "test-key")

    def test_provider_error_does_not_expose_response(self):
        model = CloudModel({"provider": "google", "api_key": "test-key"})
        with patch("receptionist.models.byok.httpx.post") as post:
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            post.return_value = httpx.Response(401, request=request, text="raw provider detail")
            with self.assertRaisesRegex(ProviderError, "HTTP 401") as result:
                model.generate("synthetic prompt")
            self.assertNotIn("raw provider detail", str(result.exception))

    def test_sidebar_keeps_local_default_and_uses_selected_key_once(self):
        with patch("receptionist.models.byok.httpx.post") as post, \
                patch("receptionist.models.local.LocalModel.generate", return_value="local reply"):
            post.return_value.json.return_value = {
                "content": [{"type": "text", "text": "cloud reply"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
            app = AppTest.from_file("streamlit_app.py").run()
            app.selectbox[0].set_value("anthropic").run()
            keys = {widget.label: widget for widget in app.text_input}
            keys["OpenAI API key"].set_value("unused-key")
            keys["Anthropic API key"].set_value("anthropic-key").run()
            app.chat_input[0].set_value("Explain an API contract.").run()
            post.assert_not_called()
            app.checkbox[0].check().run()
            app.chat_input[0].set_value("hello").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.checkbox[0].value)
            reply = app.session_state.messages[-1]
            self.assertEqual(reply["details"]["provider"], "anthropic")
            self.assertNotIn("anthropic-key", repr(app.session_state.messages))
            self.assertEqual(post.call_args.kwargs["headers"]["x-api-key"], "anthropic-key")
            self.assertEqual(app.session_state.usage_metrics.baseline_model, "anthropic/claude-haiku-4-5")
            app.chat_input[0].set_value("hello").run()
            post.assert_called_once()
