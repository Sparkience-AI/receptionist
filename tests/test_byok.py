import json
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
                result = model.generate("prompt", "system", on_usage=lambda *counts: usage.append(counts))
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
            app.checkbox[1].uncheck().run()
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

    def test_native_streams_yield_deltas_and_report_usage(self):
        anthropic = [
            {"type":"message_start","message":{"usage":{"input_tokens":4,"output_tokens":0}}},
            {"type":"content_block_delta","delta":{"type":"text_delta","text":"O"}},
            {"type":"content_block_delta","delta":{"type":"text_delta","text":"K"}},
            {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}},
            {"type":"message_stop"},
        ]
        gemini = [
            {"candidates":[{"content":{"parts":[{"text":"hidden","thought":True},{"text":"O"}]}}]},
            {"candidates":[{"content":{"parts":[{"text":"K"}]},"finishReason":"STOP"}]},
            {"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1,"thoughtsTokenCount":1}},
        ]
        for provider, events in (("anthropic", anthropic), ("google", gemini)):
            with self.subTest(provider=provider), patch("receptionist.models.byok.httpx.stream") as request:
                context = request.return_value
                response = context.__enter__.return_value
                seen = []
                def lines():
                    for event in events:
                        seen.append(event)
                        yield "data: " + json.dumps(event)
                        yield ""
                response.iter_lines.return_value = lines()
                usage = []
                model = CloudModel({"provider":provider,"name":"test-model","api_key":"test-key"})
                reply = model.stream("prompt", "system", on_usage=lambda *counts: usage.append(counts))
                self.assertEqual(next(reply), "O")
                self.assertLess(len(seen), len(events))
                self.assertEqual(list(reply), ["K"])
                self.assertEqual(usage, [(4,2)])
                context.__exit__.assert_called_once()
                if provider == "anthropic":
                    self.assertTrue(request.call_args.kwargs["json"]["stream"])
                else:
                    self.assertTrue(request.call_args.args[1].endswith(":streamGenerateContent?alt=sse"))

    def test_interruption_preserves_yielded_text_and_cancel_closes_stream(self):
        for provider, event in (
            ("anthropic", {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}),
            ("google", {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}),
        ):
            for cancel in (True, False):
                with self.subTest(provider=provider,cancel=cancel), patch("receptionist.models.byok.httpx.stream") as request:
                    context = request.return_value
                    response = context.__enter__.return_value
                    response.iter_lines.return_value = iter(["data: " + json.dumps(event), ""])
                    reply = CloudModel({"provider":provider,"name":"test-model","api_key":"test-key"}).stream("prompt")
                    self.assertEqual(next(reply), "partial")
                    if cancel:
                        reply.close()
                    else:
                        with self.assertRaisesRegex(ProviderError, "stream ended early"):
                            next(reply)
                    context.__exit__.assert_called_once()
