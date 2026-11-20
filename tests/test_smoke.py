"""Offline checks for the core app paths."""
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from receptionist.classifier import classify
from receptionist.config import load_config
from receptionist.metrics import MetricsTracker
from receptionist.router import route_query


class CoreSmokeTests(unittest.TestCase):
    def test_classifier_respects_changed_threshold(self):
        prompt = "Count these items?"
        def classify_at(threshold):
            return classify(prompt, {
                "classification": {"thresholds": {"escalate_score_threshold": threshold}},
            }).decision
        self.assertEqual(classify_at(1), "local")
        self.assertEqual(classify_at(0), "cloud")

    def test_cost_survives_history_eviction_and_matches_baseline(self):
        key = "openai/test"
        tracker = MetricsTracker({key: {"input_per_1k": 1, "output_per_1k": 2}}, key)
        for _ in range(201):
            tracker.record("cloud", key, 300, 200, estimated=False)
        metrics = tracker.get_metrics()
        self.assertEqual(len(metrics["history"]), 200)
        self.assertEqual(metrics["cloud_tokens"], 100500)
        self.assertEqual(metrics["actual_cost"], Decimal("140.7"))
        self.assertEqual(metrics["savings"], 0)
        self.assertEqual(tracker.cloud_budget_reason(140.7, key), "session cloud budget reached")

    def test_auto_routes_to_configured_providers(self):
        with patch("receptionist.router.LocalModel") as local, patch("receptionist.router.CloudModel") as cloud:
            self.assertEqual(route_query("hello", {})[0], "local")
            cloud.assert_not_called()
            self.assertEqual(route_query("Explain what a contract is.", {})[0], "cloud")
            local.return_value.generate.assert_called_once()
            cloud.return_value.generate.assert_called_once()

    def test_chat_local_failure_and_one_shot_cloud(self):
        config = load_config("config.yaml")
        config["api_keys"] = {"openai": "test-key"}
        config["cloud_budget_usd"] = None
        prompt = "Explain what a contract is."
        answer = "    first line\n        second line"
        usage = SimpleNamespace(prompt_tokens=6, completion_tokens=4)
        for streaming in (True, False):
            with self.subTest(streaming=streaming), \
                    patch("receptionist.config.load_config", return_value=config), \
                    patch("receptionist.models.local.subprocess.run") as ollama, \
                    patch("receptionist.models.cloud.OpenAIClient") as openai:
                create = openai.return_value.chat.completions.create
                if streaming:
                    create.return_value = iter([
                        SimpleNamespace(choices=[SimpleNamespace(
                            finish_reason="stop", delta=SimpleNamespace(content=answer),
                        )]),
                        SimpleNamespace(choices=[], usage=usage),
                    ])
                else:
                    create.return_value = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                        finish_reason="stop", message=SimpleNamespace(content=answer),
                    )])
                app = AppTest.from_file("streamlit_app.py").run()
                self.assertEqual(app.radio[0].value, "Local Only")
                app.checkbox[1].set_value(streaming).run()
                ollama.side_effect = FileNotFoundError()
                app.chat_input[0].set_value(prompt).run()
                self.assertFalse(app.exception)
                self.assertIn("Request failed", app.error[0].value)
                openai.assert_not_called()

                app.checkbox[0].check().run()
                app.chat_input[0].set_value(prompt).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.checkbox[0].value)
                messages = app.session_state.messages
                self.assertEqual(messages[-1]["classification"]["reason"], "manual override")
                self.assertEqual(messages[-1]["content"], answer)
                self.assertEqual(messages[-1]["details"]["provider"], "openai")
                self.assertGreaterEqual(messages[-1]["details"]["seconds"], 0)
                self.assertEqual(messages[-1]["details"]["cloud_cost"], Decimal("0.0015"))

                ollama.side_effect = None
                ollama.return_value = SimpleNamespace(returncode=0, stdout=answer.encode())
                app.chat_input[0].set_value(prompt).run()
                self.assertFalse(app.exception)
                messages = app.session_state.messages
                self.assertEqual(messages[-1]["model"], "local")
                self.assertEqual(messages[-1]["content"], answer)
                create.assert_called_once()
                sent = create.call_args.kwargs['messages']
                self.assertEqual([m['role'] for m in sent], ['system', 'user'])
                self.assertEqual(sent[-1]['content'], prompt)
                captions = [caption.value for caption in app.caption]
                self.assertTrue(any('Each reply uses only the current prompt' in caption for caption in captions))
                self.assertTrue(any('Estimated cloud cost: $0.001500' in caption for caption in captions))
                self.assertEqual(len(app.chat_message), 6)
                self.assertEqual(app.session_state.usage_metrics.get_metrics()["cloud_tokens"], 10)
