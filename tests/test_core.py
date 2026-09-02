import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import config, consistency, json_utils, storage, validator  # noqa: E402
from core.lm_client import LMClient, AuthRequired  # noqa: E402
from core.embedding_cache import EmbeddingCache  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


class JsonUtilsTests(unittest.TestCase):
    def test_extract_from_code_fence(self):
        payload = '```json\n{"a": 1}\n```'
        self.assertEqual(json_utils.extract_json_object(payload), {"a": 1})

    def test_extract_from_plain_text(self):
        payload = 'Result: {"a": 2, "b": [1,2]}'
        self.assertEqual(json_utils.extract_json_object(payload), {"a": 2, "b": [1, 2]})

    def test_strips_think_blocks(self):
        payload = '<think>hidden</think>{"k": "v"}'
        self.assertEqual(json_utils.extract_json_object(payload), {"k": "v"})

    def test_returns_none_on_plain_text(self):
        self.assertIsNone(json_utils.extract_json_object("just prose"))

    def test_cosine_similarity_identical(self):
        v = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(json_utils.cosine_similarity(v, v), 1.0)

    def test_cosine_similarity_zero_for_empty(self):
        self.assertEqual(json_utils.cosine_similarity([], [1, 2]), 0.0)

    def test_remove_forbidden_tags(self):
        text = "a young blonde woman, slim figure, dark hair, urban street"
        out = json_utils.remove_forbidden_tags(text, config.FORBIDDEN_TAGS)
        self.assertNotIn("blonde", out.lower())
        self.assertNotIn("slim", out.lower())
        self.assertNotIn("young", out.lower())


class StorageTests(unittest.TestCase):
    def test_load_or_init_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(storage.load_or_init(Path(tmp) / "x.json"), [])

    def test_save_and_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            data = [{"name": "01_test", "prompt": "x", "negative_prompt": "y"}]
            storage.save(path, data)
            loaded = storage.load_or_init(path)
            self.assertEqual(loaded, data)

    def test_next_number(self):
        self.assertEqual(storage.next_number([]), 1)
        self.assertEqual(storage.next_number([{"name": "03_foo"}]), 4)
        self.assertEqual(storage.next_number([{"name": "no_number"}]), 1)

    def test_next_number_avoids_collision(self):
        data = [{"name": "01_a"}, {"name": "02_b"}, {"name": "01_c"}]
        self.assertEqual(storage.next_number(data), 3)

    def test_find_duplicate_by_name(self):
        data = [{"name": "01_x", "prompt": "p1"}]
        self.assertEqual(storage.find_duplicate(data, name="01_x"), "name")
        self.assertIsNone(storage.find_duplicate(data, name="02_y"))

    def test_find_duplicate_by_prompt(self):
        data = [{"name": "01_x", "prompt": "p1"}]
        self.assertEqual(storage.find_duplicate(data, prompt="p1"), "prompt")
        self.assertIsNone(storage.find_duplicate(data, prompt="p2"))

    def test_find_duplicate_ignores_empty_strings(self):
        data = [{"name": "01_x", "prompt": "p1"}]
        self.assertIsNone(storage.find_duplicate(data, name=""))
        self.assertIsNone(storage.find_duplicate(data, prompt="  "))

    def test_find_duplicate_normalizes_whitespace(self):
        data = [{"name": "01_x", "prompt": "foo, bar, baz"}]
        self.assertEqual(storage.find_duplicate(data, prompt="foo,  bar,   baz"), "prompt")
        self.assertEqual(storage.find_duplicate(data, prompt="  foo, bar, baz\n"), "prompt")
        self.assertIsNone(storage.find_duplicate(data, prompt="foo,bar,baz"))


class ValidatorTests(unittest.TestCase):
    def test_assemble_positive_uses_context_token(self):
        out = validator.assemble_positive({
            "subject": "",
            "pose": "leaning against railing",
            "environment": "rooftop at dusk",
            "relationships": "subject centered, city skyline behind",
            "lighting": "warm sodium lamps",
            "camera": "35mm low angle",
            "state": "jacket slipping off shoulder",
        })
        self.assertIn(config.DEFAULT_CONTEXT_TOKEN, out)
        self.assertIn("leaning against railing", out)
        self.assertIn("subject centered, city skyline behind", out)

    def test_assemble_positive_strips_quality_bait(self):
        out = validator.assemble_positive({
            "subject": "masterpiece portrait",
            "pose": "looking at viewer",
            "environment": "studio",
            "lighting": "soft light",
            "camera": "85mm",
            "state": "8k ultra detailed",
        })
        low = out.lower()
        self.assertNotIn("masterpiece", low)
        self.assertNotIn("8k", low)
        self.assertNotIn("ultra detailed", low)

    def test_remove_quality_bait_function(self):
        out = validator.remove_quality_bait(
            "studio portrait, masterpiece, best quality, 8k, sharp focus"
        )
        self.assertNotIn("masterpiece", out.lower())
        self.assertNotIn("best quality", out.lower())
        self.assertNotIn("8k", out.lower())

    def test_assemble_negative_default_empty(self):
        neg = validator.assemble_negative({
            "lighting": "soft window light",
            "environment": "park during golden hour",
        })
        self.assertIn("deformed", neg)

    def test_assemble_negative_night_extras(self):
        neg = validator.assemble_negative({
            "lighting": "soft moonlight",
            "environment": "forest clearing",
        })
        self.assertIn("daylight", neg)
        self.assertIn("urban", neg)

    def test_assemble_negative_indoor_extras(self):
        neg = validator.assemble_negative({
            "lighting": "lamp",
            "environment": "small bedroom",
        })
        self.assertIn("outdoor", neg)

    def test_assemble_negative_uses_separate_forbidden_list(self):
        from unittest.mock import patch
        with patch("core.validator.NEGATIVE_BASE", "woman, young"), \
             patch("core.validator.FORBIDDEN_TAGS_NEGATIVE", ()):
            neg = validator.assemble_negative({
                "lighting": "soft light",
                "environment": "bedroom",
            })
            self.assertIn("woman", neg.lower())
            self.assertIn("young", neg.lower())

    def test_validate_result_short_prompt(self):
        err = validator.validate_result({"prompt": "x", "negative_prompt": "y"})
        self.assertIsNotNone(err)

    def test_assemble_negative_includes_standard_artifacts(self):
        neg = validator.assemble_negative({
            "lighting": "soft window light",
            "environment": "park during golden hour",
        })
        for tag in ["text", "watermark", "jpeg artifacts", "low quality", "extra digits"]:
            self.assertIn(tag, neg)

    def test_clean_field_deduplicates_prose(self):
        out = validator._clean_field("woman and woman standing.")
        self.assertEqual(out, "woman, and, standing")

    def test_assemble_positive_deduplicates_cross_field(self):
        out = validator.assemble_positive({
            "subject": "fashion model",
            "pose": "standing",
            "state": "standing",
            "environment": "rooftop",
        })
        self.assertEqual(out.count("standing"), 1)

    def test_assemble_positive_subject_without_token_prepends_token(self):
        out = validator.assemble_positive({
            "subject": "fashion photographer in silk dress",
            "pose": "leaning",
            "environment": "rooftop",
        })
        self.assertTrue(out.startswith(config.DEFAULT_CONTEXT_TOKEN))
        self.assertIn("fashion photographer", out)

    def test_assemble_positive_subject_with_token_keeps_it(self):
        out = validator.assemble_positive({
            "subject": "{prompt} in silk dress",
            "pose": "leaning",
        })
        self.assertTrue(out.startswith(config.DEFAULT_CONTEXT_TOKEN))
        self.assertIn("in silk dress", out)


class ConsistencyTests(unittest.TestCase):
    def test_valid_environment_ok(self):
        env = {
            "location": "small park bench",
            "time_of_day": "evening",
            "weather": "light rain",
            "props": "espresso machine",
        }
        self.assertIsNone(consistency.validate_environment(env))

    def test_invalid_time_of_day(self):
        env = {"location": "kitchen", "time_of_day": "midnight", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("time_of_day", err)

    def test_location_time_leak_rejected(self):
        env = {"location": "small apartment kitchen at night", "time_of_day": "evening", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("time-of-day", err)

    def test_location_weather_leak_rejected(self):
        env = {"location": "forest in the rain", "time_of_day": "morning", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("weather", err)

    def test_day_with_night_weather(self):
        env = {"location": "rooftop", "time_of_day": "noon", "weather": "moonlight"}
        err = consistency.validate_environment(env)
        self.assertIsNotNone(err)

    def test_night_with_day_weather(self):
        env = {"location": "alley", "time_of_day": "night", "weather": "bright midday sun"}
        err = consistency.validate_environment(env)
        self.assertIsNotNone(err)

    def test_noon_with_blizzard(self):
        env = {"location": "street", "time_of_day": "noon", "weather": "heavy snow blizzard"}
        err = consistency.validate_environment(env)
        self.assertIsNotNone(err)

    def test_evening_with_rain_ok(self):
        env = {"location": "park", "time_of_day": "evening", "weather": "light rain"}
        self.assertIsNone(consistency.validate_environment(env))

    def test_location_nighttime_leak_rejected(self):
        env = {"location": "rooftop nighttime", "time_of_day": "evening", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("time-of-day", err)

    def test_location_rainy_leak_rejected(self):
        env = {"location": "forest rainy", "time_of_day": "morning", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("weather", err)

    def test_location_sunny_leak_rejected(self):
        env = {"location": "beach sunny", "time_of_day": "night", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("weather", err)

    def test_location_at_golden_hour_leak_rejected(self):
        env = {"location": "balcony at golden hour", "time_of_day": "evening", "weather": "clear"}
        err = consistency.validate_environment(env)
        self.assertIn("time-of-day", err)


class LMClientAuthTests(unittest.TestCase):
    def _fake_response(self, status_code: int, json_payload=None):
        response = type("R", (), {})()
        response.status_code = status_code
        response.raise_for_status = lambda: None
        response.json = lambda: json_payload if json_payload is not None else {"models": []}
        return response

    def test_no_token_no_header(self):
        client = LMClient(token="")
        self.assertEqual(client._headers(), {})

    def test_token_sets_bearer_header(self):
        client = LMClient(token="abc123")
        self.assertEqual(client._headers()["Authorization"], "Bearer abc123")

    def test_401_raises_auth_required(self):
        import unittest.mock as mock
        client = LMClient()
        with mock.patch("core.lm_client.requests") as fake_requests:
            fake_resp = self._fake_response(401)
            fake_requests.request.return_value = fake_resp
            with self.assertRaises(AuthRequired):
                client.list_models_meta()

    def test_set_token_resets_attempt_flag(self):
        client = LMClient()
        client._token_attempted = True
        client.set_token("xyz")
        self.assertFalse(client._token_attempted)
        self.assertEqual(client.token, "xyz")

    def test_probe_embedding_model_returns_true_when_vectors_present(self):
        import unittest.mock as mock
        client = LMClient()
        with mock.patch("core.lm_client.requests") as fake_requests:
            response = type("R", (), {})()
            response.status_code = 200
            response.raise_for_status = lambda: None
            response.json = lambda: {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            fake_requests.request.return_value = response
            self.assertTrue(client.probe_embedding_model())

    def test_probe_embedding_model_returns_false_on_empty(self):
        import unittest.mock as mock
        client = LMClient()
        with mock.patch("core.lm_client.requests") as fake_requests:
            response = type("R", (), {})()
            response.status_code = 200
            response.raise_for_status = lambda: None
            response.json = lambda: {"data": []}
            fake_requests.request.return_value = response
            self.assertFalse(client.probe_embedding_model())

    def test_probe_embedding_model_returns_false_on_error(self):
        import unittest.mock as mock
        client = LMClient()
        with mock.patch("core.lm_client.requests") as fake_requests:
            fake_requests.request.side_effect = Exception("boom")
            self.assertFalse(client.probe_embedding_model())

    def test_unload_all_models_calls_endpoint(self):
        import unittest.mock as mock
        from core.model_info import ModelInfo
        info = ModelInfo(
            key="m1", type="llm", architecture="qwen3vl",
            quantization="Q4_K_S", bits_per_weight=4,
            max_context_length=4096, params_string="4B", size_bytes=1,
            vision=False, tool_use=True,
            loaded_context_length=25088,
        )
        client = LMClient()
        client.list_models_meta = MagicMock(return_value=[info])
        with mock.patch("core.lm_client.requests") as fake_requests:
            response = type("R", (), {})()
            response.status_code = 200
            response.raise_for_status = lambda: None
            response.json = lambda: {"instance_id": "m1"}
            fake_requests.request.return_value = response
            count = client.unload_all_models()
        self.assertEqual(count, 1)
        body = fake_requests.request.call_args.kwargs["json"]
        self.assertEqual(body["instance_id"], "m1")

    def test_chat_uses_v1_payload_shape(self):
        import unittest.mock as mock
        client = LMClient()
        client.get_model_info = MagicMock(return_value=None)
        response = type("R", (), {})()
        response.status_code = 200
        response.raise_for_status = lambda: None
        response.json = lambda: {
            "output": [{"type": "message", "content": "hello"}],
            "stats": {"input_tokens": 1, "total_output_tokens": 1},
        }
        with mock.patch("core.lm_client.requests") as fake_requests:
            fake_requests.request.return_value = response
            result = client.chat("m", "sys", "usr", temperature=0.5, max_tokens=100)
        self.assertEqual(result, "hello")
        called = fake_requests.request.call_args
        self.assertEqual(called.args[0], "POST")
        self.assertTrue(called.args[1].endswith("/chat"))
        body = called.kwargs["json"]
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["system_prompt"], "sys")
        self.assertEqual(body["input"], "usr")
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["max_output_tokens"], 100)
        self.assertNotIn("messages", body)
        self.assertNotIn("max_tokens", body)

    def test_chat_applies_sampling_preset_when_info_available(self):
        import unittest.mock as mock
        from core.model_info import ModelInfo
        client = LMClient()
        info = ModelInfo(
            key="m",
            type="llm",
            architecture="qwen3vl",
            quantization="Q4_K_S",
            bits_per_weight=4,
            max_context_length=8192,
            params_string="4B",
            size_bytes=1,
            vision=False,
            tool_use=True,
            reasoning_allowed=["off", "on"],
            reasoning_default="on",
            loaded_context_length=None,
        )
        client.get_model_info = MagicMock(return_value=info)
        response = type("R", (), {})()
        response.status_code = 200
        response.raise_for_status = lambda: None
        response.json = lambda: {"output": [{"type": "message", "content": "x"}]}
        with mock.patch("core.lm_client.requests") as fake_requests:
            fake_requests.request.return_value = response
            client.chat("m", "sys", "usr", max_tokens=10)
        body = fake_requests.request.call_args.kwargs["json"]
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["repeat_penalty"], 1.05)
        self.assertEqual(body["reasoning"], "off")

    def test_models_cache_fetched_once(self):
        import unittest.mock as mock
        client = LMClient()
        response = type("R", (), {})()
        response.status_code = 200
        response.raise_for_status = lambda: None
        response.json = lambda: {
            "models": [
                {"type": "llm", "key": "m1", "max_context_length": 1024,
                 "capabilities": {"vision": False, "trained_for_tool_use": False}},
            ]
        }
        with mock.patch("core.lm_client.requests") as fake_requests:
            fake_requests.request.return_value = response
            client.list_models_meta()
            client.list_models_meta()
            client.get_model_info("m1")
        self.assertEqual(fake_requests.request.call_count, 1)

    def test_models_cache_invalidated_after_load(self):
        import unittest.mock as mock
        client = LMClient()
        empty = type("R", (), {})()
        empty.status_code = 200
        empty.raise_for_status = lambda: None
        empty.json = lambda: {"models": []}

        load_resp = type("R", (), {})()
        load_resp.status_code = 200
        load_resp.raise_for_status = lambda: None
        load_resp.json = lambda: {"status": "loaded"}

        with mock.patch("core.lm_client.requests") as fake_requests:
            fake_requests.request.side_effect = [empty, load_resp, empty]
            client.list_models_meta()
            client.load_model("m1", context_length=1024)
            client.list_models_meta()
        self.assertEqual(fake_requests.request.call_count, 3)

    def test_query_embed_is_cached(self):
        import unittest.mock as mock
        client = LMClient()
        cache = EmbeddingCache()

        with mock.patch.object(client, "embed", side_effect=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5]],
            [],
            [],
        ]) as emb:
            client.max_similarity_with_cache("q", ["a", "b"], cache)
            client.max_similarity_with_cache("q", ["a", "b"], cache)
        self.assertEqual(emb.call_count, 2)


if __name__ == "__main__":
    unittest.main()
