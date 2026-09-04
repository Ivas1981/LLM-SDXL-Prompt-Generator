import unittest
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.lm_client import LMClient
from core.pipeline import load_system_prompts, _format_user_hint, _build_parts
from core import storage


class PipelineTests(unittest.TestCase):
    def test_load_system_prompts(self):
        prompts_dir = ROOT / "prompts"
        prompts = load_system_prompts(prompts_dir)
        self.assertEqual(len(prompts), 9)
        self.assertIn("step1_concept.txt", prompts)
        self.assertIn("step7_post_process.txt", prompts)

    def test_load_system_prompts_sfw_replaces_state_prompt(self):
        prompts_dir = ROOT / "prompts"
        with unittest.mock.patch("core.pipeline.NSFW", False):
            prompts = load_system_prompts(prompts_dir)
        self.assertIn("step4_state.txt", prompts)
        self.assertIn("state", prompts["step4_state.txt"])
        self.assertNotIn("nudity:", prompts["step4_state.txt"])

    def test_load_system_prompts_sfw_keeps_sfw_assemble(self):
        prompts_dir = ROOT / "prompts"
        with unittest.mock.patch("core.pipeline.NSFW", False):
            prompts = load_system_prompts(prompts_dir)
        self.assertIn("step7_assemble.txt", prompts)
        self.assertIn("no nudity", prompts["step7_assemble.txt"].lower())

    def test_load_system_prompts_nsfw_replaces_assemble_prompt(self):
        prompts_dir = ROOT / "prompts"
        nsfw_file = prompts_dir / "step7_assemble_nsfw.txt"
        if not nsfw_file.exists():
            self.skipTest("step7_assemble_nsfw.txt missing")
        with unittest.mock.patch("core.pipeline.NSFW", True):
            prompts = load_system_prompts(prompts_dir)
        self.assertIn("step7_assemble.txt", prompts)
        self.assertIn("nudity", prompts["step7_assemble.txt"].lower())
        self.assertIn('"nudity"', prompts["step7_assemble.txt"])
        self.assertNotIn("state = physical/emotional condition + scene-motivated nudity", prompts["step7_assemble.txt"].lower())

    def test_step2_time_of_day_matches_valid_times(self):
        content = (ROOT / "prompts" / "step2_environment.txt").read_text(encoding="utf-8")
        from core.consistency import VALID_TIMES
        for t in VALID_TIMES:
            self.assertIn(t, content)

    def test_format_user_hint_replaces_names(self):
        from core.pipeline import _format_user_hint, STEP_USER_HINT
        step_name = "step1_concept.txt"
        ctx = {}
        result = _format_user_hint(step_name, ctx, ["old1", "old2"])
        self.assertIn("old1", result)
        self.assertIn("old2", result)

    def test_format_user_hint_handles_braces_in_context(self):
        from core.pipeline import _format_user_hint
        step_name = "step2_environment.txt"
        ctx = {"step1_concept": "cyberpunk {neon} city"}
        result = _format_user_hint(step_name, ctx, [])
        self.assertIn("cyberpunk {neon} city", result)
        self.assertNotIn("{step1_concept}", result)

    def test_safe_name_cleans_string(self):
        from core.pipeline import _safe_name
        self.assertEqual(_safe_name("Hello World! 123"), "hello_world_123")
        self.assertEqual(_safe_name(""), "scene")

    def test_step4_user_hint_nsfw_includes_nudity(self):
        with unittest.mock.patch("core.pipeline.NSFW", True):
            hint = _format_user_hint("step4_state.txt", {"step1_concept": "c", "step2_environment": "e", "step3_pose": "p"}, [])
        self.assertIn("partial or full nudity", hint)

    def test_step3_pose_hint_nsfw_includes_minimal_clothing(self):
        with unittest.mock.patch("core.pipeline.NSFW", True):
            hint = _format_user_hint("step3_pose.txt", {"step1_concept": "c", "step2_environment": "e"}, [])
        self.assertIn("minimal remaining clothing", hint)
        self.assertIn("removed or displaced", hint)

    def test_step3_pose_hint_sfw_has_full_clothing(self):
        with unittest.mock.patch("core.pipeline.NSFW", False):
            hint = _format_user_hint("step3_pose.txt", {"step1_concept": "c", "step2_environment": "e"}, [])
        self.assertIn("Generate clothing and pose", hint)
        self.assertNotIn("minimal remaining clothing", hint)

    def test_step4_user_hint_sfw_excludes_nudity(self):
        with unittest.mock.patch("core.pipeline.NSFW", False):
            hint = _format_user_hint("step4_state.txt", {"step1_concept": "c", "step2_environment": "e", "step3_pose": "p"}, [])
        self.assertIn("physical state and natural expression", hint)
        self.assertNotIn("nudity", hint)

    def test_build_parts_sfw_ignores_nudity(self):
        parsed = {"state": "", "nudity": "bare chest"}
        with unittest.mock.patch("core.pipeline.NSFW", False):
            parts = _build_parts(parsed)
        self.assertEqual(parts["state"], "")

    def test_build_parts_nsfw_uses_nudity_fallback(self):
        parsed = {"state": "", "nudity": "bare chest"}
        with unittest.mock.patch("core.pipeline.NSFW", True):
            parts = _build_parts(parsed)
        self.assertEqual(parts["state"], "")
        self.assertEqual(parts["nudity"], "bare chest")

    def test_generate_batch_adds_entries(self):
        import unittest.mock as mock
        from core.lm_client import LMClient
        from core.pipeline import generate_batch

        lm = LMClient()
        prompts = {f"step{i}_concept.txt": f"prompt{i}" for i in range(1, 9)}
        prompts["step2_environment.txt"] = "env"
        prompts["step3_pose.txt"] = "pose"
        prompts["step4_state.txt"] = "state"
        prompts["step5_lighting.txt"] = "light"
        prompts["step6_camera.txt"] = "cam"
        prompts["step7_assemble.txt"] = '{"subject": "x", "pose": "y", "state": "z", "environment": "w", "relationships": "r", "lighting": "l", "camera": "c"}'
        prompts["step8_name.txt"] = "test_scene"

        fake_result = {
            "prompt": "test prompt",
            "negative_prompt": "neg",
            "_parts": {"subject": "x", "pose": "y", "state": "z", "environment": "w", "relationships": "r", "lighting": "l", "camera": "c"},
            "_raw_assembled": {"prompt": "", "negative_prompt": ""},
            "_name_raw": "test_scene",
            "_scene_name": "test_scene",
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with mock.patch("core.pipeline.run_pipeline", return_value=(fake_result, None)):
                with mock.patch.object(lm, "max_similarity_with_cache", return_value=(0.0, None)):
                    added = generate_batch(
                        lm=lm,
                        model="m",
                        prompts=prompts,
                        data=[],
                        target_count=1,
                        on_progress=None,
                        save_path=tmp_path,
                    )
            self.assertEqual(added, 1)
            data = storage.load_or_init(tmp_path)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["name"], "m")
            self.assertEqual(data[0]["prompt"], "")
            self.assertEqual(data[0]["negative_prompt"], "")
            self.assertEqual(data[1]["name"], "01_test_scene")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_post_process_with_local_model_updates_prompts(self):
        from core.pipeline import _post_process_with_local_model
        lm = unittest.mock.Mock()
        lm.chat.return_value = '{"positive": "a, b", "negative": "c, d"}'
        pos, neg = _post_process_with_local_model(lm, "model", "system", "{prompt}, a, a, b", "c, c, d")
        self.assertEqual(pos, "{prompt}, a, b")
        self.assertEqual(neg, "c, d")

    def test_post_process_with_local_model_preserves_context_token_when_missing(self):
        from core.pipeline import _post_process_with_local_model
        from core import config
        lm = unittest.mock.Mock()
        lm.chat.return_value = '{"positive": "a, b", "negative": "c, d"}'
        pos, neg = _post_process_with_local_model(lm, "model", "system", "{prompt}, a, b", "c, d")
        self.assertIn(config.DEFAULT_CONTEXT_TOKEN, pos)
        self.assertEqual(neg, "c, d")

    def test_generate_batch_skips_duplicate_model_marker(self):
        import unittest.mock as mock
        from core.lm_client import LMClient
        from core.pipeline import generate_batch

        lm = LMClient()
        prompts = {f"step{i}_concept.txt": f"prompt{i}" for i in range(1, 9)}
        prompts["step2_environment.txt"] = "env"
        prompts["step3_pose.txt"] = "pose"
        prompts["step4_state.txt"] = "state"
        prompts["step5_lighting.txt"] = "light"
        prompts["step6_camera.txt"] = "cam"
        prompts["step7_assemble.txt"] = '{"subject": "x", "pose": "y", "state": "z", "environment": "w", "relationships": "r", "lighting": "l", "camera": "c"}'
        prompts["step8_name.txt"] = "test_scene"

        fake_result = {
            "prompt": "test prompt",
            "negative_prompt": "neg",
            "_parts": {"subject": "x", "pose": "y", "state": "z", "environment": "w", "relationships": "r", "lighting": "l", "camera": "c"},
            "_raw_assembled": {"prompt": "", "negative_prompt": ""},
            "_name_raw": "test_scene",
            "_scene_name": "test_scene",
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with mock.patch("core.pipeline.run_pipeline", return_value=(fake_result, None)):
                with mock.patch.object(lm, "max_similarity_with_cache", return_value=(0.0, None)):
                    added = generate_batch(
                        lm=lm,
                        model="existing-model",
                        prompts=prompts,
                        data=[{"name": "existing-model", "prompt": "", "negative_prompt": ""}],
                        target_count=1,
                        on_progress=None,
                        save_path=tmp_path,
                    )
            self.assertEqual(added, 1)
            data = storage.load_or_init(tmp_path)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["name"], "existing-model")
            self.assertEqual(data[1]["name"], "01_test_scene")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
