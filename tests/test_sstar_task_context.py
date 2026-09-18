from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocess"))

import sstar_task_context as tc  # noqa: E402


class EnrichSampleContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry_row = {
            "video_id": "56yT3H_DjVE",
            "domain": "Sports Competition",
            "official_task_type": "Action Recognition",
            "sub_category": "Esports",
            "question_id": "482-1",
            "answer": "B",
            "options": ["A", "B"],
            "question": "which animal?",
            "duration": "medium",
        }

    def test_restores_only_the_three_registry_fields(self) -> None:
        manifest = {"video_id": "56yT3H_DjVE", "run_id": "r1", "baseline": "star"}
        enriched, audit = tc.enrich_sample_context(manifest, self.registry_row, mask=False)

        self.assertEqual(enriched["domain"], "Sports Competition")
        self.assertEqual(enriched["official_task_type"], "Action Recognition")
        self.assertEqual(enriched["sub_category"], "Esports")
        self.assertEqual(sorted(audit["recovered_fields"]), ["domain", "official_task_type", "sub_category"])
        self.assertEqual(audit["question_id"], "482-1")
        # the original manifest is untouched; the sample row never carries registry payload
        self.assertNotIn("domain", manifest)
        for forbidden in ("question", "answer", "options", "duration", "question_id"):
            self.assertNotIn(forbidden, enriched)
        # provenance only: question_id is allowed, the task payload is not
        for forbidden in ("question", "answer", "options", "duration"):
            self.assertNotIn(forbidden, audit)
        self.assertEqual(audit["question_id"], "482-1")

    def test_existing_unknown_values_are_filled(self) -> None:
        manifest = {"video_id": "v", "domain": "unknown", "official_task_type": "", "sub_category": None}
        enriched, _ = tc.enrich_sample_context(manifest, self.registry_row, mask=False)
        self.assertEqual(enriched["domain"], "Sports Competition")
        self.assertEqual(enriched["official_task_type"], "Action Recognition")
        self.assertEqual(enriched["sub_category"], "Esports")

    def test_agreeing_existing_values_are_kept(self) -> None:
        manifest = {"video_id": "v", "domain": "Sports Competition"}
        enriched, audit = tc.enrich_sample_context(manifest, self.registry_row, mask=False)
        self.assertEqual(enriched["domain"], "Sports Competition")
        self.assertNotIn("domain", audit["recovered_fields"])

    def test_conflicting_existing_value_is_fatal(self) -> None:
        manifest = {"video_id": "v", "domain": "Knowledge"}
        with self.assertRaises(ValueError):
            tc.enrich_sample_context(manifest, self.registry_row, mask=False)

    def test_masked_control_writes_unknown_and_records_it(self) -> None:
        manifest = {"video_id": "v", "domain": "Sports Competition"}
        enriched, audit = tc.enrich_sample_context(manifest, self.registry_row, mask=True)
        for field in tc.TASK_CONTEXT_FIELDS:
            self.assertEqual(enriched[field], "unknown")
        self.assertEqual(sorted(audit["masked_fields"]), ["domain", "official_task_type", "sub_category"])
        self.assertEqual(audit["recovered_fields"], [])

    def test_unjoined_run_is_left_alone(self) -> None:
        manifest = {"video_id": "u", "domain": "unknown"}
        enriched, audit = tc.enrich_sample_context(manifest, None, mask=False)
        self.assertEqual(enriched["domain"], "unknown")
        self.assertFalse(audit["registry_joined"])


class RegistryTests(unittest.TestCase):
    def test_duplicate_video_id_is_fatal(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.jsonl"
            path.write_text('{"video_id":"a"}\n{"video_id":"a"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                tc.load_task_registry(path)

    def test_registry_sha_and_lookup(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.jsonl"
            path.write_text('{"video_id":"a","domain":"D"}\n', encoding="utf-8")
            registry, sha = tc.load_task_registry(path)
            self.assertEqual(registry["a"]["domain"], "D")
            self.assertEqual(len(sha), 64)

    def test_none_registry_is_empty(self) -> None:
        registry, sha = tc.load_task_registry(None)
        self.assertEqual(registry, {})
        self.assertIsNone(sha)

    def test_allowlist_reading(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "videos.txt"
            path.write_text("a\nb\n\n", encoding="utf-8")
            self.assertEqual(tc.read_video_allowlist(path), {"a", "b"})
        self.assertIsNone(tc.read_video_allowlist(None))


class PilotSelectionTests(unittest.TestCase):
    def test_selection_is_order_independent_and_hashed(self) -> None:
        videos = [f"v{index}" for index in range(40)]
        first = tc.deterministic_pilot_videos(videos, 16)
        shuffled = list(reversed(videos))
        self.assertEqual(first, tc.deterministic_pilot_videos(shuffled, 16))
        self.assertEqual(len(first), 16)
        self.assertEqual(len(set(first)), 16)


if __name__ == "__main__":
    unittest.main()
