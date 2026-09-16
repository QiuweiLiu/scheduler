import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_p9d_topology_dataset import (
    FEATURE_SCHEMA_VERSION,
    LABEL_SCHEMA_VERSION,
    V3_LABEL_POLICY,
    audit_model_input,
    build_chain,
    build_dataset,
    build_graph,
    future_chain_layers,
    future_layers,
    make_feature,
    make_label,
    make_label_v3,
)


def event(event_id, event_type, node_type, step_id, parents=None, action=None, model_id=None):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "node_type": node_type,
        "step_id": step_id,
        "parent_step_ids": [] if parents is None else parents,
        "action": action,
        "model_id": model_id,
    }


class P9dTopologyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            event("run:1", "run", "run_control", 0, model_id="stack-a"),
            event("api:2", "api_call", "planner", 1, action="planner.generate", model_id="planner-a"),
            event("action:3", "action", "videotool_temporal", 2, parents=[1], action="frame-selector", model_id="cpu-a"),
            event("action:4", "action", "videotool_temporal", 2, parents=[1], action="image-grid-selector", model_id="cpu-a"),
            event("api:5", "api_call", "answer_generation", 3, parents=[2], action="generalist.generate", model_id="answer-a"),
        ]
        self.graph = build_graph(self.events)
        self.anchor = {
            "sample_id": "run::api:2",
            "video_id": "video-a",
            "registry_video_id": "video-a",
            "run_id": "run",
            "current_node_id": "api:2",
            "split": "train",
            "sample_row": {
                "baseline": "langgraph_react",
                "model_stack_id": "stack-a",
                "planner_model_id": "planner-a",
                "domain": "domain-a",
                "question_type": "temporal",
                "required_modalities": ["scene"],
            },
        }

    def test_parent_step_reconstruction_preserves_width(self):
        layers = future_layers(self.graph, "api:2", horizon=5)
        self.assertEqual(layers, [["action:3", "action:4"], ["api:5"]])
        self.assertEqual(self.graph["stats"]["missing_parent_refs"], 0)
        self.assertEqual(self.graph["stats"]["forward_parent_refs"], 0)

    def test_run_control_anchor_is_joinable(self):
        self.assertEqual(future_layers(self.graph, "run:1", horizon=1), [["api:2"]])

    def test_feature_has_no_future_or_execution_truth(self):
        feature = make_feature(self.anchor, self.events, 1, {"trace_sha256": "hash-a"})
        self.assertEqual(feature["schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(audit_model_input(feature["model_input"]), [])
        self.assertNotIn("video_id", feature["model_input"])
        self.assertNotIn("runtime_ms", feature["model_input"])
        self.assertNotIn("status", feature["model_input"])

    def test_label_keeps_identity_only_on_label_side(self):
        label = make_label(self.anchor, self.graph, {"trace_sha256": "hash-a"}, horizon=5)
        self.assertEqual(label["schema_version"], LABEL_SCHEMA_VERSION)
        self.assertEqual(label["label_summary"]["layer_widths"], [2, 1])
        self.assertEqual(label["future_layers"][0]["nodes"][0]["predecessor_node_ids"], ["api:2"])
        self.assertEqual(label["future_layers"][0]["nodes"][0]["execution_lane"], "unknown")

    def test_step_parent_expands_to_last_event_only(self):
        events = [
            event("run:1", "run", "run_control", 0, model_id="stack-a"),
            event("api:2", "api_call", "planner", 1, action="planner.generate", model_id="planner-a"),
            event("action:3", "action", "videotool_temporal", 1, action="frame-selector", model_id="cpu-a"),
            event("api:4", "api_call", "planner", 2, parents=[1], action="planner.generate", model_id="planner-a"),
        ]
        graph = build_graph(events)
        # action:3 chains to api:2; api:4 sees parent step 1 -> last event action:3 only.
        self.assertEqual(graph["predecessors"]["action:3"], ["api:2"])
        self.assertEqual(graph["predecessors"]["api:4"], ["action:3"])
        self.assertGreaterEqual(graph["stats"]["step_parent_dropped_edges"], 1)

    def test_future_layers_use_longest_path_depth(self):
        events = [
            event("run:1", "run", "run_control", 0, model_id="stack-a"),
            event("api:2", "api_call", "planner", 1, action="planner.generate", model_id="planner-a"),
            event("action:3", "action", "videotool_temporal", 1, action="frame-selector", model_id="cpu-a"),
            event("api:4", "api_call", "planner", 2, parents=[1], action="planner.generate", model_id="planner-a"),
        ]
        graph = build_graph(events)
        layers = future_layers(graph, "api:2", horizon=5)
        # action:3 at depth 1; api:4 at depth 2 (not merged into one BFS shell).
        self.assertEqual(layers, [["action:3"], ["api:4"]])

    def test_workload_scale_and_termination_contract(self):
        events = [
            dict(
                self.events[1],
                input={"parameters": {"start_time": 100, "end_time": 150, "query": "what"}, "nested_api_call_count": 2},
            ),
            self.events[2],
        ]
        graph = build_graph(
            [event("run:1", "run", "run_control", 0, model_id="stack-a")] + events
        )
        node = graph["graph_nodes"]["api:2"]
        self.assertEqual(node["workload_scale"]["clip_len"], 50.0)
        self.assertEqual(node["workload_scale"]["query_char_len"], 4)
        self.assertEqual(node["workload_scale"]["nested_api_call_count"], 2)
        label = make_label(self.anchor, self.graph, {"trace_sha256": "hash-a"}, horizon=5)
        self.assertIn(label["label_summary"]["termination_status"], ("terminated", "censored"))
        self.assertEqual(
            label["label_summary"]["layer_widths_capped_4plus"],
            [min(width, 4) for width in label["label_summary"]["layer_widths"]],
        )

    def test_end_to_end_split_outputs_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "groups": {
                            "P_dev": {"subsplit": {"train": ["video-a"], "validation": [], "test": []}},
                            "P_holdout_diag": {"video_ids": ["video-h"]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            dev_samples = root / "dev.jsonl"
            dev_samples.write_text(
                json.dumps(
                    {
                        "video_id": "video-a",
                        "run_id": "run-a",
                        "source_event_id": "run-a:api:2",
                        "source": "core",
                        "split": "train",
                        "source_trace_sha256": "hash-a",
                        "domain": "domain-a",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            holdout_samples = root / "holdout.jsonl"
            holdout_samples.write_text(
                json.dumps(
                    {
                        "video_id": "video-h",
                        "run_id": "run-h",
                        "source_event_id": "run-h:run:1",
                        "source": "final_holdout_v1",
                        "source_split": "final_holdout_v1",
                        "split": "validation",
                        "source_trace_sha256": "hash-h",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_root = root / "raw"
            for run_id, video_id, trace_hash in (("run-a", "video-a", "hash-a"), ("run-h", "video-h", "hash-h")):
                run_dir = raw_root / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "run_manifest.json").write_text(
                    json.dumps({"run_id": run_id, "status": "success", "trace_sha256": trace_hash}),
                    encoding="utf-8",
                )
                trace_events = self.events if run_id == "run-a" else [self.events[0], self.events[1]]
                (run_dir / "trace.jsonl").write_text(
                    "".join(json.dumps({**event, "event_id": f"{run_id}:{event['event_id']}"}) + "\n" for event in trace_events),
                    encoding="utf-8",
                )
            output = root / "output"
            manifest = build_dataset(
                behavior_role_samples=dev_samples,
                holdout_role_samples=holdout_samples,
                video_registry=registry,
                raw_roots=[raw_root],
                output_root=output,
            )
            self.assertEqual(manifest["counts"]["rows_by_split"]["train"], 1)
            self.assertEqual(manifest["counts"]["rows_by_split"]["holdout"], 1)
            self.assertTrue((output / "features_train.jsonl.gz").is_file())
            self.assertTrue((output / "labels_holdout.jsonl.gz").is_file())
            # Cross-platform reproducibility: JSONL must use LF, not CRLF.
            with gzip.open(output / "labels_train.jsonl.gz", "rb") as handle:
                first_line = handle.readline()
            self.assertTrue(first_line.endswith(b"\n"))
            self.assertFalse(first_line.endswith(b"\r\n"))

    def _v3_events(self):
        def make(event_id, event_type, node_type, step_id, parents=None, action=None, model_id=None, inp=None, status="success", retry_of=None):
            payload = {
                "event_id": event_id,
                "event_type": event_type,
                "node_type": node_type,
                "step_id": step_id,
                "parent_step_ids": [] if parents is None else parents,
                "action": action,
                "model_id": model_id,
                "status": status,
                "retry_of": retry_of,
            }
            if inp is not None:
                payload["input"] = inp
            return payload

        planner_ok = lambda tool_name: {"parse_status": "json", "parsed_decision": {"tool_name": tool_name}}
        return [
            make("run:1", "run", "run_control", 0, action="baseline_start", model_id="stack-a"),
            make("api:2", "api_call", "planner", 1, action="planner.generate", model_id="planner-a", inp=planner_ok("frame-selector")),
            make(
                "action:3",
                "action",
                "videotool_temporal",
                1,
                action="frame-selector",
                model_id="cpu-a",
                inp={"standard_tool_call": {"id": "compat-react-step-1", "name": "FrameSelector"}},
            ),
            make("api:4", "api_call", "planner", 2, parents=[1], action="planner.generate", model_id="planner-a", inp=planner_ok("summarization-tool")),
            make(
                "api:5",
                "api_call",
                "answer_generation",
                2,
                parents=[1],
                action="generalist.generate",
                model_id="answer-a",
                inp={"raw_model_output": "Answer:"},
            ),
            make(
                "action:6",
                "action",
                "videotool_generalist",
                2,
                parents=[1],
                action="summarization-tool",
                model_id="cpu-a",
                inp={"standard_tool_call": {"id": "compat-react-step-2", "name": "Summarizer"}},
            ),
            make("api:7", "api_call", "planner", 3, parents=[2], action="planner.generate", model_id="planner-a", status="error"),
            make(
                "api:8",
                "api_call",
                "planner",
                3,
                parents=[2],
                action="planner.generate",
                model_id="planner-a",
                inp=planner_ok("frame-selector"),
                retry_of="api:7",
            ),
            make(
                "action:9",
                "action",
                "videotool_temporal",
                3,
                parents=[2],
                action="frame-selector",
                model_id="cpu-a",
                inp={"standard_tool_call": {"id": "compat-react-step-3", "name": "FrameSelector"}},
            ),
            make(
                "api:10",
                "api_call",
                "answer_generation",
                3,
                parents=[2],
                action="generalist.generate",
                model_id="answer-a",
                inp={"raw_model_output": "D"},
            ),
            make("run:11", "run", "answer_generation", 4, parents=[3], action="answer", model_id="answer-a"),
        ]

    def test_v3_chain_merges_nested_and_recovers_edges(self):
        chain = build_chain(self._v3_events(), "langgraph_react")
        self.assertEqual(chain["violations"], [])
        self.assertEqual(
            chain["nodes"],
            ["api:2", "action:3", "api:4", "action:6", "api:7", "api:8", "action:9", "api:10", "run:11"],
        )
        self.assertEqual(chain["predecessors"]["action:3"], ["api:2"])  # planner -> tool
        self.assertEqual(chain["predecessors"]["api:8"], ["api:7"])  # retry -> failed attempt
        summarize_label = chain["node_labels"]["action:6"]
        self.assertTrue(summarize_label["merged_nested_call"])
        self.assertEqual([call["event_id"] for call in summarize_label["nested_calls"]], ["api:5"])
        self.assertEqual(summarize_label["resource_signature"]["nested_model_class"], "answer-a")
        self.assertTrue(summarize_label["resource_applicable"])
        self.assertFalse(chain["node_labels"]["run:11"]["resource_applicable"])
        self.assertTrue(chain["node_labels"]["api:8"]["is_retry"])
        self.assertEqual(chain["node_labels"]["api:7"]["status_class"], "error")

    def test_v3_future_layers_are_compute_only_and_width_one(self):
        chain = build_chain(self._v3_events(), "langgraph_react")
        label = make_label_v3(
            {"sample_id": "s", "video_id": "v", "registry_video_id": "v", "run_id": "run", "split": "train", "current_node_id": "api:2"},
            chain,
            {"trace_sha256": "hash"},
            horizon=5,
        )
        self.assertEqual(label["reconstruction_policy"], V3_LABEL_POLICY)
        self.assertEqual(label["label_summary"]["termination_status"], "censored")
        self.assertEqual(label["label_summary"]["remaining_future_compute_nodes"], 7)
        self.assertEqual(label["label_summary"]["layer_widths"], [1, 1, 1, 1, 1])
        self.assertTrue(label["label_summary"]["width_is_deterministic"])
        self.assertNotIn("run:11", [node["node_id"] for layer in label["future_layers"] for node in layer["nodes"]])
        layers, remaining = future_chain_layers(chain, "api:4", horizon=5)
        self.assertEqual(remaining, 5)
        self.assertEqual(len(layers), 5)

    def test_v3_seriality_gate_collects_violations(self):
        events = self._v3_events()
        events.append(
            {
                "event_id": "action:12",
                "event_type": "action",
                "node_type": "videotool_temporal",
                "step_id": 3,
                "parent_step_ids": [2],
                "action": "image-grid-selector",
                "model_id": "cpu-a",
                "status": "success",
                "input": {"standard_tool_call": {"id": "compat-react-step-3", "name": "ImageGridSelect"}},
            }
        )
        chain = build_chain(events, "langgraph_react")
        self.assertTrue(any("multi-tool step" in violation for violation in chain["violations"]))

    def test_v3_parent_step_jump_is_a_violation(self):
        events = self._v3_events()
        events[3]["parent_step_ids"] = [0]
        chain = build_chain(events, "langgraph_react")
        self.assertTrue(any("non-canonical parent_step_ids" in violation for violation in chain["violations"]))


if __name__ == "__main__":
    unittest.main()
