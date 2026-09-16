from __future__ import annotations

from scripts.repair_canonical_activity_labels import repair_rows


def test_repair_recanonicalizes_legacy_frame_tools_without_touching_raw_fields() -> None:
    rows, summary = repair_rows([
        {
            "prefix_id": "run:semantic:1",
            "prefix_raw_actions": ["image-grid-selector"],
            "prefix_activities": ["other"],
            "target_next_raw_action": "patch-zoomer",
            "target_next_activity": "other",
            "state_features": {
                "state_present": True,
                "prefix": {
                    "last_actions": ["other"],
                    "action_histogram": {"other": 1},
                },
            },
            "future_events_included_in_input": False,
            "ground_truth_included_in_input": False,
            "source_trace_sha256": "trace-hash",
        },
    ])
    assert rows[0]["prefix_activities"] == ["sample_seek"]
    assert rows[0]["target_next_activity"] == "spatial_qa"
    assert rows[0]["prefix_raw_actions"] == ["image-grid-selector"]
    assert rows[0]["state_features"]["prefix"]["action_histogram"] == {"sample_seek": 1}
    assert summary["mapping_counts"] == {
        "image-grid-selector->sample_seek": 1,
        "patch-zoomer->spatial_qa": 1,
    }
