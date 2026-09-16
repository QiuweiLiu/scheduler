#!/usr/bin/env python3
import unittest

from tracing.analysis.workload_v02_simulator import GPU, Job, Node, Template, choose_action, plan_gpu_admission


def make_node(model: str, resident_mb: float) -> Node:
    return Node(
        node_id="node",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id=model,
        runtime_ms=100.0,
        load_ms=0.0,
        workspace_peak_mb=19430.0,
        resident_model_mb=resident_mb,
        status="success",
    )


class AdmissionTests(unittest.TestCase):
    def test_new_model_admission_evicts_before_ledger_check(self) -> None:
        gpu = GPU(
            index=0,
            capacity_mb=32760.0,
            resident={"Qwen3-4B": 7600.0, "Qwen2.5-VL-3B-Instruct": 7100.0},
        )
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        admitted, evicted, model_mb, workspace_mb, total_mb = plan_gpu_admission(
            gpu, node, {"memory_p95_mb": 19430.0}
        )
        self.assertTrue(admitted)
        self.assertEqual(evicted, ("Qwen2.5-VL-3B-Instruct", "Qwen3-4B"))
        self.assertEqual(model_mb, 17000.0)
        self.assertEqual(workspace_mb, 2430.0)
        self.assertEqual(total_mb, 19430.0)
        self.assertEqual(
            gpu.resident,
            {"Qwen3-4B": 7600.0, "Qwen2.5-VL-3B-Instruct": 7100.0},
        )

    def test_resident_target_does_not_evict_itself(self) -> None:
        gpu = GPU(
            index=0,
            capacity_mb=20000.0,
            resident={"Qwen3-VL-8B-Instruct": 17000.0, "Qwen3-4B": 7600.0},
        )
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        admitted, evicted, _, _, total_mb = plan_gpu_admission(
            gpu, node, {"memory_p95_mb": 19430.0}
        )
        self.assertTrue(admitted)
        self.assertEqual(evicted, ("Qwen3-4B",))
        self.assertEqual(total_mb, 19430.0)
        self.assertIn("Qwen3-VL-8B-Instruct", gpu.resident)

    def test_model_larger_than_gpu_is_oom_without_eviction_plan(self) -> None:
        gpu = GPU(
            index=0,
            capacity_mb=16000.0,
            resident={"Qwen3-4B": 7600.0},
        )
        node = make_node("Qwen3-VL-8B-Instruct", 17000.0)
        admitted, evicted, _, _, total_mb = plan_gpu_admission(
            gpu, node, {"memory_p95_mb": 19430.0}
        )
        self.assertFalse(admitted)
        self.assertEqual(evicted, ())
        self.assertGreater(total_mb, gpu.capacity_mb)
        self.assertEqual(gpu.resident, {"Qwen3-4B": 7600.0})


    def test_myopic_joint_action_selects_shorter_ready_node(self) -> None:
        def make_job(job_id: str, node_id: str, model_id: str) -> Job:
            node = Node(
                node_id=node_id,
                sequence_index=0,
                predecessors=(),
                successors=(),
                lane="gpu",
                model_id=model_id,
                runtime_ms=100.0,
                load_ms=0.0,
                workspace_peak_mb=1100.0,
                resident_model_mb=1000.0,
                status="success",
            )
            template = Template(
                template_id=job_id,
                video_id="video",
                split="train",
                baseline="test",
                nodes=(node,),
                by_id={node_id: node},
            )
            return Job(
                job_instance_id=job_id,
                template=template,
                arrival_ms=0.0,
                deadline_ms=None,
                service_class="normal",
                node_state={node_id: "ready"},
            )

        slow = make_job("job-slow", "node-slow", "slow-model")
        fast = make_job("job-fast", "node-fast", "fast-model")
        stats = {
            "slow-model|gpu|0|exact": {
                "runtime_p50_ms": 100.0,
                "runtime_p90_ms": 120.0,
                "load_p50_ms": 0.0,
                "memory_p95_mb": 1100.0,
                "count": 3,
            },
            "fast-model|gpu|0|exact": {
                "runtime_p50_ms": 10.0,
                "runtime_p90_ms": 20.0,
                "load_p50_ms": 0.0,
                "memory_p95_mb": 1100.0,
                "count": 3,
            },
        }
        items = [(0.0, 0.0, 0, "node-slow"), (0.0, 0.0, 1, "node-fast")]
        selected, gpu_index, _, candidate_count, feasible_count = choose_action(
            "myopic",
            items,
            [slow, fast],
            [GPU(index=0, capacity_mb=32760.0)],
            stats,
            0,
        )
        self.assertEqual(selected[3], "node-fast")
        self.assertEqual(gpu_index, 0)
        self.assertEqual(candidate_count, 2)
        self.assertEqual(feasible_count, 2)

    def test_round_robin_joint_action_keeps_oldest_ready_node(self) -> None:
        node_a = make_node("model-a", 1000.0)
        node_b = make_node("model-b", 1000.0)
        template_a = Template("job-a", "video", "train", "test", (node_a,), {"node": node_a})
        template_b = Template("job-b", "video", "train", "test", (node_b,), {"node": node_b})
        job_a = Job("job-a", template_a, 0.0, None, "normal", {"node": "ready"})
        job_b = Job("job-b", template_b, 0.0, None, "normal", {"node": "ready"})
        stats = {
            "model-a|gpu|0|exact": {"runtime_p50_ms": 100.0, "runtime_p90_ms": 100.0, "load_p50_ms": 0.0, "memory_p95_mb": 1100.0, "count": 3},
            "model-b|gpu|0|exact": {"runtime_p50_ms": 1.0, "runtime_p90_ms": 1.0, "load_p50_ms": 0.0, "memory_p95_mb": 1100.0, "count": 3},
        }
        selected, _, _, candidate_count, _ = choose_action(
            "round_robin",
            [(0.0, 0.0, 0, "node"), (0.0, 0.0, 1, "node")],
            [job_a, job_b],
            [GPU(index=0, capacity_mb=32760.0)],
            stats,
            0,
        )
        self.assertEqual(selected[2], 0)
        self.assertEqual(candidate_count, 2)


if __name__ == "__main__":
    unittest.main()
