"""Frozen-workload profile contract (evidence-based, train-only).

The v0.4.1 workload templates carry the per-node role (``node_type``) and the
source-trace GPU model on every GPU node, but they do not carry
``workload_shape`` or an episode-level ``gpu_identity``.  The measured profile
family is keyed by (model, role) and by a measured GPU class, so this module
derives those fields from the template evidence itself:

* ``node.workload_shape := node.role`` for GPU nodes of models covered by the
  profiles; a covered-model GPU node whose role is outside the measured strata
  raises -- the contract must cover every stratum the frozen workload contains;
* ``episode.gpu_identity :=`` the uniform node-level ``gpu_model`` evidence;
  mixed, absent, or class-mismatched evidence raises.

Nothing here reads future truth: roles and the GPU model come from the frozen
template rows, and the strata sizes come from the frozen token priors
(``request_preemption_profile_v1`` medians) that the calibration probes used.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from tracing.analysis.workload_v02_simulator import Template, gpu_identity_class

_PROFILE_KEYS = ("colocation_profile", "prefetch_interference",
                 "request_preemption", "batching_engine")


def covered_strata(extension_config: Mapping[str, Any] | None) -> dict[str, set[str]]:
    """Return model -> measured shapes from the co-location/interference cells."""

    covered: dict[str, set[str]] = {}
    for key in ("colocation_profile", "prefetch_interference"):
        profile = (extension_config or {}).get(key) or {}
        for cell in profile.get("cells") or []:
            if not isinstance(cell, Mapping):
                continue
            if key == "colocation_profile":
                for model_field, shape_field in (("model_a", "shape_a"), ("model_b", "shape_b")):
                    model = str(cell.get(model_field) or "")
                    shape = str(cell.get(shape_field) or "")
                    if model and shape:
                        covered.setdefault(model, set()).add(shape)
            else:
                model = str(cell.get("infer_model") or "")
                shape = str(cell.get("infer_shape") or "")
                if model and shape:
                    covered.setdefault(model, set()).add(shape)
    return covered


def apply_real_workload_profile_contract(
    templates: Mapping[str, Template],
    episodes: Iterable[Mapping[str, Any]],
    extension_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Template], list[dict[str, Any]]]:
    """Derive ``workload_shape`` / ``gpu_identity`` from frozen template evidence."""

    covered = covered_strata(extension_config)
    if not covered:
        raise ValueError("profile contract requires covered strata in the extension config")

    evidence = {
        str(node.gpu_model).strip()
        for template in templates.values()
        for node in template.nodes
        if node.lane == "gpu" and node.gpu_model
    }
    if len(evidence) != 1:
        raise ValueError(
            "profile contract requires one uniform node gpu_model evidence, got "
            f"{sorted(evidence)}"
        )
    identity = evidence.pop()
    expected_classes = {
        gpu_identity_class(str(((extension_config or {}).get(key) or {}).get("gpu_identity") or ""))
        for key in _PROFILE_KEYS
    } - {""}
    if expected_classes and {gpu_identity_class(identity)} != expected_classes:
        raise ValueError(
            f"profile contract GPU evidence {identity!r} does not match the profile "
            f"identities {sorted(expected_classes)}"
        )

    new_templates: dict[str, Template] = {}
    for template_id, template in templates.items():
        changed = False
        nodes = []
        for node in template.nodes:
            if node.lane == "gpu" and node.model_id in covered:
                if not node.role:
                    raise ValueError(
                        f"profile contract: covered node {node.node_id} has no role"
                    )
                if node.role not in covered[node.model_id]:
                    raise ValueError(
                        f"profile contract: stratum {node.model_id}|{node.role} has no "
                        "measured cell"
                    )
                nodes.append(replace(node, workload_shape=node.role))
                changed = True
            else:
                nodes.append(node)
        if changed:
            new_nodes = tuple(nodes)
            new_templates[template_id] = replace(
                template, nodes=new_nodes, by_id={node.node_id: node for node in new_nodes}
            )
        else:
            new_templates[template_id] = template

    new_episodes: list[dict[str, Any]] = []
    for episode in episodes:
        existing = episode.get("gpu_identity") or episode.get("gpu_model")
        if existing and gpu_identity_class(str(existing)) != gpu_identity_class(identity):
            raise ValueError(
                f"profile contract: episode {episode.get('episode_id')!r} declares "
                f"{existing!r}, evidence is {identity!r}"
            )
        new_episodes.append({**dict(episode), "gpu_identity": identity})
    return new_templates, new_episodes
