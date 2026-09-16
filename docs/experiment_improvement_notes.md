# Experiment Design Improvement Notes

Recorded 2026-08-20 after R8-P4 completion. These notes document limitations
of the current experiment design and specific improvements for future work.

## 1. GPU Count: 2 → 4-8

### Current
```python
TOPOLOGIES = ((32760.0, 32760.0), (24576.0, 24576.0), (32760.0, 24576.0))
```
Only 2 GPUs. The scheduling space is too small for sophisticated methods to show advantage.

### Suggested
```python
TOPOLOGIES = (
    (32760.0, 32760.0, 32760.0, 32760.0),           # 4× homogeneous
    (24576.0, 24576.0, 24576.0, 24576.0),           # 4× small
    (32760.0, 32760.0, 24576.0, 24576.0),           # 4× mixed
    (32760.0,) * 8,                                  # 8× homogeneous
    (24576.0,) * 8,                                  # 8× small
    (32760.0,) * 4 + (24576.0,) * 4,                # 8× mixed
)
```

### Impact
- More candidate pool diversity (ready_nodes × free_gpus)
- CP-SAT solver has more meaningful optimization space
- GPU cache eviction decisions become more complex
- Requires regenerating all workload episodes

### File
`src/tracing/workloads/build_workload.py`, line 27 (`TOPOLOGIES`)

## 2. Deadline Tightness: 1.5-3× → 1.1-1.3×

### Current
```python
deadline_multiplier = (1.5, 2.0, 3.0)[episode_index % 3]
```
Deadlines are 1.5-3× estimated runtime. Very generous; deadline miss rate ~11%.
Strategy differences barely affect deadline miss rate.

### Suggested
```python
deadline_multiplier = (1.1, 1.2, 1.3)[episode_index % 3]
```

### Impact
- Expected deadline miss rate: 30-40%
- Strategy differences become visible in deadline miss rate
- More pressure on scheduling decisions
- Requires regenerating all workload episodes

### File
`src/tracing/workloads/build_workload.py`, line 238 (`deadline_multiplier`)

## 3. Unified Objective Function

### Current Problem
Three methods use three different objectives:

| Method | Objective | Problem |
|---|---|---|
| PredOpt-v2 | `0.25*priority + runtime/100k + future*1.0/100k + 0.05*memory` | Hand-tuned weights, no documentation |
| CP-RHO | Same as PredOpt, CP-SAT optimization | Same objective, no added value |
| RL | `-(completion/100k + 0.5*deadline + 0.01*evictions + 5*failed)` | Different from PredOpt, hard to compare |

### Suggested: Unified Scoring for PredOpt/CP-RHO

Replace the hand-tuned weights with a principled cost function:

```python
# Current (arbitrary weights)
PREDOPT_V2_PRIORITY_WEIGHT = 0.25
PREDOPT_V2_FUTURE_WEIGHT = 1.0
PREDOPT_V2_MEMORY_WEIGHT = 0.05

# Suggested (principled cost)
score = runtime_ms
      + load_ms_if_cache_miss
      + future_cost_fraction * runtime_ms
      + eviction_penalty * num_evictions
```

Where:
- `runtime_ms`: direct node execution time (completion time contribution)
- `load_ms_if_cache_miss`: model loading time when cache miss (0 if hit)
- `future_cost_fraction`: `future_cost / (future_cost + runtime_ms)` — proportional impact
- `eviction_penalty`: penalty for evicting models from GPU memory

### Suggested: RL Reward Tuning

```python
# Current
reward = -(completion/100000 + 0.5*deadline + 0.01*evictions + 5*failed)

# Suggested (increase deadline weight)
reward = -(completion/100000 + 1.0*deadline + 0.01*evictions + 10*failed)
```

### Files
- `src/tracing/analysis/workload_v02_simulator.py`, lines 56-58 (PredOpt weights)
- `src/tracing/scheduling/cp_rho.py`, `solve_first_action` (CP-RHO objective)
- `scripts/r8_bc_ppo.py`, `episode_reward` (RL reward)
- `scripts/r7_rl_train_eval.py`, `reward` (RL reward)

## 4. Academic Comparison Context

### Literature reference
- **Murakkab (OSDI'26)**: Agentic workflow orchestration on cloud GPUs. Uses declarative abstractions + profile-guided optimization. Reduces GPU usage by 2.8×.
- **Agent.xpu (arXiv'25)**: Agentic LLM scheduling on heterogeneous SoCs. Uses flow-aware NPU-iGPU coordination. 1.2-4.9× proactive throughput improvement.
- **Hydra (TPDS'23)**: Deadline-aware DL training scheduling on 45 heterogeneous GPUs. Deadline 1.2-2.0×, tight mode.
- **PriorityDRL (JPDC'26)**: Dual-agent deadline-aware GPU scheduling. Deadline 1.1-1.3×.

### Key differences from current work
| Dimension | Current | Academic typical | Impact |
|---|---|---|---|
| GPU count | 2 | 8-64 | Scheduling space too small |
| Deadline | 1.5-3× | 1.1-1.3× | Not tight enough |
| Objective | 3 different ones | Unified | Unfair comparison |
| Workload type | Video agent DAG | DL training / LLM inference | Novel but niche |
| Node granularity | ms-level | min-hour level | 1000× finer |

## 5. Regeneration Procedure

To apply these changes, the full workflow is:

1. Modify `build_workload.py`: update `TOPOLOGIES` and `deadline_multiplier`
2. Regenerate workload episodes:
   ```bash
   # S_train (20,000 episodes)
   PYTHONPATH=src python scripts/build_r6_causal_v2.py ... --split train
   # S_val (1,000 episodes)
   PYTHONPATH=src python scripts/build_r6_causal_v2.py ... --split validation
   ```
3. Re-run P0-P3 (BC data collection, training, evaluation)
4. Re-run P4 (PPO training)
5. Re-run P5 (T_final evaluation)

Note: This invalidates all current experimental results. The current results
stand as valid for the 2-GPU, loose-deadline, hand-tuned-weight setup.