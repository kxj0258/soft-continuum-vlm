# Experiment 004: Deterministic VLM Planner Debug

## Command

```bash
python scripts/run_vlm_planner_demo.py \
  --task obstacle_avoid_pick \
  --language "绕过黑色障碍物，轻轻抓住蓝色圆柱" \
  --mock-env \
  --max-steps 60 \
  --output outputs/rollouts/vlm_planner_debug.json
```

## Purpose

Validate the deterministic VLM planner stub. It converts Chinese or English
language into structured target objects, avoidance constraints, gentle-contact
limits, and subgoals.

## Current Limit

This is not a real VLM. It is a deterministic rule parser for pipeline testing.
Mock-env results must not be used as final paper evidence.

## Next Step

Replace `DeterministicVLMPlanner.plan()` with a real VLM-backed planner while
keeping the same `BasePlanner` interface and output schema.
