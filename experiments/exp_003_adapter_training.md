# Experiment 003: Adapter Training

## Command

```bash
python scripts/train_adapter.py \
  --demo data/demos/debug_obstacle_avoid_pick.npz \
  --epochs 3 \
  --batch-size 8 \
  --output outputs/checkpoints/adapter_debug.pt \
  --metrics-output outputs/metrics/adapter_debug.json
```

## Generated Files

- `outputs/checkpoints/adapter_debug.pt`
- `outputs/metrics/adapter_debug.json`

## Data Schema

The adapter consumes deterministic language features, proprioception, contact state,
and morphology vectors. It predicts the flattened Feagine action vector using
`section_angles`, `grip_command`, and `grasper_rotation`.

## Current Limits

The language encoder is a deterministic hash stub, and the dataset comes from mock
rollouts. The result validates the pipeline, not real VLM or real robot performance.

## Next Step

Use the same `.npz` schema with Feagine MuJoCo rollouts once real scene/object/contact
state extraction is available.
