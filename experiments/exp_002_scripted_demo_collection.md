# Experiment 002: Scripted Demo Collection

## Command

```bash
python scripts/collect_scripted_demos.py \
  --task obstacle_avoid_pick \
  --num-episodes 4 \
  --max-steps 50 \
  --output data/demos/debug_obstacle_avoid_pick.npz \
  --mock-env \
  --seed 0
```

## Generated Files

- `data/demos/debug_obstacle_avoid_pick.npz`
- `data/demos/debug_obstacle_avoid_pick.json`

## Dataset Schema

Each step stores proprioception, contact, language, deterministic language feature,
morphology, JSON action, flattened action vector, reward, done, success, task name,
phase, episode id, and step id.

## Mock Physics Limits

The mock environment is deterministic and headless. It only approximates tip motion,
grasping, pushing, and grasper rotation, so it is useful for pipeline tests rather
than physics claims.

## Next Step

Replace `--mock-env` with `--env feagine_mujoco` after the Feagine rollout policy
and scene state extraction are stable.
