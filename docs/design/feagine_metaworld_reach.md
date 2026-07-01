# Feagine MetaWorld Reach Tasks

## Architecture

M5 introduces a new task boundary without rewriting the existing low-level
Feagine and mock environments:

```text
4D policy / expert action
        ↓
FeagineMetaWorldEnv
        ↓
FeagineActionAdapter
        ↓
runtime action
        ↓
FeagineMujocoEnv or another low-level backend
        ↓
post-step observation
        ↓
FeagineMetaWorldTask reward / success / info
```

The backend reward is retained only as backend diagnostic state. The public
reward and success condition come from the new task object.

## Task contract

Every new task implements:

```text
reset_task()
compute_reward()
compute_success()
get_goal()
get_object_state()
get_task_info()
```

`get_task_context()` provides trusted deterministic context to the hidden
grasper-orientation controller.

## Reach tasks

- `feagine_reach_left`: fixed left-side goal.
- `feagine_reach_right`: fixed right-side goal.
- `feagine_reach_3d`: seeded random goal inside conservative bounds.

All three use:

```text
reward = -||tip - goal||
success = ||tip - goal|| < success_threshold
```

The default goals are relative to the current vertical-base research layout.
They must be replaced by the M2 simulation-verified workspace coordinates before
the physical scene is frozen.

## Metrics

Each wrapper step reports the current tip-goal distance, complete episode
distance history, structured IK result, cumulative IK success rate, semantic
low-level command, runtime action, and backend diagnostics.

## Scripted expert

`FeagineReachExpert` emits:

```text
clip((goal - tip) / delta_xyz_scale, -1, 1) + gripper_open
```

It has no access to section angles or grasper rotation and therefore exercises
the same 4D interface intended for RL, VLM skills, and VLA policies.
