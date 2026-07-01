# Feagine Action Adapter

## Responsibility

`FeagineActionAdapter` is the only conversion boundary between the normalized
top-level action and the Feagine low-level command:

```text
[dx, dy, dz, gripper_control]
        ↓
scale task-space delta
        ↓
current tip + delta = target tip
        ↓
IK with reduced-delta retries
        ↓
section_angles
```

In parallel, it maps gripper intent and computes the hidden grasper rotation
from deterministic task context.

## Command model

`FeagineLowLevelCommand` uses the semantic fields:

```text
section_angles
grasper_rotation
gripper_open_close
```

`to_runtime_action()` translates only the final field to the existing runtime
name `grip_command`. Top-level tasks and policies must not emit either
`section_angles` or `grasper_rotation`.

## Gripper mapping

The mapping is linear and deterministic:

```text
-1.0 -> 0.0 -> fully open
+1.0 -> 1.0 -> fully closed
```

Intermediate values remain continuous.

## Orientation rules

- No task context: hold the current grasper rotation.
- `approach` / `pregrasp`: point toward the supplied orientation target.
- `grasp` / `align_grasper`: align with the object principal axis, with target
  direction as a fallback.
- `place` / `align_place`: align with the place or target principal axis.
- `transport`, `lift`, and `retract`: hold the current rotation.

An explicit `desired_grasper_rotation` is accepted only through trusted task
context; it is not part of the 4D policy action.

## Failure behavior

The adapter uses the M3 reduced-delta retry sequence. If all IK attempts fail,
the command holds the current section angles while still applying the bounded
gripper and deterministic orientation commands. The structured conversion
result retains `IkResult` for future metrics and debugging.
