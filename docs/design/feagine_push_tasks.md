# Feagine Push Tasks

## Scope

M6a adds three state-based contact tasks:

- `feagine_push_left_to_right`
- `feagine_push_right_to_left`
- `feagine_contact_push`

They reuse the M5 `FeagineMetaWorldEnv` and therefore expose only the normalized
4D action.

## State machine

```text
approach -> push -> complete
```

The approach phase ends when the tip is within `approach_threshold` of the
object. Completion requires object-goal distance below the success threshold,
at least one attributed target-object contact during the episode, contact force
below its limit, and penetration below its limit. This prevents unrelated
collisions or externally moved objects from being counted as successful pushes.

## Reward

```text
- tip_object_distance
- object_goal_distance
- contact_force_weight * max_contact_force
- penetration_weight * max_penetration
+ success_bonus
```

The global maximum force is penalized even when it came from an unrelated
collision. Per-contact names additionally produce target-contact and
wrong-contact counts so evaluation can distinguish intentional pushing from
accidental scene collisions.

## Expert

`FeaginePushExpert` first targets a pre-contact point behind the object relative
to the object-to-goal direction. In the push phase it targets the goal directly.
It always emits an open-gripper 4D action and never generates section angles or
grasper rotation.

## Integration limitation

The task code evaluates existing object/contact observations; it does not add or
move MuJoCo objects. Scene object names, collision geometry, friction, force
limits, and goal coordinates require real-scene verification before these tasks
can be treated as simulation baselines.
