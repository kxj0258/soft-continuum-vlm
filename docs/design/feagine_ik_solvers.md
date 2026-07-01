# Feagine IK Solver Design

## Boundary

`src/soft_continuum_vlm/controllers/ik/` defines a pure kinematics boundary:

```text
target_tip_position
current_tip_position
current_section_angles
        ↓
IkSolver.solve(...)
        ↓
IkResult
```

The solver does not read task phases and does not generate `grip_command` or
`grasper_rotation`. `FeagineActionAdapter` will combine these concerns in M4.

## Result semantics

- `success` means the solver produced a finite, bounded command that is safe to apply.
- `converged` means the predicted position is within `position_tolerance`.
- `status` distinguishes `converged`, `step_applied`, `no_progress`,
  `max_iterations`, and `fallback_hold`.
- Failed solves return the original section angles rather than a partially converged command.

This distinction is required because differential IK normally returns one useful
small step before the final target has converged.

## Solvers

`PccIkSolver` performs iterative damped least-squares updates and is intended for
larger target changes. `DifferentialIkSolver` performs exactly one bounded
Jacobian step and is intended for the 4D action interface's small Cartesian
increments.

Both implementations currently use the deterministic approximate PCC functions
from `continuum_kinematics.py`. They anchor model displacement at the measured
current tip position, so model absolute-position bias does not directly shift
the requested target.

## Failure fallback

`solve_with_retries()` applies the requested Cartesian delta at successively
smaller scales. The default sequence is `1.0`, `0.5`, and `0.25`. If every
attempt fails, it returns `fallback_hold` with the unchanged section angles.

## Integration limitation

The current approximate PCC backend is deterministic and testable but is not a
calibrated replacement for Feagine/MuJoCo kinematics. M4 may inject a calibrated
solver without changing the `IkSolver` or `IkResult` contract.
