# Milestone 6/8 Design: Deterministic Planner and Evaluation

## Goal

Implement the first offline, deterministic version of Milestone 6 and Milestone 8.
The system must run on `--mock-env`, avoid model downloads, preserve the Feagine
action schema, and export JSON / CSV / PNG artifacts suitable for pipeline
debugging and paper draft tables.

## Architecture

- `planners/`: convert language, observation, and task name into structured
  subgoals and safety constraints.
- `controllers/vlm_planner_controller.py`: compose the deterministic planner,
  existing scripted/PCC action path, and `SafetyProjector`.
- `evaluation/`: run mock rollouts for baselines, compute metrics, and export
  paper-draft plots/tables.
- `scripts/`: provide command-line entry points for planner demo, baseline
  evaluation, and paper figure export.

## Data Flow

```text
language + observation
  -> DeterministicVLMPlanner.plan()
  -> VlmPlannerController.act()
  -> MockContinuumEnv.step()
  -> metrics / rollout logs / plots
```

## Constraints

- Do not touch `../feagine_simulation` or `../feagine-simulation`.
- Do not download OpenVLA, Octo, VLM, or VLA weights.
- Keep action fields as `section_angles`, `grip_command`, and `grasper_rotation`.
- Mark all outputs as mock-env debug, not final paper claims.

## Testing

Add unit and smoke tests for planner parsing, planner-controller info logging,
metric aggregation, baseline evaluation JSON/CSV output, and figure export.
