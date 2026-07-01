# soft-continuum-vlm

`soft-continuum-vlm` is a research codebase for soft continuum robot manipulation, centered on Feagine/MuJoCo experiments and deterministic baselines before any real VLM/VLA integration.

The current repository is not a generic software demo. It is a reproducible robotics research scaffold for:

- Feagine MuJoCo runtime inspection and control.
- Structured observations for soft continuum manipulation.
- Deterministic task planners, controllers, scripted experts, and safety projection.
- Mock-environment training/evaluation plumbing for the Soft Embodiment Adapter.
- Real Feagine tabletop scene generation, contact diagnostics, reachable red-object pick diagnostics, and viewer demonstrations.

Real VLM, OpenVLA, Octo, VLA adapter integration, large model downloads, and paper-level real-data training are intentionally not active milestones yet.

## Current Status

The project currently has two layers of functionality.

### Stable Research Scaffold

- `MockContinuumEnv` for CI/debug runs.
- `FeagineMujocoEnv` wrapper for real Feagine/MuJoCo runtime entry points.
- Structured observation schema with `rgb`, `depth`, `language`, `proprioception`, `robot_state`, `objects`, and `contact`.
- Current low-level Feagine runtime command schema:

```python
action = {"section_angles": [...], "grip_command": 0.0, "grasper_rotation": 0.0}
```

- Approximate PCC kinematics, local Jacobian calibration, `PccIkController`, `TaskPhaseExpert`, and `SafetyProjector`.
- Deterministic VLM planner stub for language-to-subgoal parsing.
- Adapter model, adapter policy wrapper, rollout/evaluation pipeline, and figure export commands.

### Current Real Feagine Tabletop Work

The current active work is to converge the real Feagine MuJoCo environment before learning or VLM/VLA work.

Implemented real-scene tools include:

- Generate tabletop MJCF scenes, including a reachable red-object variant.
- Inspect scene body/geom/contact state.
- Validate red-object settling and tabletop contacts.
- Scan Feagine reachability and grasper-red contact.
- Diagnose controlled gripper close contact.
- Tune reachable red-pick parameters.
- Run post-close hold/retract/lift-like diagnostics.
- Replay the reachable red-pick diagnostic in the MuJoCo viewer.
- Open keyboard or Tkinter control panels for manual Feagine inspection.

Important current result:

- The reachable red-object diagnostic can produce red-grasper contact and measurable red-object planar motion.
- It does not yet justify a stable grasp claim.
- It does not yet justify an object lifting success claim.
- The reachable scene uses a gray `red_pedestal` to raise `red_object` into the currently reachable grasper workspace; this pedestal affects contact interpretation and should be handled explicitly in later geometry attribution diagnostics.

## Setup

Use the project conda environment:

```powershell
conda activate feagine_vlm
```

Install the package in editable mode:

```powershell
pip install -e .
```

Verify the external Feagine runtime:

```powershell
python scripts/verify_feagine_install.py
```

Feagine must stay outside this repository. Use the installed package, a relative external path, or `FEAGINE_SIM_ROOT`; do not copy `feagine-simulation/` or `feagine_simulation/` into this repo.

## Development Baseline

Default development environment:

```powershell
conda activate feagine_vlm
```

Baseline verification commands:

```powershell
pip install -e .
pytest
python scripts/verify_feagine_install.py
```

After any MuJoCo or Feagine scene change, run:

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/feagine_scene.json
```

Future top-level action interface baseline:

```text
Top-level action:
[dx, dy, dz, gripper_control]

Low-level Feagine command:
section_angles + grasper_rotation + gripper_open_close
```

The old top-level `section_angles` action interface is deprecated for future development.
Current runtime wrappers and diagnostics may still use low-level Feagine control fields such as
`section_angles`, `grasper_rotation`, and `grip_command` until `FeagineActionAdapter` lands.

## Real Feagine Tabletop Commands

Generate the reachable tabletop scene:

```powershell
python scripts/generate_feagine_tabletop_scene.py --variant reachable --output outputs/scenes/feagine_tabletop_scene_reachable.xml
```

View a generated tabletop scene:

```powershell
python scripts/view_feagine_tabletop_scene.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml
```

Validate passive red-object settling in the reachable scene:

```powershell
python scripts/validate_reachable_scene.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_scene_validation.json
```

Inspect tabletop scene state:

```powershell
python scripts/inspect_tabletop_scene_state.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/tabletop_scene_state.json
```

Diagnose passive tabletop contacts:

```powershell
python scripts/diagnose_tabletop_contacts.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --steps 200 --output outputs/diagnostics/tabletop_contacts.json
```

Scan reachable Feagine tip poses:

```powershell
python scripts/scan_feagine_reachability.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/feagine_reachability.json
```

Generate the deterministic approximate-PCC workspace model and recommended left/right task regions:

```powershell
python scripts/sweep_feagine_workspace.py --samples 5000 --seed 0 --output-dir outputs/workspace
```

This offline sweep writes `feagine_workspace_points.npy`, `feagine_workspace.json`, and
`feagine_workspace.png`. Its approximate-PCC backend does not perform MuJoCo collision
checks, so the proposed shelf regions must be confirmed with the real Feagine scene before
the layout is frozen.

Scan grasper-red contact locally:

```powershell
python scripts/scan_grasper_red_contact_local.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/grasper_red_contact_local.json
```

Run reachable red-pick attempt diagnostic:

```powershell
python scripts/run_reachable_red_pick_attempt.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_attempt.json
```

Run reachable red-pick hold/retract diagnostic:

```powershell
python scripts/run_reachable_red_pick_hold_retract.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_hold_retract.json
```

Tune reachable red-pick parameters:

```powershell
python scripts/tune_reachable_red_pick_parameters.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_parameter_tuning.json
```

Run the current post-close retract/lift-like diagnostic:

```powershell
python scripts/run_reachable_red_pick_retract_lift.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_retract_lift.json
```

Replay the current diagnostic in MuJoCo viewer:

```powershell
python scripts/view_reachable_red_pick_demo.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --trial gentle_retract_scale_down --realtime-factor 0.5
```

Open the tabletop Tkinter control panel:

```powershell
python scripts/control_feagine_tabletop_panel_ui.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml
```

Open the keyboard control script:

```powershell
python scripts/control_feagine_ui.py
```

The viewer/control tools are for diagnosis and manual observation only. They do not claim stable grasp or object lifting.

## Mock and Research Pipeline Commands

Run the minimal mock demo headlessly:

```powershell
python scripts/run_demo_env.py --headless
```

Manually test the current 4D MetaWorld-style task contracts without launching
MuJoCo or training:

```powershell
pytest tests/test_action_space.py tests/test_feagine_action_adapter.py tests/test_feagine_metaworld_reach.py tests/test_feagine_push_tasks.py tests/test_feagine_pick_place_tasks.py tests/test_feagine_gym_state_env.py
```

The current state-only Gymnasium-style wrapper is `FeagineGymStateEnv`. It
returns `reset() -> (observation, info)` and
`step(action_4d) -> (observation, reward, terminated, truncated, info)` while
reusing the existing 4D action adapter. It intentionally does not import
Gymnasium or run PPO/SAC/TD3 yet.

Run the current real Feagine/MuJoCo 4D Reach smoke rollout headlessly:

```powershell
python scripts/run_feagine_metaworld_reach.py --task feagine_reach_right --steps 50 --seed 0 --headless --output outputs/diagnostics/feagine_reach_right_smoke.json
```

This command uses `FeagineReachExpert` to emit only normalized 4D actions, then
logs the converted low-level Feagine command and IK/task metrics. It is a Reach
smoke validation only; it does not validate Push, Pick-Place, or RL training.
Omit `--headless` or pass `--render-mode human` only when you explicitly want a
MuJoCo viewer window.

Inspect real Feagine/MuJoCo model names from config:

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/a03_type_2_scene.json
```

Run the deterministic task-phase expert in mock mode:

```powershell
python scripts/run_task_phase_expert.py --task obstacle_avoid_pick --mock-env --max-steps 80 --output outputs/rollouts/mock_obstacle_avoid_pick_expert.json
```

Collect a small mock demonstration dataset:

```powershell
python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 3 --max-steps 80 --output data/demos/obstacle_avoid_pick_task_phase_debug.npz --mock-env --seed 0
```

Train the adapter on an existing demonstration file:

```powershell
python scripts/train_adapter.py --demo data/demos/obstacle_avoid_pick_task_phase_debug.npz --epochs 3 --batch-size 8 --output outputs/checkpoints/adapter_debug.pt --metrics-output outputs/metrics/adapter_training_debug.json
```

Evaluate mock policies:

```powershell
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert vlm_planner_ik --mock-env --num-episodes 2 --max-steps 80 --seed 0 --output outputs/metrics/mock_policy_eval.json --csv-output outputs/metrics/mock_policy_eval.csv
```

Export figures from saved metrics:

```powershell
python scripts/export_paper_figures.py --metrics outputs/metrics/mock_policy_eval.json --output-dir outputs/figures
```

## Tests

Run all tests only when explicitly needed:

```powershell
pytest
```

Prefer targeted tests during narrow development. Examples:

```powershell
pytest tests/test_generate_feagine_tabletop_scene.py
```

```powershell
pytest tests/test_tune_reachable_red_pick_parameters.py
```

```powershell
pytest tests/test_reachable_red_pick_retract_lift.py
```

Tests that require MuJoCo or Feagine must skip or fail clearly when the runtime is unavailable; they must not download large model weights.

## Repository Layout

```text
configs/      Environment, task, method, and evaluation configs.
data/         Local generated data roots; demo binaries are ignored by Git.
docs/         Design notes and planning specs.
experiments/  Reproduction notes for major experiment stages.
outputs/      Local generated scenes, diagnostics, metrics, figures, and checkpoints; most outputs are ignored by Git.
scripts/      CLI entry points for setup, inspection, diagnostics, control, data, training, evaluation, and visualization.
src/          Importable Python package.
tests/        Unit and targeted integration tests.
```

## Development Constraints

- Do not copy `feagine-simulation/` or `feagine_simulation/` into this repository.
- Do not modify `../feagine-simulation` or `../feagine_simulation`.
- Use `grasper_rotation`, not `gripper_rotation`.
- Keep VLM/VLA behavior deterministic until an explicit integration milestone exists.
- Do not download OpenVLA, Octo, or other large model weights during setup, tests, or demos.
- Do not treat mock-env results as paper conclusions.
- Do not claim stable grasp or object lifting from the current reachable red-pick diagnostics.
- Windows documentation commands should be single-line commands without continuation characters.
