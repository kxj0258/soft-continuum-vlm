# PLANS.md

This roadmap keeps the repository focused on the paper project: embodiment-aware manipulation for contact-safe soft continuum arms.

## Global Constraints

- Do not copy `feagine-simulation/` or `feagine_simulation/` into this repository.
- Do not modify `../feagine-simulation` or `../feagine_simulation`.
- Reference Feagine only through installation, relative external paths, or `FEAGINE_SIM_ROOT`.
- Keep action schema fixed to `section_angles`, `grip_command`, and `grasper_rotation`.
- Do not introduce `gripper_rotation`.
- Do not download OpenVLA, Octo, VLM/VLA, or other large model weights.
- If MuJoCo or Feagine is unavailable, tests must graceful skip or fail clearly.
- Mock-env outputs are for CI/debug only and cannot be used as final paper conclusions.
- Windows command examples must be one-line commands without continuation characters.

## Completed Baseline Milestones

### Milestone 0: Repository and Feagine Verification

Output:

- Python package scaffold.
- Config files.
- Feagine install verification script.
- Unit tests that do not require a graphics window.

Verification command:

```powershell
python scripts/verify_feagine_install.py
```

### Milestone 1: Feagine MuJoCo Environment Wrapper

Output:

- `FeagineMujocoEnv` reset/step/render/close wrapper.
- Structured observations with robot state, objects, and contact.
- Human viewer and headless behavior.

Representative commands:

```powershell
python scripts/run_demo_env.py --headless
```

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/a03_type_2_scene.json
```

### Milestone 2: Tasks

Implemented task definitions:

- `pick_red_object`
- `obstacle_avoid_pick`
- `contact_push`
- `rotate_and_place`

Representative command:

```powershell
pytest tests/test_tasks.py tests/test_configs.py
```

### Milestone 3: Controllers and Safety

Output:

- Approximate continuum kinematics.
- `PccIkController`.
- `TaskPhaseExpert`.
- `SafetyProjector`.

Representative command:

```powershell
pytest tests/test_continuum_kinematics.py tests/test_pcc_ik_controller.py tests/test_task_phase_expert.py tests/test_safety_projector.py tests/test_safety_projector_modes.py
```

### Milestone 4: Mock Demonstration Data Path

Output:

- Metadata-rich `.npz` demo collection path.
- Mock-mode deterministic demonstration generation.

Representative command:

```powershell
python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 3 --max-steps 80 --output data/demos/obstacle_avoid_pick_task_phase_debug.npz --mock-env --seed 0
```

### Milestone 5: Soft Embodiment Adapter

Output:

- Adapter model.
- Adapter training script.
- Adapter policy wrapper.
- Train/validation metrics output.

Representative command:

```powershell
python scripts/train_adapter.py --demo data/demos/obstacle_avoid_pick_task_phase_debug.npz --epochs 3 --batch-size 8 --output outputs/checkpoints/adapter_debug.pt --metrics-output outputs/metrics/adapter_training_debug.json
```

### Milestone 6: Deterministic VLM Planner Stub

Output:

- Deterministic language-to-subgoal planner.
- Chinese/English keyword parsing for color, action, and safety constraints.
- No network calls and no model downloads.

Representative command:

```powershell
python scripts/run_vlm_planner_demo.py --task pick_red_object --language "pick the red object gently" --mock-env --output outputs/diagnostics/vlm_planner_demo.json
```

### Milestone 7: Evaluation and Figure Export

Output:

- Policy evaluation pipeline.
- CSV/JSON metrics.
- Figure export from saved metrics.

Representative commands:

```powershell
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert vlm_planner_ik --mock-env --num-episodes 2 --max-steps 80 --seed 0 --output outputs/metrics/mock_policy_eval.json --csv-output outputs/metrics/mock_policy_eval.csv
```

```powershell
python scripts/export_paper_figures.py --metrics outputs/metrics/mock_policy_eval.json --output-dir outputs/figures
```

## Current Real Feagine Tabletop Milestones

### SCENE-1: Tabletop Scene Generation

Output:

- `scripts/generate_feagine_tabletop_scene.py`
- Default and reachable tabletop scene variants.
- Local output scene XML under `outputs/scenes/`.

Representative command:

```powershell
python scripts/generate_feagine_tabletop_scene.py --variant reachable --output outputs/scenes/feagine_tabletop_scene_reachable.xml
```

### CONTACT-1: Scene and Contact Diagnostics

Output:

- Scene validation.
- Passive contact reports.
- Tabletop entity state inspection.

Representative commands:

```powershell
python scripts/validate_reachable_scene.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_scene_validation.json
```

```powershell
python scripts/diagnose_tabletop_contacts.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --steps 200 --output outputs/diagnostics/tabletop_contacts.json
```

### REAL-PICK-1 to REAL-PICK-4: Reachable Red Pick Diagnostics

Output:

- Reachability scan.
- Grasper-red contact scan.
- Preclose/close parameter sweep.
- Tuned reachable persistent-contact parameters.
- Post-close retract/lift-like following diagnostic.

Representative commands:

```powershell
python scripts/tune_reachable_red_pick_parameters.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_parameter_tuning.json
```

```powershell
python scripts/run_reachable_red_pick_retract_lift.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_retract_lift.json
```

Current conclusion:

- Contact and planar red-object motion are observed.
- Contact persistence is insufficient.
- Red-object z motion is weak.
- The red object remains on the pedestal.
- No stable grasp claim.
- No object lifting success claim.

### DEMO-1: Viewer Replay

Output:

- `scripts/view_reachable_red_pick_demo.py`
- MuJoCo viewer playback of selected REAL-PICK-4 trial schedules.

Representative command:

```powershell
python scripts/view_reachable_red_pick_demo.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --trial gentle_retract_scale_down --realtime-factor 0.5
```

## Next Milestone: SCENE-GRIPPER-1

Goal:

- Attribute red-object contacts to concrete Feagine grasper finger geoms, arm/body geoms, pedestal geom, and tabletop geom.
- Determine whether observed red-object motion is finger enclosure, arm/body pushing, pedestal blocking, or sweeping contact.

Expected output:

- A geometry attribution diagnostic report.
- No controller, policy, training, dataset collection, or UI changes unless explicitly requested.
