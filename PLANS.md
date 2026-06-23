# PLANS.md

This roadmap keeps the repository focused on reproducible research code for
"Embodiment-Aware Vision-Language Manipulation for Contact-Safe Soft Continuum
Arms". VLM and VLA integrations stay deterministic stubs until their explicit
milestones.

## Milestone 0: Project Skeleton and Feagine Install Verification

### Goal
Create the Python research repository, documentation, configs, unit tests, and
Feagine verification scripts without copying or modifying the sibling Feagine
distribution.

### Inputs and outputs
Input: project brief, sibling `../feagine_simulation` or
`../feagine-simulation`, optional
`FEAGINE_SIM_ROOT`.
Output: importable package, YAML configs, install script, verification script,
and non-Feagine unit tests.

### Files to change
`AGENTS.md`, `PLANS.md`, `README.md`, `pyproject.toml`, `.gitignore`,
`configs/`, `scripts/setup_feagine_env.sh`,
`scripts/verify_feagine_install.py`, `src/soft_continuum_vlm/utils/`,
`tests/`, and `experiments/`.

### Verification
Run `python scripts/verify_feagine_install.py` and `pytest`. If Feagine is not
installed, the verification script must explain the missing import or path.

### Risks and fallback
Risk: local Feagine folder names differ. Fallback: set `FEAGINE_SIM_ROOT` and
keep all project code path-relative.

## Milestone 1: Feagine MuJoCo Environment Wrapper

### Goal
Wrap the Feagine MuJoCo runtime for `a03_type_2` with a stable environment
interface that exposes observations, contact info, robot state, and close/reset
semantics.

### Inputs and outputs
Input: `configs/env/feagine_mujoco_a03_type_2.yaml`, Feagine MuJoCo runtime,
and robot asset path from `feagine_mujoco.robot_asset_path(model_type="mjcf")`.
Output: `FeagineMujocoEnv` with deterministic fallback behavior and graceful
errors when runtime imports are missing.

Milestone 1 implementation note: `FeagineMujocoEnv` loads the installed MuJoCo
runtime on `reset()`, constructs `mujoco.MjData` and `FeagineMjcfRobot`, and
advances physics with `mujoco.mj_step` inside `step()`. The environment action
schema directly mirrors Feagine controls: `section_angles` calls
`robot.drive_section_angles(...)`, `grip_command` calls
`robot.set_grip_command(...)`, and `grasper_rotation` calls
`robot.drive_grasper_rotation(...)`. No project-level delta-action mapping is
performed in the environment wrapper.

### Files to change
`src/soft_continuum_vlm/envs/base_env.py`,
`src/soft_continuum_vlm/envs/feagine_mujoco_env.py`,
`scripts/run_demo_env.py`, and `tests/test_env_imports.py`.

### Verification
Run runtime wrapper tests with fake MuJoCo modules, then run
`python scripts/verify_feagine_install.py`, `python scripts/run_demo_env.py`,
and `pytest` in the `feagine_vlm` environment. `run_demo_env.py` must be
headless and must not open a MuJoCo graphics window.

### Risks and fallback
Risk: the installed wheel stores the `a03_type_2` preset metadata inside the
legacy bundle directory name `a03`. Fallback: resolve `a03_type_2` to `a03`
only when `a03/preset.yaml` declares `preset_id: a03_type_2`.

## Milestone 2: Four Task Definitions

### Goal
Define task specs for pick, obstacle-avoid pick, contact push, and rotate/place
around the soft continuum arm and rotating two-finger gripper.

### Inputs and outputs
Input: task YAML files and Feagine scene metadata.
Output: deterministic task classes with language, target object, success
criteria, and evaluation stubs.

### Files to change
`configs/task/*.yaml`, `src/soft_continuum_vlm/tasks/`, and task tests.

### Verification
Run config parsing tests and task constructor/evaluation smoke tests.

### Risks and fallback
Risk: full object state is not available yet. Fallback: keep task evaluation as
explicit stubs with documented input, output, and integration path.

## Milestone 3: PCC IK, Scripted Expert, and Safety Projector

### Goal
Implement a safe scripted control baseline using PCC IK, scripted waypoints, and
contact-aware action projection.

### Inputs and outputs
Input: robot state, scene state, task spec, and safety limits.
Output: clipped Feagine action dict with `section_angles`, `grip_command`, and
`grasper_rotation`.

### Files to change
`configs/method/scripted_pcc_ik.yaml`,
`src/soft_continuum_vlm/controllers/pcc_ik_controller.py`,
`src/soft_continuum_vlm/controllers/scripted_expert.py`,
`src/soft_continuum_vlm/controllers/safety_projector.py`, and controller tests.

### Verification
Run safety projector unit tests and scripted expert action-shape tests.

### Risks and fallback
Risk: soft-arm kinematics are under-specified. Fallback: start with bounded
delta actions and add PCC equations behind the same interface.

## Milestone 4: Expert Data Collection

### Goal
Collect deterministic scripted demonstrations for all four tasks.

### Inputs and outputs
Input: environment config, task configs, scripted expert config, and random
seed.
Output: replayable trajectory files with observations, actions, contacts, task
metadata, and versioned config snapshots.

### Files to change
`scripts/collect_scripted_demos.py`, `src/soft_continuum_vlm/data/dataset.py`,
`src/soft_continuum_vlm/data/replay_buffer.py`, and `experiments/`.

### Verification
Run a small headless collection job and load the saved dataset in unit tests.

Milestone 4 implementation note: `MockContinuumEnv` provides deterministic
headless rollouts for CI and debug data generation. `collect_scripted_demos.py`
can write `.npz` arrays plus sidecar metadata JSON with action schema,
observation schema version, seed, command line, and environment type. The mock
path does not require MuJoCo, Feagine, graphics, network access, or model
weights.

### Risks and fallback
Risk: MuJoCo is unavailable on a machine. Fallback: skip simulator-dependent
tests and keep dataset schema tests independent of Feagine.

## Milestone 5: Soft Embodiment Adapter Training

### Goal
Train the small adapter that maps deterministic vision-language features,
proprioception, contact state, and morphology into continuum actions.

### Inputs and outputs
Input: scripted demonstration dataset and fixed synthetic or cached feature
tensors.
Output: adapter checkpoint, metrics, and action decoder compatibility tests.

### Files to change
`configs/method/soft_embodiment_adapter.yaml`,
`src/soft_continuum_vlm/models/soft_embodiment_adapter.py`,
`src/soft_continuum_vlm/models/action_decoder.py`,
`scripts/train_adapter.py`, and model tests.

### Verification
Run shape tests, one tiny CPU training step, and checkpoint load/save tests.

Milestone 5 implementation note: `DemoDataset` loads scripted `.npz` files for
plain Python and PyTorch DataLoader use. `train_adapter.py` trains the small MLP
on deterministic language features, proprioception, contact state, morphology,
and flattened Feagine action vectors. `evaluate_adapter.py` runs a first mock
evaluation pass and writes JSON metrics.

### Risks and fallback
Risk: torch is unavailable in lightweight environments. Fallback: mark model
tests with `pytest.importorskip("torch")` while keeping packaging explicit.

## Milestone 6: VLM Planner Integration

### Goal
Connect a deterministic VLM planner interface that converts language and scene
state into symbolic subgoals for the IK controller.

### Inputs and outputs
Input: language command, object detector output, scene state, and task spec.
Output: subgoal sequence and controller targets.

### Files to change
`configs/method/vlm_planner_ik.yaml`, `src/soft_continuum_vlm/perception/`,
planner modules, and planner tests.

### Verification
Run deterministic planner tests with no network access and no model downloads.

### Risks and fallback
Risk: real VLM behavior is non-deterministic. Fallback: keep a deterministic
stub and add real model adapters only behind explicit fixtures.

## Milestone 7: OpenVLA/OCTO-Style Baseline Interface Reservation

### Goal
Reserve interfaces for VLA baselines without downloading OpenVLA, Octo, or
large weights during setup, tests, or demos.

### Inputs and outputs
Input: common observation/action schema and optional external baseline config.
Output: disabled-by-default baseline adapters with clear installation notes.

### Files to change
`configs/method/`, `src/soft_continuum_vlm/models/`, optional baseline adapter
modules, and skip-safe tests.

### Verification
Run tests in an offline environment and confirm no large model downloads occur.

### Risks and fallback
Risk: baseline packages are heavy or unstable. Fallback: keep adapters as
protocol shells until an integration milestone explicitly enables them.

## Milestone 8: Experimental Evaluation and Paper Figure Export

### Goal
Evaluate scripted, adapter, VLM-planner, and reserved VLA baselines across the
four contact-safe manipulation tasks and export reproducible paper assets.

### Inputs and outputs
Input: trained checkpoints, trajectory logs, task configs, and evaluation seeds.
Output: metrics tables, plots, qualitative rollout frames, and experiment notes.

### Files to change
`experiments/`, future evaluation scripts, dataset loaders, and figure export
utilities.

### Verification
Run a small deterministic evaluation suite and regenerate at least one table or
plot from logged data.

### Risks and fallback
Risk: full evaluation is slow or hardware-dependent. Fallback: keep smoke tests
small and make simulator-heavy runs opt-in.
