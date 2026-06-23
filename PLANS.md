# PLANS.md

## Milestone 0

Current milestone.

### Goal
Repository skeleton, packaging, configs, docs, and initialization.

### Inputs and outputs
Inputs: task brief requirements.
Outputs: importable package root, starter config files, repo docs, and a clean git root commit.

### Files
- `AGENTS.md`
- `PLANS.md`
- `README.md`
- `pyproject.toml`
- `.gitignore`
- `configs/env/feagine_mujoco_a03_type_2.yaml`
- `configs/task/pick_red_object.yaml`
- `configs/task/obstacle_avoid_pick.yaml`
- `configs/task/contact_push.yaml`
- `configs/task/rotate_and_place.yaml`
- `configs/method/scripted_pcc_ik.yaml`
- `configs/method/vlm_planner_ik.yaml`
- `configs/method/soft_embodiment_adapter.yaml`
- `src/soft_continuum_vlm/__init__.py`
- `src/soft_continuum_vlm/envs/__init__.py`
- `src/soft_continuum_vlm/tasks/__init__.py`
- `src/soft_continuum_vlm/controllers/__init__.py`
- `src/soft_continuum_vlm/perception/__init__.py`
- `src/soft_continuum_vlm/models/__init__.py`
- `src/soft_continuum_vlm/data/__init__.py`
- `src/soft_continuum_vlm/utils/__init__.py`
- `experiments/README.md`
- `experiments/exp_001_minimal_env_check.md`

### Verification
Run the repository file listing command from the brief and confirm no path under `C:\\work_kxj\\feagine-simulation` appears.

### Risks and fallback
If a required directory is missing, create it without adding unrelated files. If git is not initialized yet, initialize it only inside `C:\\work_kxj\\soft-continuum-vlm`.

## Milestone 1

### Goal
Feagine-aware configuration loading and environment discovery.

### Inputs and outputs
Inputs: YAML config paths and `FEAGINE_SIM_ROOT`.
Outputs: resolved Feagine root, parsed config dictionaries, and skip-safe runtime checks.

### Files
- `src/soft_continuum_vlm/utils/`
- `src/soft_continuum_vlm/config/`
- `tests/`

### Verification
Unit tests for default path, compatibility path, and environment override behavior.

### Risks and fallback
If Feagine is absent, runtime checks must report a skip instead of failing hard.

## Milestone 2

### Goal
Task registry and initial task metadata loading.

### Inputs and outputs
Inputs: task YAML files.
Outputs: normalized task definitions for later execution code.

### Files
- `configs/task/`
- `src/soft_continuum_vlm/tasks/`
- `tests/`

### Verification
Task config parsing tests and schema smoke checks.

### Risks and fallback
Keep the registry static and deterministic until integration is available.

## Milestone 3

### Goal
Method configuration loading for scripted and VLM-guided modes.

### Inputs and outputs
Inputs: method YAML files.
Outputs: method descriptors and controller selection metadata.

### Files
- `configs/method/`
- `src/soft_continuum_vlm/controllers/`
- `tests/`

### Verification
Parsing tests for all method configs.

### Risks and fallback
Avoid downloading model weights or external assets.

## Milestone 4

### Goal
Minimal environment and adapter interfaces.

### Inputs and outputs
Inputs: resolved env config and task/method descriptors.
Outputs: lightweight interface classes for future runtime code.

### Files
- `src/soft_continuum_vlm/envs/`
- `src/soft_continuum_vlm/models/`
- `tests/`

### Verification
Import tests and constructor smoke tests.

### Risks and fallback
Keep the interfaces thin so later implementations can evolve without churn.

## Milestone 5

### Goal
Experiment scaffolding and reproducible notes.

### Inputs and outputs
Inputs: repository commands and verification outcomes.
Outputs: experiment notes that document the first end-to-end check.

### Files
- `experiments/`

### Verification
Manual command replay from the experiment note.

### Risks and fallback
Keep notes factual and repeatable.

## Milestone 6

### Goal
CLI entry points and local workflow scripts.

### Inputs and outputs
Inputs: configs and repository paths.
Outputs: small helper scripts for setup and verification.

### Files
- `scripts/`
- `src/soft_continuum_vlm/`

### Verification
Script invocation tests or command smoke checks.

### Risks and fallback
Scripts must avoid hard-coded machine-specific paths.

## Milestone 7

### Goal
Behavioral tests around safety and skip conditions.

### Inputs and outputs
Inputs: environment availability and config state.
Outputs: tests that prove graceful skips when dependencies are missing.

### Files
- `tests/`

### Verification
Pytest runs with and without Feagine present.

### Risks and fallback
Use skip markers instead of brittle environment assumptions.

## Milestone 8

### Goal
Research iteration hardening and documentation cleanup.

### Inputs and outputs
Inputs: accumulated implementation details and test results.
Outputs: stable docs, clearer notes, and a maintainable handoff surface.

### Files
- `README.md`
- `AGENTS.md`
- `PLANS.md`

### Verification
Final repo review against the task brief.

### Risks and fallback
Keep the repository focused on the research scaffold and avoid scope creep.
