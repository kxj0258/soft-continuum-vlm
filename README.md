# soft-continuum-vlm

Research code for **Embodiment-Aware Vision-Language Manipulation for
Contact-Safe Soft Continuum Arms**.

This repository targets a robotics paper codebase, not a generic software demo.
The first stage focuses on a reproducible MuJoCo simulation scaffold around the
Feagine `a03_type_2` soft continuum arm and rotating two-finger gripper. VLM,
OpenVLA, Octo, and other large-model components remain deterministic stubs until
their integration milestones.

## Directory Layout

```text
configs/      Environment, task, and method YAML files.
scripts/      Setup, verification, demo, data, and training entry points.
src/          Importable Python package.
tests/        Unit tests that do not require a MuJoCo graphics window.
experiments/  Reproducibility notes and future experiment records.
```

Feagine must stay outside this repository. By default the code looks for sibling
directories `../feagine_simulation` and `../feagine-simulation`. If your local
distribution uses a different path, set `FEAGINE_SIM_ROOT`.

## Installation

Use Python >= 3.9. Activate your intended environment first, for example:

```bash
conda activate feagine_vlm
pip install -e .
```

Then install or verify the Feagine runtime:

```bash
bash scripts/setup_feagine_env.sh
python scripts/verify_feagine_install.py
```

If your Feagine distribution is not at one of the default sibling paths:

```bash
export FEAGINE_SIM_ROOT=/path/to/feagine_simulation
bash scripts/setup_feagine_env.sh
python scripts/verify_feagine_install.py
```

The setup script uses the current Python environment, builds
`feagine-simulation-core`, installs the `feagine_mujoco-*.whl`, and optionally
installs SAPIEN when `INSTALL_FEAGINE_SAPIEN=1`.

## Run Checks

```bash
pytest
python scripts/run_demo_env.py
```

`run_demo_env.py` performs a headless Feagine MuJoCo reset and step using
Feagine control fields: `section_angles`, `grip_command`, and
`grasper_rotation`.

## Current Scope

Implemented now:

- Relative path utilities with `FEAGINE_SIM_ROOT` override.
- Feagine verification script with clear errors and warnings.
- Abstract environment interface and Feagine MuJoCo wrapper using Feagine
  control fields directly as actions.
- Task, perception, data, controller, safety projector, and adapter stubs.
- Unit tests for paths, configs, imports, safety projection, and adapter shape
  when torch is available.

Not implemented yet:

- Real MuJoCo physics stepping.
- Real object detection or VLM planning.
- Large VLA model downloads or training.
