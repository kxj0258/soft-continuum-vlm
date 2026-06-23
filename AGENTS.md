# AGENTS.md

This repository is research code for a robotics paper, not a generic software demo.

- Do not copy `feagine-simulation/` into this repository.
- Do not copy `feagine_simulation/` into this repository.
- Do not modify files under `../feagine-simulation` or `../feagine_simulation`.
- Reference Feagine only through install scripts, relative paths, or `FEAGINE_SIM_ROOT`.
- All paths in source code, configs, and tests must work from relative project locations.
- Prefer testable, reproducible, extensible code over quick visual demos.
- After each code change, run the relevant unit tests.
- Update `PLANS.md` before adding complex behavior.
- Keep VLM and VLA modules deterministic until an explicit integration milestone.
- Do not download OpenVLA, Octo, or large model weights during setup, tests, or demos.
- If MuJoCo or Feagine is unavailable, tests must skip gracefully.
- All TODO comments and deferred engineering notes must state the expected input, output, and integration path.
