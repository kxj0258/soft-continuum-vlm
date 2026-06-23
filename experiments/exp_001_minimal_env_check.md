# Experiment 001: Minimal Environment Check

Command sequence:

1. `conda create -n feagine_vlm python=3.10`
2. `conda activate feagine_vlm`
3. `pip install -e .`
4. `bash scripts/setup_feagine_env.sh`
5. `python scripts/verify_feagine_install.py`
6. `pytest`
7. `python scripts/run_demo_env.py`

Expected observations:

- Config loading succeeds for the YAML scaffold in `configs/`.
- Feagine verification either passes or clearly reports that Feagine is not installed.
- `pytest` passes the non-Feagine tests without requiring a MuJoCo graphics window.
