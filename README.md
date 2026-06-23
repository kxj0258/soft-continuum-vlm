# soft-continuum-vlm

Repository scaffold for an embodiment-aware vision-language manipulation research stack.

## Environment

Use the `feagine_vlm` conda environment:

```bash
conda create -n feagine_vlm python=3.10
conda activate feagine_vlm
pip install -e .
bash scripts/setup_feagine_env.sh
python scripts/verify_feagine_install.py
pytest
python scripts/run_demo_env.py
```

Feagine discovery should prefer `FEAGINE_SIM_ROOT`, then the default sibling `feagine-simulation` path, and finally the compatibility sibling `feagine_simulation` path when the default is absent.
