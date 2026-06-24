# 实验 001：最小环境检查

## 命令序列

```powershell
conda create -n feagine_vlm python=3.10
conda activate feagine_vlm
pip install -e .
bash scripts/setup_feagine_env.sh
python scripts/verify_feagine_install.py
pytest
python scripts/run_demo_env.py
```

## 预期现象

- `configs/` 下的 YAML 配置可以正常加载。
- Feagine 验证脚本要么通过，要么清楚说明缺失的路径或 Python import。
- `pytest` 中非 Feagine 测试不依赖 MuJoCo 图形窗口。
- `run_demo_env.py` 可用于本地验证 runtime wrapper；无显示环境可改用 `python scripts/run_demo_env.py --headless`。
