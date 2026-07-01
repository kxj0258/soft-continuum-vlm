# AGENTS.md

本仓库是机器人论文研究代码，不是通用软件 demo。

- 不要把 `feagine-simulation/` 复制到本仓库。
- 不要把 `feagine_simulation/` 复制到本仓库。
- 不要修改 `../feagine-simulation` 或 `../feagine_simulation` 下的文件。
- 只能通过安装脚本、相对路径或 `FEAGINE_SIM_ROOT` 引用 Feagine。
- 源代码、配置和测试中的路径必须能从项目相对位置运行。
- 优先选择可测试、可复现、可扩展的实现，不要只做快速视觉 demo。
- 每次代码改动后运行相关单元测试。
- 添加复杂行为前先更新 `PLANS.md`。
- 在明确集成里程碑前，VLM 和 VLA 模块必须保持 deterministic。
- setup、测试和 demo 过程中不要下载 OpenVLA、Octo 或大型模型权重。
- 如果 MuJoCo 或 Feagine 不可用，测试必须 graceful skip。
- 所有 TODO 注释和 deferred engineering notes 都必须说明预期输入、输出和集成路径。
- 当前 Windows 环境中的文档命令示例必须写成单行，不使用续行符。

## Codex Narrow-Step Development Rules

For this repository, every Codex task must be narrow and acceptance-test driven.

Default rules:

1. Do not scan or rewrite the whole repository unless explicitly requested.
2. Read only the files listed in the user prompt.
3. Modify only the files listed in the user prompt.
4. Do not run full `pytest` unless explicitly requested.
5. Prefer targeted tests such as `pytest tests/test_xxx.py`.
6. Do not implement VLM, VLA, OpenVLA, Octo, adapter training, or dataset collection unless the current step explicitly asks for it.
7. Do not touch `../feagine_simulation` or `../feagine-simulation`.
8. Do not download model weights.
9. Do not introduce new dependencies unless explicitly requested.
10. End every response with:
    - files changed
    - commands run
    - whether acceptance criteria passed
    - remaining issues

## Project Instructions for Codex

### Environment

This project uses the following conda environment:

```bash
conda activate feagine_vlm
```

### Default testing policy

Do not automatically run tests, validation commands, lint commands, format commands, build commands, installation commands, asset-sync commands, training commands, evaluation commands, MuJoCo rendering commands, or simulation launch commands after modifying code.

Only run such commands when the user explicitly asks for them in the current task.

Do not automatically run:

```bash
pip install -e .
pytest
python -m pytest
python scripts/verify_feagine_install.py
python scripts/inspect_feagine_scene.py
python -m feagine_mujoco_dev.sync_assets --preset a03
```

Do not automatically launch:

```text
MuJoCo viewer
headless simulation rollouts
demo collection
training scripts
evaluation scripts
rendering scripts
asset synchronization commands
```

After making changes, report:

1. Files changed
2. What changed
3. Whether tests were intentionally not run
4. Recommended manual verification commands
5. Risks or follow-up issues

Never claim tests passed unless they were explicitly requested and actually run.
