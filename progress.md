# 进度记录

## 2026-06-23

- 用显式 UTF-8 解码读取三个开发附件。
- 检查 git 状态、近期提交、仓库文件、`README.md`、`AGENTS.md`、`PLANS.md`、Feagine/mock env、schema、controllers、采集/评估脚本、任务和代表性测试。
- 创建本地计划文件。
- 先写 Stage 1 测试，确认缺少 `mujoco_state` 和 `scene_registry` 的红灯状态。
- 实现 `mujoco_state.py`、`scene_registry.py`、结构化 `FeagineMujocoEnv` observation、scene config、inspection 脚本、README 章节和实验 006 记录。
- Stage 1 相关测试通过：`14 passed`。
- 运行 inspection 脚本，当前 shell 缺少真实 Feagine/MuJoCo runtime，脚本清楚报告原因。
- 先写 Stage 2 测试，确认控制器、专家和 metrics 缺失。
- 实现近似 PCC kinematics、升级 `PccIkController`、添加 `TaskPhaseExpert`、扩展 `SafetyProjector` 模式、改造 demo collection、添加 rollout debug 脚本和实验记录。
- Stage 2 相关测试通过：`21 passed`。
- 成功运行 Stage 2 mock rollout 和 demo collection 命令。
- 先写 Stage 3 测试，确认 policy、rollout 和 evaluation 实现缺失。
- 实现 policy package、rollout dataclass/runner、policy evaluation 脚本、AdapterPolicy、VLM planner IK policy、plotting 输出、adapter train/val metrics、evaluation configs 和实验记录。
- 为根目录 `scripts.*` 测试添加 `scripts/__init__.py`，并把 `.` 加入 pytest `pythonpath`。
- 完整 `pytest` 通过：`68 passed, 4 skipped`。
- 成功重跑 mock rollout、demo collection、policy evaluation 和 figure export。
- 真实 Feagine inspection 和 collection 命令因 runtime 缺失清楚失败，并且没有使用 mock fallback。

## 2026-06-24

- 根据用户要求开始中文化 README 和其他 Markdown 文档。
- 将命令示例改为 Windows 单行命令，避免使用续行符。
- 已中文化 `README.md`、`PLANS.md`、`AGENTS.md`、`experiments/`、`data/`、`outputs/`、`docs/superpowers/specs/` 和本地计划记录。
- 已扫描 Markdown，确认没有反斜杠续行和 `bash` 命令块。
