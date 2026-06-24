# 任务计划

## 目标

按三个粘贴文本顺序完成开发，并把 README 与其他 Markdown 文档改为中文说明，同时让文档命令适配当前 Windows 环境。

## 已完成阶段

- [x] Stage 1：添加 MuJoCo 状态工具、SceneRegistry、结构化 Feagine observation、inspection 脚本、文档和测试。
- [x] Stage 2：添加连续体近似运动学、非零 PCC IK、任务阶段专家、安全投影模式、采集/调试脚本、metrics、文档和测试。
- [x] Stage 3：添加 policy 接口、AdapterPolicy、VLM planner IK policy、统一 rollout/evaluation、训练增强、图表导出、配置、文档和测试。
- [x] 验证：运行 `pytest`、mock rollout、mock demo collection、mock policy evaluation、figure export，并确认真实 Feagine 命令在 runtime 缺失时清楚失败。
- [x] 文档中文化：README、PLANS、experiments、data/outputs 说明和本地计划记录全部改为中文，命令示例改为 Windows 单行。

## 约束

- 不复制或修改 `../feagine-simulation` 或 `../feagine_simulation`。
- action schema 固定为 `section_angles`、`grip_command` 和 `grasper_rotation`。
- 不引入 `gripper_rotation`。
- 不下载 OpenVLA、Octo、VLM、VLA 或大型模型权重。
- Feagine/MuJoCo 不可用时，测试和脚本必须清楚 skip 或失败。
- mock-env 输出只能用于 CI/debug，不能作为论文结果。
- 文档命令必须使用单行形式，不使用续行符。

## 遇到的问题

| 问题 | 尝试 | 处理 |
|---|---|---|
| PowerShell 默认输出把 UTF-8 中文附件显示成乱码。 | 直接 `Get-Content`。 | 使用显式 UTF-8 输出编码重新读取。 |
| 当前 shell 缺少真实 Feagine/MuJoCo runtime。 | 运行 scene inspection 和真实 demo collection。 | 脚本清楚报告缺少 `pyfeagine_sim_core`、`feagine_mujoco` 或 `mujoco`，并且没有 fallback 到 mock。 |
