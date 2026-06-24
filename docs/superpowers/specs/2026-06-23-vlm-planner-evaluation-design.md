# Milestone 6/8 设计：确定性 Planner 与评估

## 目标

实现 Milestone 6 和 Milestone 8 的第一个离线确定性版本。系统必须能在 `--mock-env` 下运行，不下载模型权重，保持 Feagine action schema，并导出 JSON、CSV、PNG artifact，用于 pipeline 调试和论文草稿表格。

## 架构

- `planners/`：把语言、observation 和 task name 转换成结构化 subgoal 与安全约束。
- `controllers/vlm_planner_controller.py`：组合确定性 planner、现有 scripted/PCC action path 和 `SafetyProjector`。
- `evaluation/`：运行 mock baseline rollout，计算 metrics，并导出论文草稿图表和表格。
- `scripts/`：提供 planner demo、baseline evaluation 和 paper figure export 的命令入口。

## 数据流

```text
language + observation
  -> DeterministicVLMPlanner.plan()
  -> VlmPlannerController.act()
  -> MockContinuumEnv.step()
  -> metrics / rollout logs / plots
```

## 约束

- 不修改 `../feagine_simulation` 或 `../feagine-simulation`。
- 不下载 OpenVLA、Octo、VLM 或 VLA 权重。
- action 字段固定为 `section_angles`、`grip_command` 和 `grasper_rotation`。
- 所有 mock-env 输出都必须标注为 debug artifact，不能作为最终论文结论。

## 测试

添加 planner 解析、planner-controller info logging、metric aggregation、baseline evaluation JSON/CSV 输出和 figure export 的单元测试与 smoke test。
