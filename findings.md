# 发现记录

## 当前基线

- 顶层 action 已在配置和设计文档中定义为 `[dx, dy, dz, gripper_control]`。
- 归一化范围为 `[-1, 1]`，默认 `delta_xyz_scale` 为 `0.01` 米。
- 当前 Feagine runtime 仍接收低层 `section_angles`、`grasper_rotation` 和 `grip_command`。
- 顶层迁移必须保留低层 runtime 边界，不能把 `section_angles` 从底层命令中删除。
- 现有 `Action` 位于 `envs/base_env.py`，类型为低层 `Mapping[str, Any]`；M1 应新增独立顶层类型，避免直接破坏 runtime 环境。
- 当前代码搜索未发现已经接入的 Gymnasium `Box`，M1 不应为动作契约强行引入新依赖。

## Git 管理决策

- 开发分支：`codex/roadmap-implementation`。
- 一个窄步骤对应一个提交。
- 不自动 push，不 force push，不 amend 用户历史。
- 生成数据、模型权重和仿真输出不纳入自动提交。

## 验证政策

- 当前任务未授权执行测试或验证。
- 后续新增测试只作为可执行验收规范写入，结果统一记录为“未运行”。

## 既有仓库能力

- `FeagineMujocoEnv` 和 `MockContinuumEnv` 已提供结构化 observation。
- 已存在近似 PCC kinematics、`PccIkController`、`TaskPhaseExpert`、`SafetyProjector`、policy、rollout 和 evaluation 管线。
- 既有 mock 结果仅用于 CI/debug，不能作为论文结论。
- 真实 Feagine runtime 缺失时必须清楚失败或 graceful skip。
