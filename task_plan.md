# Feagine MetaWorld/VLM 路线执行计划

## 目标

按 `todolist.md` 的依赖顺序，以可验收窄步骤实现统一 4D action、工作空间建模、动作适配、任务、RL、视觉、数据和 VLM/VLA 链路。

## 执行规则

- 当前开发分支：`codex/roadmap-implementation`。
- 每个窄步骤完成后只提交该步骤涉及的文件，使用说明性 Git 提交信息。
- 不自动 push、不改写历史、不提交生成的模型权重、数据集或 `outputs/` 产物。
- 未经用户明确要求，不运行测试、验证、lint、format、build、install 或仿真命令。
- 测试可以作为验收规范先写入，但必须明确标记为“未运行”。
- Feagine 只能通过安装包、相对外部路径或 `FEAGINE_SIM_ROOT` 使用。

## 阶段

- [x] M0：提交已有 4D action 开发基线，提交 `3247ce1`。
- [ ] M1：实现统一 4D action 数据契约、缩放映射和单元测试（代码已完成，等待用户手动验证）。
- [ ] M2：实现 PCC 工作空间采样、椭球拟合与任务区域输出（离线后端代码已完成，等待手动验证和仿真碰撞复核）。
- [ ] M3：实现统一 IK 接口、PCC IK、微分 IK 和安全回退（代码已完成，等待用户手动验证）。
- [ ] M4：实现 `FeagineActionAdapter` 与自动夹爪姿态控制（代码已完成，等待用户手动验证）。
- [ ] M5：实现 MetaWorld 风格任务基类与 Reach 任务纵向链路。
- [ ] M6：实现 Push 和 Pick-Place 任务及 deterministic experts。
- [ ] M7：实现 Gymnasium 环境与 RL baseline 接口。
- [ ] M8：实现双相机、多模态 observation、轨迹与回放工具。
- [ ] M9：实现 deterministic VLM planner、skills、数据导出和 VLA 接口。
- [ ] M10：实现复杂任务与统一论文评估协议。

## 当前窄步骤：M4

1. 定义语义低层命令及 `grip_command` runtime 别名转换。
2. 先增加动作缩放、观测解析、夹爪映射、自动旋转和 IK 失败保持的目标测试，但不执行。
3. 实现 `FeagineActionAdapter.convert()` 与结构化转换结果。
4. 默认使用微分 IK，并通过 M3 重试接口执行缩小增量回退。
5. 更新公开导出、配置、设计文档、TODO 与进度记录，并创建原子提交。

## 已知约束与风险

- 严格 TDD 要求运行红灯/绿灯测试，但仓库规则禁止自动运行；本路线采用“测试先写、用户手动运行”的降级方式。
- 当前低层 runtime 仍使用 `section_angles`、`grasper_rotation`、`grip_command`，M1 不删除低层命令。
- `gripper_control` 的正负方向必须在 M1 固定，后续不得由各任务自行解释。
- 近似 PCC 工作空间不包含 MuJoCo 碰撞、穿模和真实结构误差，不能单独作为最终货架坐标依据。
- M3 solver 仍使用近似 PCC 后端；真实 Feagine 标定应通过同一 `IkSolver` 接口注入，而不是改变上层 action 契约。
- M4 尚未直接改写 `FeagineMujocoEnv.step()`；M5 的统一任务/环境流程应调用 `command.to_runtime_action()` 进入现有低层 runtime。

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---|---|
| 读取不存在的 `src/soft_continuum_vlm/envs/base.py` | 按常见命名检查环境基类 | `rg` 已确认实际文件为 `base_env.py`，后续改读该文件 |

## 既有阶段记录

- 已实现 MuJoCo 状态工具、SceneRegistry、结构化 Feagine observation、inspection 脚本、近似 PCC kinematics、`PccIkController`、`TaskPhaseExpert`、`SafetyProjector`、policy/rollout/evaluation 和 adapter 训练管线。
- 既有完整验证记录为 `68 passed, 4 skipped`；该结果来自 2026-06-23，不代表当前 M1 改动已经通过验证。
- 真实 Feagine inspection 和 collection 在当时环境中因缺少 runtime 清楚失败，没有回退到 mock。
