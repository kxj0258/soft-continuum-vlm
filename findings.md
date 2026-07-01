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

## M2 工作空间建模发现

- 已有 `continuum_kinematics.py` 提供可替换的近似 PCC FK：`section_angles_to_tip_delta()`。
- 已有 `scan_feagine_reachability.py` 强依赖 MuJoCo/Feagine，并针对少量手工命令和红色物体诊断；它不适合作为完整、可离线复现的工作空间模型。
- M2 应新增纯数值核心，使用确定性随机采样和近似 PCC FK；仿真后端可在后续以同一输出 schema 接入。
- 椭球拟合、任务区域划分和输出写入应与 MuJoCo 分离，确保 runtime 不可用时仍可单元测试。
- M2 输出 schema 已固定为点云 NPY、包含椭球与推荐布局的 JSON，以及三维 PNG。
- 推荐布局使用工作空间 5%-95% 分位范围和 `use_workspace_ratio` 安全收缩，避免直接把极值点作为货架中心。

## M3 IK 接口发现

- 现有 `PccIkController` 面向低层 action 字典，并混合夹爪字段，不适合作为新的纯 IK solver 接口。
- `continuum_kinematics.py` 已提供近似 FK 和数值 Jacobian，可作为 M3 两套 solver 的确定性后端。
- 新 solver 应只负责 `target_tip_position + current_section_angles + current_tip_position -> IkResult`，不处理夹爪；M4 adapter 再组合开合和姿态。
- 为减小近似 FK 与真实 observation 的偏差，solver 应以当前实测 tip 为锚点，只用 FK 预测角度变化带来的相对位移。
- 统一结果需要区分“产生可用安全步长”和“已经收敛到容差内”，否则微分 IK 的单步控制会被错误视为失败。

## M4 Action Adapter 发现

- 当前 observation 的机械臂状态位于 `observation["robot_state"]`，包含 `tip_pose.position`、`section_angles`、`grip_command` 和 `grasper_rotation`。
- 当前 runtime 只接受 `grip_command`，而新接口使用语义名 `gripper_open_close`；adapter 需要显式命令对象提供 runtime 别名转换，不能让两个名字在上层混用。
- 当前 Feagine 约定为 `grip_command=0` 打开、`grip_command=1` 闭合，因此顶层 `[-1, 1]` 应线性映射到 `[0, 1]`。
- 自主旋转应只读取 task context 和当前状态。默认保持当前角度；approach/grasp 可按目标方向或物体主轴对齐，transport/lift/retract 保持姿态。

## M5 任务链路发现

- 现有 `BaseTask`/`MockContinuumEnv` 仍围绕低层 action 字典和旧任务评估设计，直接改写会破坏既有 mock/数据管线。
- M5 应新增独立 `FeagineMetaWorldTask` 和包装环境，把旧低层 backend 视为执行器；包装层是唯一暴露 4D action 的入口。
- `FeagineMujocoEnv.reset()` 与 `MockContinuumEnv.reset()` 都接受额外关键字，因此包装层可统一传递 `seed`。
- 新包装层应忽略 backend 自身 reward，使用任务对象计算 reward/success，同时保留 backend done/info 作为截断和诊断信息。
- Reach expert 可以直接用 `(goal-tip)/delta_xyz_scale` 生成裁剪后的 4D action，并固定夹爪打开。
- `FeagineMetaWorldEnv` 不从 `envs/__init__.py` 急切导出，避免 `controllers -> action_space -> envs.__init__ -> wrapper -> controllers` 的循环导入；调用方使用其完整模块路径。

## M6a Push 任务发现

- 现有 contact observation 已提供 `max_force`、`max_penetration` 和逐接触 geom/body 名称，足以区分目标物体接触与无关碰撞。
- Push 任务需要轻量状态 `approach -> push -> complete`；任务基类应增加 `update_task_state()` hook，并由统一 `evaluate()` 在 reward/success/metrics 前调用。
- 包装环境必须在 task evaluation 后重新注入 task info，确保返回 observation 中的 phase 是本步更新后的状态。
- Push expert 应先移动到物体后方的预接触点，再沿 object→goal 方向推动，且始终输出打开夹爪的 4D action。

## M6b Pick-Place 任务发现

- 现有 object state 已约定 `pose.position`、`pose.orientation` 和可选 `grasped`，可作为新状态机的确定性输入。
- Pick-Place 成功不能由“夹爪闭合”推断；抓取阶段必须观察 `object.grasped`，抬升必须比较 reset 时物体高度，放置必须同时满足目标距离和夹爪打开。
- 八个操作阶段需要扩展为带终态的状态机：`approach -> align_grasper -> close_gripper -> lift -> transport -> align_place -> release -> retract -> complete`。
- 为避免“已放到目标但未收回”提前结束，最终 success 只在 retract 距离达到且 place 条件仍成立时报告。
