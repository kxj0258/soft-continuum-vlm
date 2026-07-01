# 完整开发规划

## 里程碑 0：开发基线冻结

- 明确 4D 顶层动作与低层 Feagine 命令边界。
- 固定坐标系、单位、夹爪正负方向、命名规范。
- 保持 VLM/VLA deterministic。
- 建立 graceful-skip 和输出目录规范。

验收：接口设计、配置、开发约束不存在互相冲突。

## 里程碑 1：PCC 工作空间与固定场景

- 实现 `scripts/sweep_feagine_workspace.py`。
- 采样分段角度，执行 FK/仿真末端读取。
- 过滤越界、碰撞、穿模和数值异常点。
- 输出 NPY、JSON、PNG，并拟合椭球参数。
- 根据安全裕量划分左侧 pick、右侧 place 区域。
- 固化“竖直居中机械臂 + 左右货架”场景配置。

验收：输出可复现工作空间、任务区域和固定布局参数。

## 里程碑 2：统一 4D Action

- 建立 `action_space.py`。
- 定义 `[dx, dy, dz, gripper_control] ∈ [-1,1]^4`。
- 配置 `delta_xyz_scale`，默认 0.01 m。
- 明确世界坐标系增量语义。
- 明确夹爪控制正负约定。
- 从 task、expert、policy 公共接口移除顶层 `section_angles`。

验收：所有顶层环境 action shape 均为 `(4,)`。

## 里程碑 3：IK 与 FeagineActionAdapter

- 建立统一 IK result/interface。
- 实现 PCC 全局逆解。
- 实现 Jacobian 微分 IK。
- 实现角度限制、阻尼、连续性约束。
- 实现“缩小增量重试 → 保持当前姿态”的失败回退。
- 实现自动 `grasper_rotation` 控制。
- 输出低层 `section_angles + grasper_rotation + grip_command`。

验收：零增量保持、三轴方向正确、越界目标安全失败、轨迹连续。

## 里程碑 4：MetaWorld 风格任务核心

- 建立统一任务基类和注册机制。
- 统一 reset、reward、success、goal、task info。
- 打通 `4D action → adapter → simulation → observation → reward`。
- 先实现 Reach Left、Reach Right、Reach 3D。
- 为 Reach 编写 deterministic scripted expert。

验收：Reach 成为第一条完整纵向链路，并输出距离与 IK 指标。

## 里程碑 5：接触与 Push

- 实现左右互推和 contact push。
- 统一接触对象归因。
- 记录接触力、穿透、物体位移。
- 增加误碰撞与目标接触区分。
- 编写 Push scripted expert。

验收：推动成功可重复，接触力受控，指标能区分成功与误碰。

## 里程碑 6：Pick-Place

- 实现左右货架互取互放。
- 建立八阶段状态机。
- 自动计算各阶段夹爪朝向。
- 分别统计 grasp、lift、transport、place。
- 编写 deterministic expert。
- 先完成无遮挡简单版本，再加入障碍物。

验收：不得仅凭接触或平面位移宣称抓取成功；必须检测离开支撑面和最终放置。

## 里程碑 7：Gymnasium 与 RL

- 封装 state-only Gymnasium 环境。
- 支持 terminated/truncated、seed 和 goal-conditioned observation。
- 先建立 random 与 scripted 基线。
- 依次接入 PPO、SAC、TD3。
- Reach/Pick-Place 条件成熟后接入 HER。
- 统一 checkpoint、曲线和评估结果格式。

验收：Reach 明显优于随机策略；Push 和 Pick-Place 按阶段指标评估。

## 里程碑 8：双相机与多模态观测

- 加入 global RGB/depth 相机。
- 加入 wrist RGB/depth 相机。
- 定义统一 state/images/depth/language/task schema。
- RL 可只选择 state，VLM/VLA 可读取完整观测。
- 图像时间戳与控制步骤严格对齐。

验收：headless 可采集双相机数据，手眼视角同步且无遮挡严重问题。

## 里程碑 9：轨迹数据与回放

- 统一 episode/step 数据 schema。
- 保存高层 4D action 和低层命令。
- 保存成功、失败、阶段、接触和随机种子。
- 建立增量采集与中断恢复。
- 实现逐步回放、视频导出和控制信息叠加。
- 增加 LeRobot/RLDS 导出器。

验收：轨迹可复现，失败阶段可定位，导出格式可校验。

## 里程碑 10：VLM + Skills

- 冻结结构化 planner schema。
- 保留 deterministic planner fallback。
- 将专家拆为 reach、push、pregrasp、grasp、lift、place、retract、avoid skills。
- 所有 skill 只输出 4D action。
- 非法 VLM 输出必须回退，不直接进入控制层。

验收：VLM 只做高层规划，控制结果可由 deterministic baseline 对照。

## 里程碑 11：VLA

- 先训练轻量 behavior cloning baseline。
- 固定训练/验证/测试 episode 划分。
- 比较 state-only、vision-only、multimodal。
- 完成 LeRobot/RLDS 兼容后再接 SmolVLA、Octo。
- OpenVLA 最后接入，且单独设立权重与算力里程碑。

验收：VLA 必须与 scripted、RL、VLM+skills 使用同一评估协议。

## 里程碑 12：复杂任务与论文实验

- 按顺序加入绕障、按钮、货架插入、抽屉、铰链任务。
- 每项先有 scripted baseline，再开放 RL/VLM/VLA。
- 固定随机种子、场景版本、数据版本和指标定义。
- 生成汇总 JSON/CSV、曲线、视频和复现实验说明。

验收：每个论文结论都能追溯到配置、checkpoint、数据集和原始指标。

## 主要风险

- 工作空间可能不是单一椭球，需要保留原始点云和分区边界。
- PCC 模型与 MuJoCo 实际末端位置可能存在系统误差。
- Jacobian 在奇异区可能产生大幅动作，需要阻尼和限幅。
- `grip_command` 正负语义尚需明确冻结。
- 当前红色物体接触不等于稳定抓取或抬升。
- 多模态数据量和 I/O 成本可能迅速膨胀。
- VLA 接入过早会掩盖底层控制和数据质量问题。

## 文件变更

修改：

- `AGENTS.md`
- `README.md`
- `configs/method/scripted_pcc_ik.yaml`
- `src/soft_continuum_vlm/envs/feagine_mujoco_env.py`
- `src/soft_continuum_vlm/models/action_decoder.py`

新增：

- `configs/control/feagine_4d_action.yaml`
- `docs/design/feagine_4d_action_interface.md`
- `docs/todos/feagine_metaworld_vlm_todolist.md`
- `todolist.md`

具体内容：增加开发与测试约束、4D action 基线配置、接口设计、迁移注释和长期路线图。

## 命令运行

仅运行了 Git 状态/差异读取、文档读取、`git add --all` 和 `git commit`。未运行测试、验证、lint、format、build、install 或仿真命令。

## 建议手动验证

```powershell
git show --stat --oneline 3247ce1
```

```powershell
conda activate feagine_vlm
```

```powershell
pytest
```

```powershell
python scripts/verify_feagine_install.py
```

## 验收与剩余问题

- Git 提交：已完成。
- 完整规划：已给出。
- 技术验收标准：未执行，因本次明确禁止运行验证。
- 剩余问题：下一步应拆出单独窄任务，优先实现“工作空间采样”或“4D action 数据契约”，不建议一次实施整个路线图。