# `soft-continuum-vlm` 后续开发 TODO List

## 0. 项目开发基线与总体原则

### 0.1 环境与运行基线

* [ ] 固定使用项目依赖环境：

```bash
conda activate feagine_vlm
```

* [ ] 每次大改前运行基础检查：

```bash
pip install -e .
pytest
python scripts/verify_feagine_install.py
```

* [ ] 每次涉及 MuJoCo/Feagine 场景修改后运行：

```bash
python scripts/inspect_feagine_scene.py \
  --config configs/env/feagine_mujoco_a03_type_2.yaml \
  --steps 5 \
  --print-bodies \
  --print-geoms \
  --print-sites \
  --output outputs/scene_inspection/feagine_scene.json
```

### 0.2 新接口原则

* [ ] 废弃旧的 action 接口，不再维护旧的 `section_angles` 顶层控制入口。
* [ ] 新的顶层 action 统一采用 MetaWorld 风格：

```text
action = [dx, dy, dz, gripper_control]
```

* [ ] `dx, dy, dz` 表示末端在任务空间中的增量位移。
* [ ] `gripper_control` 表示高层夹爪开合控制。
* [ ] Feagine 夹爪的旋转自由度不直接暴露为 RL 顶层 action，而由底层控制器、任务阶段控制器或姿态对齐模块自动生成。
* [ ] 所有 RL、VLM、VLA、scripted expert 都统一使用这个 4 维 action 接口。
* [ ] 底层再将 4 维 action 转换为：

```text
section_angles
grasper_rotation
gripper_open_close
```

---

## 1. Feagine 工作空间测试与机械臂放置规划

## 1.1 PCC 工作空间建模

### 目标

基于 PCC 模型测试 Feagine 连续体机械臂的实际可达工作空间。由于连续体臂弯曲特性，工作空间预期不是传统工业机械臂的规则盒状空间，而更接近椭球形、蘑菇形或偏置椭球形区域。

### 任务

* [x] 编写 Feagine 工作空间采样脚本：

```text
scripts/sweep_feagine_workspace.py
```

* [x] 随机或网格采样 `section_angles`。
* [x] 通过正运动学或仿真读取末端 `tip_pose`。
* [x] 记录所有可达末端位置。
* [ ] 过滤掉异常点、碰撞点、穿模点。
* [x] 输出三维点云：

```text
outputs/workspace/feagine_workspace_points.npy
outputs/workspace/feagine_workspace.json
outputs/workspace/feagine_workspace.png
```

* [x] 拟合工作空间椭球：

```text
center
major_axis
minor_axis
height_range
left_right_reachable_range
front_back_reachable_range
```

### 验收标准

* [ ] 能可视化 Feagine 末端工作空间点云。
* [ ] 能得到机械臂竖直放置时的左右侧可达范围。
* [ ] 能判断哪些区域适合放置 pick 物体、place 目标和障碍物。
* [ ] 能输出推荐任务区域配置。

---

## 1.2 机械臂放置方式规划

### 当前推荐方案

机械臂竖直向上放置在桌面中央，使其自然工作空间围绕底座形成上方椭球形区域。

```text
桌面布局俯视：

      place shelf
          |
left side | right side
          |
    [ Feagine base ]
          |
      pick shelf
```

或者：

```text
左侧架子：pick object
中间区域：Feagine base
右侧架子：place target
```

### 任务

* [ ] 将 Feagine base 固定在桌面中心或略靠后位置。
* [ ] 将 pick 区域放在机械臂左侧可达区域。
* [ ] 将 place 区域放在机械臂右侧可达区域。
* [ ] 保证 pick/place 高度位于 Feagine 椭球工作空间中部，而不是边界。
* [ ] 保证任务目标不会过于靠近底座奇异区域。
* [ ] 保证目标点不落在最大弯曲边界附近。

### 推荐任务布局参数

初始可以采用相对坐标，而不是立即固定绝对坐标：

```yaml
feagine_base:
  position: [0.0, 0.0, table_height]
  orientation: vertical_up

pick_shelf:
  side: left
  relative_position: [-workspace_radius_x * 0.5, 0.0, workspace_height * 0.45]

place_shelf:
  side: right
  relative_position: [workspace_radius_x * 0.5, 0.0, workspace_height * 0.45]

task_workspace:
  use_workspace_ratio: 0.55
```

### 验收标准

* [ ] 机械臂静止竖直向上时不与桌面或架子碰撞。
* [ ] pick shelf 和 place shelf 都位于可达工作空间内部。
* [ ] scripted expert 能完成左侧 reach 和右侧 reach。
* [ ] 末端运动不需要极端弯曲即可到达任务点。

---

## 2. 统一 4D Action 接口

## 2.1 顶层 action 定义

### 目标

将整个项目的顶层动作接口统一为：

```text
action = [dx, dy, dz, gripper_control]
```

其中：

```text
dx, dy, dz:
  末端在任务空间中的相对位移增量

gripper_control:
  夹爪开合控制
```

### 任务

* [x] 新增或重构 action space 定义：

```text
src/soft_continuum_vlm/envs/action_space.py
```

* [x] 设置 action range：

```text
dx, dy, dz ∈ [-1, 1]
gripper_control ∈ [-1, 1]
```

* [x] 在环境内部进行尺度映射：

```text
delta_xyz = action[:3] * delta_xyz_scale
```

* [ ] 初始建议：

```yaml
delta_xyz_scale: 0.01  # 每步最大 1 cm
```

后续可以根据仿真稳定性调整为：

```yaml
delta_xyz_scale: 0.005
delta_xyz_scale: 0.02
```

### 验收标准

* [ ] 所有任务环境的 `action_space.shape == (4,)`。
* [ ] RL、scripted expert、VLM planner 均输出 4 维 action。
* [ ] 不再暴露旧的 `section_angles` 顶层 action。

---

## 2.2 末端增量到 Feagine 底层控制的映射

### 目标

实现：

```text
[dx, dy, dz, gripper_control]
        ↓
desired_tip_pose
        ↓
Feagine IK / differential IK / PCC solver
        ↓
section_angles
grasper_rotation
gripper_open_close
```

### 任务

* [ ] 新增 action adapter：

```text
src/soft_continuum_vlm/controllers/feagine_action_adapter.py
```

* [ ] 实现接口：

```python
class FeagineActionAdapter:
    def convert(self, action_4d, observation):
        ...
```

* [ ] 输入：

```text
action_4d
current_tip_pose
current_section_angles
current_grasper_rotation
current_gripper_state
task_context
```

* [ ] 输出：

```text
section_angles
grasper_rotation
gripper_open_close
```

### 映射流程

```text
current_tip_position = get_tip_position()
delta_xyz = action[:3] * delta_xyz_scale
target_tip_position = current_tip_position + delta_xyz

section_angles = ik_solver.solve(
    target_tip_position,
    current_section_angles
)

grasper_rotation = grasper_orientation_controller.compute(
    target_tip_position,
    task_context
)

gripper_open_close = gripper_mapper.map(
    action[3]
)
```

### 验收标准

* [ ] 给定 4D action 可以稳定输出 Feagine 底层动作。
* [ ] `dx, dy, dz = 0` 时末端基本保持不动。
* [ ] `dx > 0` 时末端朝世界坐标或任务坐标的 +x 方向运动。
* [ ] `dy > 0`、`dz > 0` 同理可验证。
* [ ] `gripper_control > 0` 对应打开或关闭，需要统一约定。
* [ ] 旧的 section-level action 不再作为环境 action 暴露。

---

## 2.3 IK / 微分运动学模块

### 目标

支持至少两种映射方式：

```text
1. PCC 逆运动学
2. 微分运动学 Jacobian 映射
```

### 任务

* [x] 整理当前已有 PCC IK 模块。
* [x] 新增统一 IK 接口：

```text
src/soft_continuum_vlm/controllers/ik/base_ik_solver.py
src/soft_continuum_vlm/controllers/ik/pcc_ik_solver.py
src/soft_continuum_vlm/controllers/ik/differential_ik_solver.py
```

* [x] `PccIkSolver` 用于较大范围目标点求解。
* [x] `DifferentialIkSolver` 用于小步长连续控制。
* [x] 支持求解失败时回退：

```text
IK success:
  use solved section_angles

IK failed:
  reduce delta scale and retry

still failed:
  hold current action
```

### 验收标准

* [ ] IK 输出不超过机械臂允许弯曲范围。
* [ ] 目标点在工作空间内部时成功率较高。
* [ ] 目标点在工作空间边界外时能安全失败。
* [ ] 微分 IK 在小步长控制下轨迹连续。

---

## 3. Feagine-MetaWorld 风格任务体系

## 3.1 任务接口重构

### 目标

废弃旧任务接口，统一为 Feagine-MetaWorld 风格任务接口。

### 任务

* [ ] 新增任务基类：

```text
src/soft_continuum_vlm/tasks/feagine_metaworld_task.py
```

* [ ] 每个任务统一包含：

```text
reset_task()
compute_reward()
compute_success()
get_goal()
get_object_state()
get_task_info()
```

* [ ] 环境 step 统一流程：

```text
4D action
  ↓
FeagineActionAdapter
  ↓
low-level Feagine command
  ↓
MuJoCo simulation step
  ↓
observation
  ↓
reward / success / info
```

### 验收标准

* [ ] 所有任务使用统一 4D action。
* [ ] 所有任务输出统一 observation。
* [ ] 所有任务都有 reward、success、info。
* [ ] 不再维护旧任务入口。

---

## 3.2 第一批任务：Reach 类

### 目标

先实现最简单的末端到达任务，用于验证 4D action、IK 和 workspace。

### 任务

* [ ] 新增任务：

```text
feagine_reach_left
feagine_reach_right
feagine_reach_3d
```

* [ ] 左侧 reach 目标放在机械臂左侧工作空间。
* [ ] 右侧 reach 目标放在机械臂右侧工作空间。
* [ ] 3D reach 在椭球工作空间内部随机采样目标点。

### Reward

```text
reward = - distance(tip, goal)
success = distance(tip, goal) < threshold
```

### 验收标准

* [ ] 随机 action 不报错。
* [ ] scripted expert 可以完成 reach。
* [ ] IK 成功率可统计。
* [ ] 能输出每个 episode 的 tip-goal distance 曲线。

---

## 3.3 第二批任务：Push 类

### 目标

实现不依赖夹爪稳定抓取的接触操作任务。

### 任务

* [ ] 新增任务：

```text
feagine_push_left_to_right
feagine_push_right_to_left
feagine_contact_push
```

* [ ] 物体放在左侧或右侧架子/平台上。
* [ ] 目标区域放在另一侧。
* [ ] 末端通过接触推动物体移动。
* [ ] 记录接触力、穿透深度、物体位移。

### Reward

```text
reward =
  - distance(tip, object)
  - distance(object, goal)
  - contact_force_penalty
  + success_bonus
```

### 验收标准

* [ ] 物体可被 Feagine 末端推动。
* [ ] 接触力不会持续爆炸。
* [ ] 能统计 object-goal distance。
* [ ] 能区分成功推动和误碰撞。

---

## 3.4 第三批任务：Pick and Place 类

### 目标

基于 Feagine 竖直放置和左右侧架子布局，实现符合其椭球工作空间特点的 pick-place 任务。

### 推荐布局

```text
左侧架子：pick object
右侧架子：place target
中间：Feagine base 竖直向上
```

### 任务

* [ ] 新增任务：

```text
feagine_pick_left_place_right
feagine_pick_right_place_left
feagine_pick_shelf_place_shelf
```

* [ ] pick 点放在左侧中等高度架子上。
* [ ] place 点放在右侧中等高度架子上。
* [ ] 架子高度位于 Feagine 工作空间中部。
* [ ] 初期不要求复杂避障。
* [ ] 后续加入中间障碍物或狭窄通道。

### 任务阶段

```text
1. approach object
2. align grasper rotation
3. close gripper
4. lift object
5. move across workspace
6. align place pose
7. open gripper
8. retract
```

### 夹爪旋转策略

* [ ] 顶层 RL action 仍然只有 `gripper_control`。
* [ ] `grasper_rotation` 由底层任务阶段控制器自动计算。
* [ ] 初期可以使用规则策略：

```text
approach: grasper points to object
grasp: align with object principal axis
transport: keep object stable
place: align with place shelf
```

### 验收标准

* [ ] scripted expert 至少能稳定完成简单 pick-place。
* [ ] 物体能被夹起而不是只被推走。
* [ ] place 后物体落在目标区域。
* [ ] 记录 grasp success、lift success、place success。

---

## 3.5 第四批任务：Obstacle / Shelf / Drawer 类

### 目标

在基础 reach、push、pick-place 成功后，再加入更复杂的交互任务。

### 任务

* [ ] `feagine_reach_around_obstacle`
* [ ] `feagine_pick_place_with_obstacle`
* [ ] `feagine_shelf_insert`
* [ ] `feagine_button_press`
* [ ] `feagine_drawer_open`
* [ ] `feagine_door_or_hinge_push`

### 实现顺序

```text
1. reach around obstacle
2. push with obstacle
3. pick-place with obstacle
4. button press
5. shelf insert
6. drawer open
```

### 验收标准

* [ ] 每个任务都有 scripted expert baseline。
* [ ] 每个任务都有 RL 训练接口。
* [ ] 每个任务都有 success metric。
* [ ] 每个任务能输出视频和 JSON 结果。

---

## 4. RL 训练接口

## 4.1 Gymnasium 风格环境封装

### 目标

为 PPO、SAC、TD3 等 RL 算法提供标准接口。

### 任务

* [ ] 新增：

```text
src/soft_continuum_vlm/envs/feagine_gym_env.py
```

* [ ] 支持：

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

* [ ] `action_space` 固定为：

```python
Box(low=-1.0, high=1.0, shape=(4,))
```

* [ ] 初期 observation 使用 state-based，不直接用图像。

### State Observation 建议

```text
tip_position
tip_velocity
gripper_state
grasper_rotation
object_position
goal_position
contact_force
contact_flag
```

### 验收标准

* [ ] 可以被 stable-baselines3 或自定义 PPO/SAC 调用。
* [ ] action space 与 MetaWorld 一致为 4D。
* [ ] reset/step 不依赖旧接口。
* [ ] 支持 headless training。

---

## 4.2 RL Baseline

### 任务

* [ ] 新增训练脚本：

```text
scripts/train_rl_baseline.py
scripts/evaluate_rl_baseline.py
```

* [ ] 第一批训练任务：

```text
feagine_reach_left
feagine_reach_right
feagine_push_left_to_right
```

* [ ] 第二批训练任务：

```text
feagine_pick_left_place_right
feagine_pick_place_with_obstacle
```

### 算法顺序

```text
1. PPO
2. SAC
3. TD3
4. HER + SAC，适合 goal-conditioned reach/pick-place
```

### 验收标准

* [ ] reach 任务可以训练出明显收敛曲线。
* [ ] push 任务能超过随机策略。
* [ ] pick-place 至少能完成部分阶段。
* [ ] 能保存 checkpoint、reward curve、success curve。

---

## 5. 手眼相机与全局视觉反馈

## 5.1 全局相机

### 目标

提供 VLM 所需的全局场景观察。

### 任务

* [ ] 在 MuJoCo 场景中加入 global camera。
* [ ] global camera 能看到：

```text
Feagine base
左侧 pick shelf
右侧 place shelf
object
target
obstacle
```

* [ ] 支持 RGB 和 depth 输出。

### 验收标准

* [ ] 能保存 global RGB 图像。
* [ ] 能保存 global depth 图像。
* [ ] headless 模式下可用。
* [ ] 不依赖 viewer 手动观察。

---

## 5.2 手眼相机

### 目标

在 Feagine 末端或夹爪附近加载 eye-in-hand camera。

### 任务

* [ ] 将 wrist camera 固定到 grasper/tip body。
* [ ] camera 随 Feagine 末端运动。
* [ ] 输出 wrist RGB。
* [ ] 输出 wrist depth。
* [ ] 后续支持局部视觉伺服。

### 验收标准

* [ ] 机械臂运动时 wrist camera 视角同步变化。
* [ ] wrist camera 能看到夹爪前方区域。
* [ ] wrist camera 能观察到待抓取物体。
* [ ] wrist camera 不被 Feagine 自身模型严重遮挡。

---

## 5.3 多模态 Observation

### 新 Observation 结构

```python
observation = {
    "state": state_vector,
    "images": {
        "global": global_rgb,
        "wrist": wrist_rgb,
    },
    "depth": {
        "global": global_depth,
        "wrist": wrist_depth,
    },
    "language": instruction,
    "task": task_info,
}
```

### 任务

* [ ] 直接替换旧 observation，不再兼容旧结构。
* [ ] 所有 task/env/policy 统一使用新 observation。
* [ ] 图像数据支持保存到 demo dataset。

### 验收标准

* [ ] RL 可以选择只用 state。
* [ ] VLM/VLA 可以读取 image + language + state。
* [ ] demo 采集能保存多相机数据。

---

## 6. VLM / VLA 技术路线

## 6.1 第一阶段：VLM 只做高层规划

### 目标

VLM 不直接输出底层动作，而是输出任务计划。

```text
language + global image + wrist image
        ↓
VLM planner
        ↓
subgoals / target object / constraints
        ↓
skill controller
        ↓
4D action
        ↓
Feagine action adapter
```

### 输出格式

```json
{
  "task": "pick_place",
  "target_object": "red_cube",
  "source": "left_shelf",
  "target": "right_shelf",
  "constraints": {
    "avoid_collision": true,
    "max_contact_force": 5.0
  },
  "subgoals": [
    "move_to_pre_grasp",
    "align_grasper",
    "close_gripper",
    "lift",
    "move_to_place",
    "open_gripper"
  ]
}
```

### 任务

* [ ] 保留 deterministic planner 作为 baseline。
* [ ] 新增真实 VLM planner 接口。
* [ ] VLM 输出必须经过 schema validation。
* [ ] 非法输出回退到 deterministic planner。

---

## 6.2 第二阶段：VLM + Skill Policy

### 目标

VLM 负责决策，skill policy 负责控制。

### Skill 列表

```text
reach_skill
push_skill
pregrasp_skill
grasp_skill
lift_skill
place_skill
retract_skill
avoid_obstacle_skill
```

### 每个 skill 输出

```text
4D action = [dx, dy, dz, gripper_control]
```

### 任务

* [ ] 将 scripted expert 拆成多个 skill。
* [ ] 每个 skill 使用统一 action。
* [ ] VLM 根据图像和语言选择 skill sequence。
* [ ] Skill 内部可以使用状态反馈和视觉反馈。

---

## 6.3 第三阶段：VLA Policy

### 目标

训练或微调视觉-语言-动作模型，使其直接输出 4D action。

### 输入

```text
global RGB
wrist RGB
state vector
language instruction
```

### 输出

```text
action = [dx, dy, dz, gripper_control]
```

### 推荐路线

```text
1. 先做自定义小模型 imitation learning
2. 再导出 LeRobot / RLDS 格式
3. 再尝试 SmolVLA / Octo / OpenVLA 类模型
```

### 任务

* [ ] 采集 scripted demos。
* [ ] 保存图像、状态、语言、动作。
* [ ] 训练 behavior cloning baseline。
* [ ] 导出到 VLA 数据格式。
* [ ] 尝试接入 SmolVLA 或 Octo。
* [ ] 最后再考虑 OpenVLA。

---

## 7. Demo 采集与数据集建设

## 7.1 Scripted Expert Demo

### 任务

* [ ] 为 reach/push/pick-place 编写 scripted expert。
* [ ] 所有 expert 输出统一 4D action。
* [ ] 采集成功轨迹和失败轨迹。
* [ ] 保存每一步：

```text
state
global image
wrist image
depth
language
4D action
low-level Feagine command
reward
success
contact info
```

### 推荐数据规模

```text
reach: 100 episodes
push: 100 episodes
pick-place: 200 episodes
obstacle pick-place: 300 episodes
```

---

## 7.2 数据回放与调试

### 任务

* [ ] 新增 demo replay 脚本：

```text
scripts/replay_feagine_demo.py
```

* [ ] 支持逐步回放。
* [ ] 支持视频导出。
* [ ] 支持显示 global/wrist camera。
* [ ] 支持显示 action 和 low-level command。

### 验收标准

* [ ] 任意 demo 可以复现轨迹。
* [ ] 可以检查失败 demo 的具体失败阶段。
* [ ] 可以导出论文或报告用视频。

---

## 8. 评估指标与实验结果

## 8.1 基础指标

* [ ] success rate
* [ ] average return
* [ ] final distance
* [ ] episode length
* [ ] IK success rate
* [ ] action smoothness
* [ ] contact force
* [ ] max penetration
* [ ] gripper success rate
* [ ] lift success rate
* [ ] place success rate

## 8.2 对比方法

### 初期

```text
random policy
scripted expert
PPO
SAC
behavior cloning
```

### 中期

```text
VLM + scripted skills
VLM + learned skills
VLA behavior cloning
```

### 后期

```text
SmolVLA
Octo
OpenVLA
```

---

## 9. 推荐整体开发顺序

## Phase 1：动作接口与工作空间

* [ ] 统一 4D action。
* [ ] 废弃旧 action 接口。
* [ ] 实现 FeagineActionAdapter。
* [ ] 测试 PCC 工作空间。
* [ ] 确定竖直放置和左右架子任务布局。

## Phase 2：基础任务

* [ ] 实现 `feagine_reach_left`。
* [ ] 实现 `feagine_reach_right`。
* [ ] 实现 `feagine_reach_3d`。
* [ ] 实现 `feagine_push_left_to_right`。
* [ ] 实现 `feagine_push_right_to_left`。

## Phase 3：Pick-Place 任务

* [ ] 实现左侧 pick、右侧 place。
* [ ] 加入 grasper rotation 自动控制。
* [ ] 加入 lift/place 阶段判断。
* [ ] 实现 scripted expert。
* [ ] 采集 pick-place demos。

## Phase 4：RL Baseline

* [ ] 封装 Gymnasium 环境。
* [ ] PPO 训练 reach。
* [ ] SAC 训练 push。
* [ ] HER + SAC 尝试 pick-place。
* [ ] 输出训练曲线和评估结果。

## Phase 5：视觉系统

* [ ] 加入 global camera。
* [ ] 加入 wrist camera。
* [ ] observation 支持 image + depth。
* [ ] 保存视觉 demo。
* [ ] 实现简单视觉检测 baseline。

## Phase 6：VLM Planner

* [ ] VLM 输入 global/wrist 图像和语言。
* [ ] VLM 输出结构化任务计划。
* [ ] 计划转 skill sequence。
* [ ] skill sequence 输出 4D action。
* [ ] 完成 VLM + skill pick-place。

## Phase 7：VLA Policy

* [ ] 整理 demo 数据集。
* [ ] 训练 behavior cloning policy。
* [ ] 导出 LeRobot / RLDS 数据格式。
* [ ] 尝试 SmolVLA / Octo。
* [ ] 后续再尝试 OpenVLA。

---

## 10. 每次开发后的记录模板

每完成一个子任务，在对话中记录：

````markdown
## 本次完成内容

1. 修改文件：
   - `...`
   - `...`

2. 新增文件：
   - `...`
   - `...`

3. 运行命令：
   ```bash
   ...
````

4. 测试结果：

   * `pytest`: passed / failed
   * 任务运行结果：
   * 报错信息：

5. 当前问题：

   * ...

6. 下一步计划：

   * ...

````

---

## 11. 当前优先级最高的前三个任务

### TODO-1：统一 4D Action 接口

目标：

```text
action = [dx, dy, dz, gripper_control]
````

完成后所有 task、expert、RL、VLM 都基于这个接口。

---

### TODO-2：测试 Feagine 竖直放置下的 PCC 工作空间

目标：

```text
得到机械臂竖直放置时的椭球形可达空间
```

并基于结果确定：

```text
left pick shelf
right place shelf
workspace center
reachable task region
```

---

### TODO-3：实现 FeagineActionAdapter

目标：

```text
4D action
  ↓
tip target
  ↓
PCC IK / differential IK
  ↓
section_angles + grasper_rotation + gripper_open_close
```

这是后续 MetaWorld-style 任务、RL、VLM、VLA 的共同底层。
