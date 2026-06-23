# soft-continuum-vlm

本仓库是机器人论文项目 **Embodiment-Aware Vision-Language Manipulation for Contact-Safe Soft Continuum Arms** 的研究代码。

中文题目方向：**面向接触安全操作的软体连续体机械臂视觉语言具身适配方法**。

这不是普通软件 demo。当前阶段优先把 Feagine MuJoCo 仿真、任务定义、安全控制接口、专家数据采集和小型 adapter 训练管线搭稳。VLM、OpenVLA、Octo 等大模型组件在明确集成里程碑之前都保持 deterministic stub，不下载大型权重。

## 当前进度

- Milestone 0：项目骨架和 Feagine 安装验证已完成。
- Milestone 1：Feagine MuJoCo headless wrapper 已完成最小 reset/step。
- Milestone 2：四个任务已有确定性评估逻辑。
- Milestone 3：PCC IK / scripted expert / safety projector 已有最小可测试控制路径。
- Milestone 4：mock scripted expert 数据采集管线已可生成 `.npz` demo 和 metadata。
- Milestone 5：Soft Embodiment Adapter 已支持 CPU tiny training、checkpoint 和 metrics 输出。

尚未完成：

- 真实物体检测或 VLM planner。
- 基于真实 Feagine MuJoCo 场景状态的专家数据批量采集。
- adapter 的正式训练配置、评估图表和论文级实验统计。
- OpenVLA/OCTO 类 baseline 的真实接入。

## 目录结构

```text
configs/      环境、任务和方法配置。
data/         本地生成的数据目录；demo 文件默认被 Git 忽略。
scripts/      安装、验证、demo、数据采集、训练和评估入口。
src/          可导入的 Python 包。
tests/        不依赖 MuJoCo 图形窗口的单元测试。
experiments/  实验记录和复现实验说明。
outputs/      checkpoint、metrics、figures、rollouts 等生成物目录。
```

Feagine 必须保留在本仓库外部。默认路径是同级目录：

- `../feagine_simulation`
- `../feagine-simulation`

如果 Feagine 目录在别处，请设置 `FEAGINE_SIM_ROOT`。

## 快速验证

推荐先激活环境：

```bash
conda activate feagine_vlm
pip install -e .
```

验证 Feagine：

```bash
python scripts/verify_feagine_install.py
```

运行单元测试：

```bash
pytest
```

运行 headless MuJoCo demo：

```bash
python scripts/run_demo_env.py
```

运行 mock 专家数据采集：

```bash
python scripts/collect_scripted_demos.py \
  --task obstacle_avoid_pick \
  --num-episodes 4 \
  --max-steps 50 \
  --output data/demos/debug_obstacle_avoid_pick.npz \
  --mock-env \
  --seed 0
```

运行 adapter tiny training：

```bash
python scripts/train_adapter.py \
  --demo data/demos/debug_obstacle_avoid_pick.npz \
  --epochs 3 \
  --batch-size 8 \
  --output outputs/checkpoints/adapter_debug.pt \
  --metrics-output outputs/metrics/adapter_debug.json
```

运行 adapter mock evaluation：

```bash
python scripts/evaluate_adapter.py \
  --checkpoint outputs/checkpoints/adapter_debug.pt \
  --task obstacle_avoid_pick \
  --num-episodes 2 \
  --max-steps 30 \
  --mock-env \
  --output outputs/metrics/eval_adapter_debug.json
```

## Observation Schema

Milestone 2 以后采用结构化 observation，不依赖真实视觉模型。任务评估、mock 环境、数据采集和 adapter 训练先读取 MuJoCo/Feagine 或测试 fixture 提供的状态字段；后续视觉模型只负责估计这些字段。

核心结构如下：

```python
observation = {
    "rgb": ...,
    "depth": ...,
    "language": "...",
    "proprioception": ...,
    "robot_state": {
        "tip_pose": {
            "position": [x, y, z],
            "orientation": [w, x, y, z],
        },
        "section_angles": [...],
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    },
    "objects": {
        "red_object": {
            "pose": {
                "position": [x, y, z],
                "orientation": [w, x, y, z],
            },
            "grasped": False,
        }
    },
    "contact": {
        "max_force": 0.0,
        "max_penetration": 0.0,
        "contacts": [
            {
                "geom1": "...",
                "geom2": "...",
                "position": [x, y, z],
                "normal": [nx, ny, nz],
                "force": [fx, fy, fz],
                "distance": 0.0,
            }
        ],
    },
}
```

四个任务的最小评估规则：

- `pick_red_object`：目标物体已抓取，且高度超过 lift 阈值。
- `obstacle_avoid_pick`：目标已抓取，且与 obstacle 的接触力不超过阈值。
- `contact_push`：目标物体进入目标区域，且最大接触力不超过安全阈值。
- `rotate_and_place`：目标物体位置误差和姿态误差均低于阈值。

## Action Schema

项目内部 action 字段直接对齐 Feagine MuJoCo runtime：

```python
action = {
    "section_angles": [...],
    "grip_command": 0.0,
    "grasper_rotation": 0.0,
}
```

字段名必须使用 Feagine 的 `grasper_rotation`，不是 `gripper_rotation`。`SafetyProjector` 会裁剪 `section_angles`、`grip_command` 和 `grasper_rotation`；当接触力或穿透深度超过限制时，会冻结连续体运动和 grasper 旋转，但保留 `grip_command`。

## Dataset Schema

`scripts/collect_scripted_demos.py` 生成 `.npz` demo 文件，并保存同名 `.json` metadata。每个 step 包含：

- `proprioception`
- `contact`
- `language`
- `language_feature`
- `morphology`
- `action`
- `action_vector`
- `reward`
- `done`
- `success`
- `task_name`
- `phase`
- `episode_id`
- `step_id`

metadata 包含 git commit、命令行、seed、task/method/env 配置、action schema 和 observation schema version。

## 开发约束

- 不要复制或修改 `../feagine_simulation` / `../feagine-simulation`。
- Feagine 只能通过安装脚本、相对路径或 `FEAGINE_SIM_ROOT` 引用。
- 所有路径必须支持相对项目位置。
- 测试不能依赖 MuJoCo 图形窗口。
- 如果 MuJoCo 或 Feagine 不可用，相关测试必须 graceful skip 或使用 fake/mock runtime。
- 不下载 OpenVLA、Octo 或大型模型权重。
- 每次代码改动后运行相关测试，并做阶段性 Git 提交。
