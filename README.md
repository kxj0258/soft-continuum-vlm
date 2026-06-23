# soft-continuum-vlm

本仓库是机器人论文项目 **Embodiment-Aware Vision-Language Manipulation for
Contact-Safe Soft Continuum Arms** 的研究代码。

中文题目方向：**面向接触安全操作的软体连续体机械臂视觉语言具身适配方法**。

这个项目不是普通软件 demo。当前阶段优先把 Feagine MuJoCo 仿真、任务定义、
安全控制接口和可复现实验骨架搭稳；VLM、OpenVLA、Octo 等大模型组件在明确
集成里程碑之前都保持 deterministic stub，不下载大型权重。

## 当前进度

- Milestone 0：项目骨架和 Feagine 安装验证已完成。
- Milestone 1：Feagine MuJoCo headless wrapper 已完成最小 reset/step。
- Milestone 2：四个任务已有确定性评估逻辑。
- Milestone 3：PCC IK / scripted expert / safety projector 已有最小可测试控制路径。

尚未完成：

- 真实物体检测或 VLM planner。
- 专家数据批量采集。
- adapter 训练循环、checkpoint 和评估图表。
- OpenVLA/OCTO 类 baseline 的真实接入。

## 目录结构

```text
configs/      环境、任务和方法配置。
scripts/      安装、验证、demo、数据采集和训练入口。
src/          可导入的 Python 包。
tests/        不依赖 MuJoCo 图形窗口的单元测试。
experiments/  实验记录和复现实验说明。
```

Feagine 必须保留在本仓库外部。默认路径是同级目录：

- `../feagine_simulation`
- `../feagine-simulation`

如果你的 Feagine 目录在别处，请设置 `FEAGINE_SIM_ROOT`。

## 安装与验证

推荐环境：

```bash
conda activate feagine_vlm
pip install -e .
```

验证 Feagine：

```bash
python scripts/verify_feagine_install.py
```

运行测试：

```bash
pytest
```

运行 headless MuJoCo demo：

```bash
python scripts/run_demo_env.py
```

如果需要从发行包重新安装 Feagine：

```bash
bash scripts/setup_feagine_env.sh
```

脚本会使用当前 Python 环境，构建 `feagine-simulation-core`，安装
`feagine_mujoco-*.whl`，并在设置 `INSTALL_FEAGINE_SAPIEN=1` 时尝试安装
SAPIEN wheel。

## Observation Schema

Milestone 2 采用结构化 observation，不依赖真实视觉模型。也就是说，任务评估
先读取 MuJoCo/Feagine 或测试 fixture 提供的状态字段，后续视觉模型只负责估计
这些字段。

核心结构如下：

```python
observation = {
    "rgb": ...,
    "depth": ...,
    "language": "...",
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

注意字段名使用 Feagine 的 `grasper_rotation`，不是 `gripper_rotation`。
`SafetyProjector` 会裁剪 section angles、grip command 和 grasper rotation；
当接触力或穿透深度超过限制时，会冻结连续体运动和 grasper 旋转，但保留
`grip_command`。

## 开发约束

- 不要复制或修改 `../feagine_simulation` / `../feagine-simulation`。
- 所有路径必须支持相对项目位置或 `FEAGINE_SIM_ROOT`。
- 测试不能依赖 MuJoCo 图形窗口。
- 如果 MuJoCo 或 Feagine 不可用，相关测试必须 graceful skip 或使用 fake runtime。
- 不下载 OpenVLA、Octo 或大型模型权重。
- 每次代码改动后运行相关测试，并做阶段性 Git 提交。
