# 实验 006：真实 Feagine 场景检查

## 目的

把真实 Feagine/MuJoCo 中的 body、geom、site 和 contact 名称绑定到任务、专家控制器和数据集使用的结构化 observation schema。

## 命令

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/a03_type_2_scene.json
```

## 预期输出

JSON 摘要会记录 model 计数、robot_state keys、qpos/qvel shape、场景对象可用性、contact 统计、section_count 和 action dimension。

如果某个配置对象无法解析，observation 中仍保留该对象，并写入 `available: false` 和 `missing_reason`。根据输出中的 body/geom 名称摘要，人工修正 YAML 里的 `body_names` 或 `geom_name_patterns`。

## 注意

mock 环境结果只用于 CI 和调试。真实论文 rollout 必须使用安装好的 `feagine_mujoco`、`mujoco` 和 `pyfeagine_sim_core`。
