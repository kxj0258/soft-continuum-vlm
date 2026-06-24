# 实验 010：Policy 评估

## 目的

通过统一 rollout runner 评估 `task_phase_expert`、`adapter` 和 `vlm_planner_ik`。

## mock smoke 命令

```powershell
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert vlm_planner_ik --mock-env --num-episodes 2 --max-steps 80 --seed 0 --output outputs/metrics/mock_policy_eval.json --csv-output outputs/metrics/mock_policy_eval.csv
```

## 真实候选命令

```powershell
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert adapter vlm_planner_ik --adapter-checkpoint outputs/checkpoints/adapter_best.pt --env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml --num-episodes 10 --max-steps 150 --seed 0 --output outputs/metrics/feagine_policy_eval.json --csv-output outputs/metrics/feagine_policy_eval.csv
```

mock 结果只用于 pipeline 检查。Feagine/MuJoCo 结果只有在场景名称和 runtime 可用性都验证后，才能作为论文实验候选。
