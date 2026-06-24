# 实验 005：Baseline 评估与论文图表导出

## 命令

```powershell
python scripts/evaluate_baselines.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --baselines scripted_expert adapter vlm_planner_ik --num-episodes 3 --max-steps 60 --mock-env --output outputs/metrics/baseline_debug.json --csv-output outputs/metrics/baseline_debug.csv

python scripts/export_paper_figures.py --metrics outputs/metrics/baseline_debug.json --output-dir outputs/figures
```

## 输出文件

- `outputs/metrics/baseline_debug.json`
- `outputs/metrics/baseline_debug.csv`
- `outputs/figures/success_rate_by_task.png`
- `outputs/figures/max_contact_force_by_task.png`
- `outputs/figures/penetration_by_task.png`
- `outputs/figures/summary_table.csv`
- `outputs/figures/summary_table.md`

## 当前限制

没有 checkpoint 时，adapter baseline 会使用随机初始化 policy 并记录 warning。这些指标是 mock-env debug artifact，只用于验证评估 pipeline，不是最终论文结果。
