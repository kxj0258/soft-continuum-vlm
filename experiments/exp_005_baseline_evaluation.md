# Experiment 005: Baseline Evaluation and Paper Figure Export

## Command

```bash
python scripts/evaluate_baselines.py \
  --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place \
  --baselines scripted_expert adapter vlm_planner_ik \
  --num-episodes 3 \
  --max-steps 60 \
  --mock-env \
  --output outputs/metrics/baseline_debug.json \
  --csv-output outputs/metrics/baseline_debug.csv

python scripts/export_paper_figures.py \
  --metrics outputs/metrics/baseline_debug.json \
  --output-dir outputs/figures
```

## Outputs

- `outputs/metrics/baseline_debug.json`
- `outputs/metrics/baseline_debug.csv`
- `outputs/figures/success_rate_by_task.png`
- `outputs/figures/max_contact_force_by_task.png`
- `outputs/figures/penetration_by_task.png`
- `outputs/figures/summary_table.csv`
- `outputs/figures/summary_table.md`

## Current Limit

The adapter baseline uses a random initialized policy when no checkpoint is
provided. These metrics are mock-env debug artifacts for validating the
evaluation pipeline, not final paper results.
