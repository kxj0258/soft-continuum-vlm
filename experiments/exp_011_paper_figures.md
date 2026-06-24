# 实验 011：论文图表导出

## 目的

只从保存的 JSON/CSV metrics 生成论文表格和图，不手工填写结果。

## 命令

```powershell
python scripts/export_paper_figures.py --metrics outputs/metrics/feagine_policy_eval.json --output-dir outputs/figures
```

## 输出文件

- `success_rate_by_task.png`
- `contact_force_by_policy.png`
- `penetration_by_policy.png`
- `safety_clip_count_by_policy.png`
- `summary_table.csv`
- `summary_table.md`
