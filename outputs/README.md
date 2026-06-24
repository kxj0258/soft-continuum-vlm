# outputs/

本目录保存训练、评估和图表导出的生成物。

- `checkpoints/`：adapter checkpoint，例如 `adapter_best.pt` 或 `adapter_debug.pt`。
- `metrics/`：训练和评估产生的 JSON/CSV 指标。
- `figures/`：由 metrics 生成的论文图表。
- `rollouts/`：rollout 摘要和逐步轨迹记录。

生成物默认被 Git 忽略。论文图表必须从保存的 JSON/CSV 指标生成。
