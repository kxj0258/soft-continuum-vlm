from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


def export_summary_artifacts(metrics: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = list(metrics.get("summary", []))
    csv_path = output / "summary_table.csv"
    md_path = output / "summary_table.md"
    fields = [
        "task",
        "baseline",
        "policy",
        "episodes",
        "success_rate",
        "mean_reward",
        "max_contact_force_mean",
        "max_penetration_mean",
        "safety_clip_count_mean",
        "safety_block_count_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    md_lines = [
        "| task | policy | episodes | success_rate | mean_reward | max_contact_force_mean | max_penetration_mean | safety_clip_count_mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        policy = row.get("policy", row.get("baseline", ""))
        md_lines.append(
            f"| {row.get('task', '')} | {policy} | {row.get('episodes', 0)} | "
            f"{float(row.get('success_rate', 0.0)):.3f} | {float(row.get('mean_reward', 0.0)):.3f} | "
            f"{float(row.get('max_contact_force_mean', 0.0)):.3f} | {float(row.get('max_penetration_mean', 0.0)):.3f} | "
            f"{float(row.get('safety_clip_count_mean', 0.0)):.3f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _bar_plot(summary, "success_rate", output / "success_rate_by_task.png", "Success rate")
    _bar_plot(summary, "max_contact_force_mean", output / "max_contact_force_by_task.png", "Max contact force")
    _bar_plot(summary, "max_penetration_mean", output / "penetration_by_task.png", "Penetration")
    _bar_plot(summary, "max_contact_force_mean", output / "contact_force_by_policy.png", "Contact force")
    _bar_plot(summary, "max_penetration_mean", output / "penetration_by_policy.png", "Penetration")
    _bar_plot(summary, "safety_clip_count_mean", output / "safety_clip_count_by_policy.png", "Safety clips")
    return [
        output / "success_rate_by_task.png",
        output / "max_contact_force_by_task.png",
        output / "penetration_by_task.png",
        output / "contact_force_by_policy.png",
        output / "penetration_by_policy.png",
        output / "safety_clip_count_by_policy.png",
        csv_path,
        md_path,
    ]


def _bar_plot(summary: list[Mapping[str, Any]], key: str, path: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row.get('task')}:{row.get('policy', row.get('baseline'))}" for row in summary] or ["none"]
    values = [float(row.get(key, 0.0)) for row in summary] or [0.0]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
