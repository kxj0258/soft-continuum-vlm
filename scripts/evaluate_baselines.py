from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.evaluation.runner import run_baseline_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mock baselines.")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--language", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mock_env:
        raise ValueError("The first baseline evaluation implementation supports --mock-env only.")
    payload = run_baseline_suite(
        tasks=list(args.tasks),
        baselines=list(args.baselines),
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        language=args.language,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for item in payload["episodes"] for key in item})
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["episodes"])
    print(f"Saved baseline metrics to {output}")
    print(f"Saved baseline CSV to {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
