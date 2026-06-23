from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.evaluation.plotting import export_summary_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-draft figures from metrics JSON.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    paths = export_summary_artifacts(metrics, args.output_dir)
    for path in paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
