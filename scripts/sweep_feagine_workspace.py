from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.continuum_kinematics import ContinuumGeometry
from soft_continuum_vlm.workspace import sample_pcc_workspace, write_workspace_outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the deterministic PCC workspace and recommend left/right task regions."
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--section-count", type=int, default=3)
    parser.add_argument("--section-length", type=float, default=0.10)
    parser.add_argument("--max-abs-section-angle", type=float, default=0.8)
    parser.add_argument("--use-workspace-ratio", type=float, default=0.55)
    parser.add_argument("--output-dir", default="outputs/workspace")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    geometry = ContinuumGeometry(
        section_count=args.section_count,
        section_length=args.section_length,
        max_abs_section_angle=args.max_abs_section_angle,
    )
    try:
        samples = sample_pcc_workspace(
            num_samples=args.samples,
            seed=args.seed,
            geometry=geometry,
        )
        outputs = write_workspace_outputs(
            samples,
            args.output_dir,
            use_workspace_ratio=args.use_workspace_ratio,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[FAIL] workspace sweep failed: {exc}")
        return 1

    print(f"[OK] sampled {samples.tip_positions.shape[0]} reachable tip positions")
    print(f"[OK] rejected {samples.rejected_count} invalid positions")
    print(f"[OK] points: {outputs['points']}")
    print(f"[OK] report: {outputs['report']}")
    print(f"[OK] image: {outputs['image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
