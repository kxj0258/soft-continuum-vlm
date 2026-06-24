from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.calibrated_jacobian import JacobianCalibration
from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test PccIkController with a calibration JSON.")
    parser.add_argument("--calibration", required=True)
    return parser.parse_args()


def _target(base_tip: list[float], delta: list[float]) -> list[float]:
    return (np.asarray(base_tip, dtype=np.float64) + np.asarray(delta, dtype=np.float64)).tolist()


def main() -> int:
    args = _parse_args()
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        print(f"[FAIL] calibration file does not exist: {calibration_path}")
        return 1

    calibration = JacobianCalibration.load_json(calibration_path)
    controller = PccIkController(calibration_path=calibration_path)
    current_tip = calibration.base_tip_position
    robot_state = {
        "tip_pose": {"position": current_tip},
        "section_angles": calibration.base_section_angles,
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }
    targets = {
        "target_y_plus": _target(current_tip, [0.0, 0.02, 0.0]),
        "target_y_minus": _target(current_tip, [0.0, -0.02, 0.0]),
        "target_z_plus": _target(current_tip, [0.0, 0.0, 0.01]),
    }

    for name, target_tip_position in targets.items():
        action, info = controller.compute_action_with_info(
            {"target_tip_position": target_tip_position},
            robot_state,
        )
        print(f"[SMOKE] target={name}")
        print(f"  error={info['position_error']}")
        print(f"  active_columns={info['active_columns']}")
        print(f"  dq={info['dq']}")
        print(f"  section_angles_after={action['section_angles']}")
        print(f"  status={info['status']}")
    print("[OK] PccIkController smoke completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
