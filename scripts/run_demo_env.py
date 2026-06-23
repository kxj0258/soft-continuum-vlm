from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv  # noqa: E402
from soft_continuum_vlm.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Feagine MuJoCo demo.")
    parser.add_argument(
        "--render-mode",
        choices=["human", "none"],
        default=None,
        help="Override the environment render mode. Default comes from the YAML config.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--steps", type=int, default=120, help="Number of demo steps to simulate.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Sleep seconds between human-viewer steps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs" / "env" / "feagine_mujoco_a03_type_2.yaml")
    if args.headless:
        config.setdefault("env", {})["render_mode"] = "none"
    elif args.render_mode is not None:
        config.setdefault("env", {})["render_mode"] = args.render_mode
    env = FeagineMujocoEnv(config)
    observation = env.reset(language="Pick up the red object with contact-safe motion.")
    section_angle_count = len(env.get_robot_state()["section_angles"])
    reward = 0.0
    done = False
    info = {"runtime": "feagine_mujoco", "applied_controls": []}
    for step_id in range(args.steps):
        bend = 0.15 if step_id > args.steps // 3 else 0.0
        _, reward, done, info = env.step(
            {
                "section_angles": [bend] * section_angle_count,
                "grip_command": 0.0,
                "grasper_rotation": 0.0,
            }
        )
        if env.config.render_mode == "human" and args.sleep > 0.0:
            time.sleep(args.sleep)
        if done:
            break
    print(f"[INFO] Observation keys: {sorted(observation.keys())}")
    print(f"[INFO] Feagine step reward={reward} done={done} runtime={info['runtime']}")
    print(f"[INFO] Applied controls: {info['applied_controls']}")
    print(f"[INFO] Render mode: {env.config.render_mode}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
