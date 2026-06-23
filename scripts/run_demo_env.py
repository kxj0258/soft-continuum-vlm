from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv  # noqa: E402
from soft_continuum_vlm.utils.config import load_yaml_config  # noqa: E402


def main() -> int:
    config = load_yaml_config(PROJECT_ROOT / "configs" / "env" / "feagine_mujoco_a03_type_2.yaml")
    env = FeagineMujocoEnv(config)
    observation = env.reset(language="Pick up the red object with contact-safe motion.")
    section_angle_count = len(env.get_robot_state()["section_angles"])
    _, reward, done, info = env.step(
        {
            "section_angles": [0.0] * section_angle_count,
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        }
    )
    print(f"[INFO] Observation keys: {sorted(observation.keys())}")
    print(f"[INFO] Feagine step reward={reward} done={done} runtime={info['runtime']}")
    print(f"[INFO] Applied controls: {info['applied_controls']}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
