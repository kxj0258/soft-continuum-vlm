from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.reach_expert import FeagineReachExpert  # noqa: E402
from soft_continuum_vlm.controllers.feagine_action_adapter import FeagineActionAdapter  # noqa: E402
from soft_continuum_vlm.controllers.ik import MujocoFkDifferentialIkSolver  # noqa: E402
from soft_continuum_vlm.envs.feagine_gym_state_env import FeagineGymStateEnv  # noqa: E402
from soft_continuum_vlm.envs.feagine_metaworld_env import FeagineMetaWorldEnv  # noqa: E402
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv  # noqa: E402
from soft_continuum_vlm.tasks import make_feagine_metaworld_task  # noqa: E402


REACH_TASKS = ("feagine_reach_left", "feagine_reach_right", "feagine_reach_3d")
IK_BACKENDS = ("approximate_pcc", "mujoco_fk")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 4D MetaWorld-style Reach smoke rollout on Feagine/MuJoCo."
    )
    parser.add_argument("--task", choices=REACH_TASKS, default="feagine_reach_right")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/feagine_reach_right_smoke.json",
        help="Path to the rollout diagnostics JSON.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use render_mode=none. This is the recommended smoke-validation mode.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("none", "human"),
        default=None,
        help="Override render mode. If omitted, --headless uses none and non-headless uses human.",
    )
    parser.add_argument(
        "--ik-backend",
        choices=IK_BACKENDS,
        default="mujoco_fk",
        help="Use approximate PCC IK or a MuJoCo-FK numeric Jacobian.",
    )
    return parser.parse_args()


def make_backend(
    *,
    headless: bool,
    backend_type: type[Any] = FeagineMujocoEnv,
    render_mode: str | None = None,
) -> Any:
    resolved_render_mode = render_mode or ("none" if headless else "human")
    return backend_type(
        {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": resolved_render_mode,
                "physics_steps_per_action": 4,
            }
        }
    )


def run_reach_rollout(
    *,
    task_name: str,
    steps: int,
    seed: int,
    output_path: str | Path,
    backend: Any,
    ik_backend: str = "mujoco_fk",
) -> dict[str, Any]:
    task = make_feagine_metaworld_task(task_name)
    env = make_gym_env(
        backend=backend,
        task=task,
        max_episode_steps=max(1, int(steps)),
        ik_backend=ik_backend,
    )
    expert = FeagineReachExpert()

    try:
        _state_observation, reset_info = env.reset(seed=seed)
        step_logs: list[dict[str, Any]] = []
        terminated = bool(reset_info.get("terminated", False))
        truncated = bool(reset_info.get("truncated", False))
        final_reward = 0.0
        final_info = dict(reset_info)
        final_state_observation = _state_observation

        for step_index in range(max(0, int(steps))):
            raw_observation = env.get_raw_observation()
            action_4d = expert.act(raw_observation)
            final_state_observation, reward, terminated, truncated, info = env.step(action_4d)
            final_reward = float(reward)
            final_info = dict(info)
            step_logs.append(
                {
                    "step": int(step_index),
                    "action_4d": _float_list(action_4d),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(info.get("success", False)),
                    "task_metrics": _jsonable(info.get("task_metrics", {})),
                    "task_info": _jsonable(info.get("task_info", {})),
                    "ik": _jsonable(info.get("ik", {})),
                    "low_level_command": _jsonable(info.get("low_level_command", {})),
                    "runtime_action": _jsonable(info.get("runtime_action", {})),
                    "tip_goal_distance_history": _jsonable(
                        info.get("tip_goal_distance_history", [])
                    ),
                }
            )
            if terminated or truncated:
                break

        report = {
            "task": task_name,
            "ik_backend": ik_backend,
            "seed": int(seed),
            "steps_requested": int(steps),
            "steps_executed": len(step_logs),
            "success": bool(final_info.get("success", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_reward": float(final_reward),
            "final_metrics": _jsonable(final_info.get("task_metrics", {})),
            "final_info": _jsonable(final_info),
            "final_observation": _jsonable(final_state_observation),
            "steps": step_logs,
        }
        _write_json(output_path, report)
        return report
    finally:
        env.close()


def make_gym_env(
    *,
    backend: Any,
    task: Any,
    max_episode_steps: int,
    ik_backend: str,
) -> FeagineGymStateEnv:
    if ik_backend == "approximate_pcc":
        return FeagineGymStateEnv(backend, task, max_episode_steps=max_episode_steps)
    if ik_backend != "mujoco_fk":
        raise ValueError(f"Unknown ik_backend: {ik_backend}")
    if not hasattr(backend, "probe_tip_position_for_section_angles"):
        raise ValueError("mujoco_fk IK requires a backend with probe_tip_position_for_section_angles().")
    adapter = FeagineActionAdapter(ik_solver=MujocoFkDifferentialIkSolver(backend))
    metaworld_env = FeagineMetaWorldEnv(backend, task, action_adapter=adapter)
    return FeagineGymStateEnv(
        backend,
        task,
        metaworld_env=metaworld_env,
        max_episode_steps=max_episode_steps,
    )


def _float_list(value: Sequence[float] | np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    try:
        backend = make_backend(
            headless=bool(args.headless),
            render_mode=args.render_mode,
        )
        report = run_reach_rollout(
            task_name=str(args.task),
            steps=int(args.steps),
            seed=int(args.seed),
            output_path=args.output,
            backend=backend,
            ik_backend=str(args.ik_backend),
        )
    except Exception as exc:
        print(f"[FAIL] Feagine MetaWorld Reach rollout failed: {exc}")
        print("[HINT] First check: python scripts/verify_feagine_install.py")
        return 1

    print(f"[OK] task={report['task']}")
    print(f"[OK] steps_executed={report['steps_executed']}")
    print(f"[OK] success={report['success']}")
    print(f"[OK] terminated={report['terminated']} truncated={report['truncated']}")
    print(f"[OK] wrote {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
