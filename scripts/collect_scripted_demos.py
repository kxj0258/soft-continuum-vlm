from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.scripted_expert import ScriptedExpert
from soft_continuum_vlm.data.schema import OBSERVATION_SCHEMA_VERSION
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.utils.config import load_yaml_config


SUPPORTED_TASK = "pick_red_object"
DEFAULT_LANGUAGE = "pick the red object"
FIELD_NAMES = (
    "proprioception",
    "tip_position",
    "section_angles",
    "grip_command",
    "grasper_rotation",
    "action_section_angles",
    "action_grip_command",
    "action_grasper_rotation",
    "language",
    "task_name",
    "phase",
    "target_distance",
    "best_distance_to_target",
    "post_close_drift",
    "reward",
    "done",
    "episode_id",
    "step_id",
    "success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect narrow-step scripted pick_red_object demonstrations.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--env", default="feagine_mujoco", choices=["mock", "feagine_mujoco"])
    parser.add_argument("--config")
    parser.add_argument("--save-config-snapshot", action="store_true")
    parser.add_argument("--calibration")
    parser.add_argument("--target-offset", nargs=3, type=float, default=[0.0, 0.02, 0.0])
    parser.add_argument("--close-loop-hold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-hold-position-gain-scale", type=float, default=0.35)
    parser.add_argument("--max-post-close-drift", type=float, default=0.02)
    return parser.parse_args()


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def make_env(args: argparse.Namespace):
    if args.mock_env or args.env == "mock":
        return MockContinuumEnv(task=args.task, max_steps=args.max_steps)
    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": "none",
                "max_episode_steps": max(int(args.max_steps) + 1, 2),
            }
        }
    return FeagineMujocoEnv(config)


def make_expert(args: argparse.Namespace, language: str) -> ScriptedExpert:
    expert = ScriptedExpert(
        task_name=SUPPORTED_TASK,
        calibration_path=args.calibration,
        target_offset=args.target_offset,
        close_loop_hold=args.close_loop_hold,
        close_hold_position_gain_scale=args.close_hold_position_gain_scale,
        max_post_close_drift=args.max_post_close_drift,
    )
    expert.reset(task_name=SUPPORTED_TASK, language=language)
    return expert


def reset_env(env: Any, args: argparse.Namespace, episode_id: int) -> dict[str, Any]:
    observation = env.reset(
        task=args.task,
        seed=int(args.seed) + int(episode_id),
        language=DEFAULT_LANGUAGE,
    )
    return dict(observation)


def step_env(
    env: Any,
    action: Mapping[str, Any],
    *,
    expert_phase: str,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    result = env.step(dict(action))
    reward: float | None = None
    done: bool | None = None
    info: dict[str, Any] = {}
    if isinstance(result, tuple):
        if len(result) == 5:
            observation, reward_value, terminated, truncated, info_value = result
            reward = float(reward_value)
            done = bool(terminated or truncated)
            info = dict(info_value or {})
        elif len(result) == 4:
            observation, reward_value, done_value, info_value = result
            reward = float(reward_value)
            done = bool(done_value)
            info = dict(info_value or {})
        elif len(result) == 2:
            observation, info_value = result
            info = dict(info_value or {})
        elif len(result) == 1:
            observation = result[0]
        else:
            raise ValueError(f"env.step returned {len(result)} values; expected observation, 4-tuple, or 5-tuple.")
    else:
        observation = result
    if reward is None:
        reward = 0.0
    if done is None:
        done = expert_phase == "done"
    else:
        done = bool(done or expert_phase == "done")
    return dict(observation), float(reward), bool(done), info


def robot_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    state = observation.get("robot_state", {})
    return dict(state) if isinstance(state, Mapping) else {}


def vector(value: Any, *, length: int | None = None) -> list[float]:
    array = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    if length is None:
        return [float(item) for item in array.tolist()]
    padded = np.zeros(length, dtype=np.float64)
    copy_count = min(length, array.size)
    if copy_count:
        padded[:copy_count] = array[:copy_count]
    return [float(item) for item in padded.tolist()]


def finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def tip_position(observation: Mapping[str, Any]) -> list[float]:
    tip_pose = robot_state(observation).get("tip_pose", {})
    position = tip_pose.get("position", []) if isinstance(tip_pose, Mapping) else []
    return vector(position, length=3)


def append_transition(
    episode_rows: list[dict[str, Any]],
    *,
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    expert_info: Mapping[str, Any],
    reward: float,
    done: bool,
    task_name: str,
    episode_id: int,
    step_id: int,
    language: str,
) -> None:
    state = robot_state(observation)
    section_angles = vector(state.get("section_angles", []))
    action_section_angles = vector(action.get("section_angles", []), length=len(section_angles) or None)
    episode_rows.append(
        {
            "proprioception": vector(observation.get("proprioception", [])),
            "tip_position": tip_position(observation),
            "section_angles": section_angles,
            "grip_command": float(state.get("grip_command", 0.0)),
            "grasper_rotation": float(state.get("grasper_rotation", 0.0)),
            "action_section_angles": action_section_angles,
            "action_grip_command": float(action.get("grip_command", 0.0)),
            "action_grasper_rotation": float(action.get("grasper_rotation", 0.0)),
            "language": language,
            "task_name": task_name,
            "phase": str(expert_info.get("phase", "")),
            "target_distance": finite_or_nan(expert_info.get("current_distance_to_target", expert_info.get("target_distance"))),
            "best_distance_to_target": finite_or_nan(expert_info.get("best_distance_to_target")),
            "post_close_drift": finite_or_nan(expert_info.get("post_close_drift")),
            "reward": float(reward),
            "done": bool(done),
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "success": False,
        }
    )


def finite_min(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return min(finite) if finite else float("nan")


def summarize_episode(
    *,
    episode_id: int,
    episode_rows: list[dict[str, Any]],
    phases: list[str],
    final_observation: Mapping[str, Any],
    final_action: Mapping[str, Any] | None,
    expert: ScriptedExpert,
    max_post_close_drift: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    reached_close_phase = "close_gripper" in phases
    best_distance = finite_min([float(row["best_distance_to_target"]) for row in episode_rows])
    final_distance = float(episode_rows[-1]["target_distance"]) if episode_rows else float("nan")
    post_close_drift = float(episode_rows[-1]["post_close_drift"]) if episode_rows else float("nan")
    final_state = robot_state(final_observation)
    final_grip_command = final_state.get("grip_command")
    if final_grip_command is None and final_action is not None:
        final_grip_command = final_action.get("grip_command")
    final_grip = finite_or_nan(final_grip_command)
    if not reached_close_phase:
        warnings.append("close_gripper phase was not reached")
    if not np.isfinite(best_distance):
        warnings.append("best_distance_to_target is missing")
    if not np.isfinite(final_distance):
        warnings.append("final target distance is missing")
    if not np.isfinite(post_close_drift):
        warnings.append("post_close_drift is missing")
    if not np.isfinite(final_grip):
        warnings.append("final_grip_command is missing")
    position_tolerance = float(getattr(expert, "position_tolerance", 0.006))
    success = (
        not warnings
        and reached_close_phase
        and final_grip >= 0.9
        and best_distance < position_tolerance
        and post_close_drift < float(max_post_close_drift)
    )
    return {
        "episode_id": int(episode_id),
        "success": bool(success),
        "steps": int(len(episode_rows)),
        "best_distance": best_distance,
        "final_distance": final_distance,
        "post_close_drift": post_close_drift,
        "final_grip_command": final_grip,
        "reached_close_phase": bool(reached_close_phase),
        "position_tolerance": position_tolerance,
        "warnings": warnings,
    }


def rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    vector_fields = {
        "proprioception",
        "tip_position",
        "section_angles",
        "action_section_angles",
    }
    float_fields = {
        "grip_command",
        "grasper_rotation",
        "action_grip_command",
        "action_grasper_rotation",
        "target_distance",
        "best_distance_to_target",
        "post_close_drift",
        "reward",
    }
    int_fields = {"episode_id", "step_id"}
    bool_fields = {"done", "success"}
    text_fields = {"language", "task_name", "phase"}
    for field in FIELD_NAMES:
        values = [row[field] for row in rows]
        if field in vector_fields:
            arrays[field] = np.asarray(values, dtype=np.float32)
        elif field in float_fields:
            arrays[field] = np.asarray(values, dtype=np.float32)
        elif field in int_fields:
            arrays[field] = np.asarray(values, dtype=np.int32)
        elif field in bool_fields:
            arrays[field] = np.asarray(values, dtype=bool)
        elif field in text_fields:
            arrays[field] = np.asarray(values)
        else:
            arrays[field] = np.asarray(values, dtype=object)
    return arrays


def write_outputs(
    *,
    output: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_config_snapshot: bool,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if save_config_snapshot:
        output.with_suffix(".config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def print_episode_summary(summary: Mapping[str, Any]) -> None:
    print(
        "[EP {episode_id:03d}] success={success} steps={steps} "
        "best_distance={best_distance:.9f} final_distance={final_distance:.9f} "
        "post_close_drift={post_close_drift:.9f} final_grip={final_grip_command:.6f}".format(
            **summary
        )
    )


def main() -> int:
    args = parse_args()
    if args.task != SUPPORTED_TASK:
        print(
            "Only pick_red_object is supported by ScriptedExpert demo collection in this narrow step.",
            file=sys.stderr,
        )
        return 2

    try:
        env = make_env(args)
    except Exception as exc:
        print(f"[FAIL] environment creation failed: {exc}", file=sys.stderr)
        return 1

    all_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    try:
        for episode_id in range(int(args.num_episodes)):
            try:
                observation = reset_env(env, args, episode_id)
                language = str(observation.get("language") or DEFAULT_LANGUAGE)
                expert = make_expert(args, language)
            except Exception as exc:
                print(f"[FAIL] episode {episode_id:03d} initialization failed: {exc}", file=sys.stderr)
                return 1

            episode_rows: list[dict[str, Any]] = []
            phases: list[str] = []
            final_action: dict[str, Any] | None = None
            final_observation = observation
            for step_id in range(int(args.max_steps)):
                action, expert_info = expert.act(final_observation)
                if "gripper_rotation" in action:
                    raise ValueError("Use 'grasper_rotation' instead of unsupported 'gripper_rotation'.")
                phase = str(expert_info.get("phase", ""))
                phases.append(phase)
                next_observation, reward, done, _env_info = step_env(
                    env,
                    action,
                    expert_phase=phase,
                )
                append_transition(
                    episode_rows,
                    observation=final_observation,
                    action=action,
                    expert_info=expert_info,
                    reward=reward,
                    done=done,
                    task_name=SUPPORTED_TASK,
                    episode_id=episode_id,
                    step_id=step_id,
                    language=language,
                )
                final_action = dict(action)
                final_observation = next_observation
                if done:
                    break

            summary = summarize_episode(
                episode_id=episode_id,
                episode_rows=episode_rows,
                phases=phases,
                final_observation=final_observation,
                final_action=final_action,
                expert=expert,
                max_post_close_drift=args.max_post_close_drift,
            )
            for row in episode_rows:
                row["success"] = bool(summary["success"])
            all_rows.extend(episode_rows)
            episode_summaries.append(summary)
            print_episode_summary(summary)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    output = Path(args.output)
    arrays = rows_to_arrays(all_rows)
    success_rate = (
        float(sum(1 for summary in episode_summaries if summary["success"]) / len(episode_summaries))
        if episode_summaries
        else 0.0
    )
    metadata = {
        "git_commit": git_commit_hash(),
        "command_line": sys.argv,
        "seed": int(args.seed),
        "task_name": SUPPORTED_TASK,
        "num_episodes": int(args.num_episodes),
        "max_steps": int(args.max_steps),
        "env_type": "mock" if args.mock_env or args.env == "mock" else "feagine_mujoco",
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "method_config": {"method": "scripted_expert", "virtual_target_pick": True},
        "expert_config": {
            "task_name": SUPPORTED_TASK,
            "calibration_path": args.calibration,
            "target_offset": [float(value) for value in args.target_offset],
            "close_loop_hold": bool(args.close_loop_hold),
            "close_hold_position_gain_scale": float(args.close_hold_position_gain_scale),
            "max_post_close_drift": float(args.max_post_close_drift),
        },
        "num_steps": int(len(all_rows)),
        "success_rate": success_rate,
        "episodes": episode_summaries,
        "fields": list(FIELD_NAMES),
        "notes": [
            "This dataset is generated from the virtual-target pick_red_object ScriptedExpert path.",
            "No RGB/depth frames are stored in this narrow step.",
        ],
    }
    try:
        metadata_path = write_outputs(
            output=output,
            arrays=arrays,
            metadata=metadata,
            save_config_snapshot=bool(args.save_config_snapshot),
        )
    except Exception as exc:
        print(f"[FAIL] output write failed: {exc}", file=sys.stderr)
        return 1

    print(f"[RESULT] episodes={len(episode_summaries)} success_rate={success_rate:.6f}")
    print(f"[OK] wrote {output}")
    print(f"[OK] wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
