from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.data.features import build_morphology_vector, encode_language_stub
from soft_continuum_vlm.data.schema import flatten_contact, flatten_proprioception, unflatten_action
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.models.soft_embodiment_adapter import SoftEmbodimentAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an adapter checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def main() -> int:
    import torch

    args = parse_args()
    if not args.mock_env:
        raise ValueError("The first evaluation implementation supports --mock-env only.")
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    input_dims = checkpoint["input_dims"]
    model = SoftEmbodimentAdapter(
        vision_language_dim=int(input_dims["vision_language_dim"]),
        proprioception_dim=int(input_dims["proprioception_dim"]),
        contact_dim=int(input_dims["contact_dim"]),
        morphology_dim=int(input_dims["morphology_dim"]),
        hidden_dim=int(checkpoint["config"]["hidden_dim"]),
        action_dim=int(input_dims["action_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rewards: list[float] = []
    successes: list[bool] = []
    contact_forces: list[float] = []
    penetrations: list[float] = []
    env = MockContinuumEnv(task=args.task, max_steps=args.max_steps)
    try:
        for episode_id in range(args.num_episodes):
            obs = env.reset(task=args.task, seed=episode_id)
            episode_success = False
            for _ in range(args.max_steps):
                language_feature = torch.as_tensor(
                    encode_language_stub(str(obs["language"]), dim=int(input_dims["vision_language_dim"])),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                proprioception = torch.as_tensor(
                    flatten_proprioception(obs), dtype=torch.float32, device=device
                ).unsqueeze(0)
                contact_state = torch.as_tensor(
                    flatten_contact(obs["contact"]), dtype=torch.float32, device=device
                ).unsqueeze(0)
                morphology = torch.as_tensor(
                    build_morphology_vector(section_count=3), dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action_vector = model(
                        vision_language_feature=language_feature,
                        proprioception=proprioception,
                        contact_state=contact_state,
                        morphology=morphology,
                    ).cpu().numpy()[0]
                action_vector = np.clip(action_vector, -1.0, 1.0)
                action = unflatten_action(action_vector, section_count=3)
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                contact_forces.append(float(obs["contact"].get("max_force", 0.0)))
                penetrations.append(float(obs["contact"].get("max_penetration", 0.0)))
                episode_success = episode_success or bool(info.get("success", False))
                if done:
                    break
            successes.append(episode_success)
    finally:
        env.close()

    metrics = {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_contact_force": float(np.mean(contact_forces)) if contact_forces else 0.0,
        "max_contact_force": float(np.max(contact_forces)) if contact_forces else 0.0,
        "mean_penetration": float(np.mean(penetrations)) if penetrations else 0.0,
        "num_episodes": args.num_episodes,
        "num_steps": len(rewards),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
