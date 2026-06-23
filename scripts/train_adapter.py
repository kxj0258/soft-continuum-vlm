from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.data.dataset import DemoDataset
from soft_continuum_vlm.data.schema import ACTION_KEYS
from soft_continuum_vlm.models.soft_embodiment_adapter import SoftEmbodimentAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the soft embodiment adapter on scripted demos.")
    parser.add_argument("--demo", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--loss", choices=["mse", "l1"], default="mse")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    from torch.utils.data import DataLoader

    args = parse_args()
    dataset = DemoDataset(args.demo)
    device = choose_device(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SoftEmbodimentAdapter(
        vision_language_dim=64,
        proprioception_dim=dataset.proprioception_dim(),
        contact_dim=dataset.contact_dim(),
        morphology_dim=dataset.morphology_dim(),
        hidden_dim=args.hidden_dim,
        action_dim=dataset.action_dim(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss() if args.loss == "l1" else torch.nn.MSELoss()
    epoch_losses: list[float] = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            language_feature = batch["language_feature"].float().to(device)
            proprioception = batch["proprioception"].float().to(device)
            contact_state = batch["contact"].float().to(device)
            morphology = batch["morphology"].float().to(device)
            target = batch["action_vector"].float().to(device)
            prediction = model(
                vision_language_feature=language_feature,
                proprioception=proprioception,
                contact_state=contact_state,
                morphology=morphology,
            )
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
        mean_loss = total_loss / max(1, total_items)
        epoch_losses.append(mean_loss)
        print(f"epoch={epoch + 1} loss={mean_loss:.6f}")

    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "num_samples": len(dataset),
        "device": str(device),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "loss": args.loss,
        },
        "action_schema": list(ACTION_KEYS),
        "input_dims": {
            "vision_language_dim": 64,
            "proprioception_dim": dataset.proprioception_dim(),
            "contact_dim": dataset.contact_dim(),
            "morphology_dim": dataset.morphology_dim(),
            "action_dim": dataset.action_dim(),
        },
        "metrics": metrics,
        "demo_path": str(Path(args.demo)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {output}")
    print(f"Saved metrics to {metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
