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
from soft_continuum_vlm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the soft embodiment adapter on scripted demos.")
    parser.add_argument("--demo", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/checkpoints/adapter_best.pt")
    parser.add_argument("--metrics-output", default="outputs/metrics/adapter_training.json")
    parser.add_argument("--loss", choices=["mse", "l1"], default="mse")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--config", default="configs/method/soft_embodiment_adapter.yaml")
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
    from torch.utils.data import DataLoader, random_split

    args = parse_args()
    if args.config and Path(args.config).exists():
        _ = load_yaml_config(args.config)
    dataset = DemoDataset(args.demo)
    device = choose_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    num_train, num_val = split_counts(len(dataset), args.train_ratio, args.val_ratio)
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = SoftEmbodimentAdapter(
        vision_language_dim=64,
        proprioception_dim=dataset.proprioception_dim(),
        contact_dim=dataset.contact_dim(),
        morphology_dim=dataset.morphology_dim(),
        hidden_dim=args.hidden_dim,
        action_dim=dataset.action_dim(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.L1Loss() if args.loss == "l1" else torch.nn.MSELoss()
    train_loss_curve: list[float] = []
    val_loss_curve: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(args.epochs):
        model.train()
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                grad_clip=args.grad_clip,
            )
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                Path(args.output).with_name(f"{Path(args.output).stem}_epoch_{epoch + 1}.pt"),
                model_state=model.state_dict(),
                args=args,
                dataset=dataset,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
            )
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "epoch_losses": train_loss_curve,
        "final_loss": train_loss_curve[-1] if train_loss_curve else 0.0,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else 0.0,
        "best_epoch": best_epoch,
        "num_train": num_train,
        "num_val": num_val,
        "num_samples": len(dataset),
        "dataset_path": str(Path(args.demo)),
        "action_schema": list(ACTION_KEYS),
        "input_dims": input_dims(dataset),
        "device": str(device),
    }
    output = Path(args.output)
    save_checkpoint(output, model_state=best_state, args=args, dataset=dataset, metrics=metrics)
    default_best = Path("outputs/checkpoints/adapter_best.pt")
    if output.resolve() != default_best.resolve():
        save_checkpoint(default_best, model_state=best_state, args=args, dataset=dataset, metrics=metrics)
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {output}")
    print(f"Saved metrics to {metrics_output}")
    return 0


def split_counts(length: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if length <= 0:
        raise ValueError("Cannot train adapter on an empty dataset.")
    total = float(train_ratio) + float(val_ratio)
    if total <= 0.0:
        raise ValueError("train-ratio + val-ratio must be positive.")
    val_count = int(round(length * float(val_ratio) / total))
    if length > 1:
        val_count = min(max(1, val_count), length - 1)
    else:
        val_count = 0
    return length - val_count, val_count


def run_epoch(*, model, loader, criterion, device, optimizer, grad_clip: float) -> float:
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
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0.0:
                    import torch

                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    return total_loss / max(1, total_items)


def input_dims(dataset: DemoDataset) -> dict[str, int]:
    return {
        "vision_language_dim": 64,
        "proprioception_dim": dataset.proprioception_dim(),
        "contact_dim": dataset.contact_dim(),
        "morphology_dim": dataset.morphology_dim(),
        "action_dim": dataset.action_dim(),
    }


def save_checkpoint(path: Path, *, model_state, args: argparse.Namespace, dataset: DemoDataset, metrics: dict) -> None:
    checkpoint = {
        "model_state_dict": model_state,
        "config": {
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "loss": args.loss,
            "weight_decay": args.weight_decay,
        },
        "action_schema": list(ACTION_KEYS),
        "input_dims": input_dims(dataset),
        "metrics": metrics,
        "demo_path": str(Path(args.demo)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(checkpoint, path)


if __name__ == "__main__":
    raise SystemExit(main())
