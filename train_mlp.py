"""Train an MLP for valence/arousal prediction on fused pretrained embeddings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from affective_dataset import AffectiveEmbeddingDataset, build_affective_dataloaders
from affective_model import AffectiveMLP
from config import CFG, ModelConfig
from model import PPGGSRCLIP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP on pretrained PPG/GSR embeddings")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with *_GSR.npy/*_PPG.npy/*_labels.npy")
    parser.add_argument("--encoder_ckpt", type=Path, default=Path("checkpoints/best_encoder.pt"))
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "sum", "mean"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default=CFG.train.device)
    parser.add_argument("--seed", type=int, default=CFG.train.seed)
    parser.add_argument("--save_path", type=Path, default=Path("checkpoints/affective_mlp_best.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_pretrained_clip(ckpt_path: Path, device: torch.device) -> PPGGSRCLIP:
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg_dict = checkpoint.get("model_config", {})
    model_cfg = ModelConfig(**cfg_dict) if cfg_dict else CFG.model
    clip_model = PPGGSRCLIP(model_cfg).to(device)

    if "ppg_encoder_state_dict" in checkpoint and "gsr_encoder_state_dict" in checkpoint:
        clip_model.ppg_encoder.load_state_dict(checkpoint["ppg_encoder_state_dict"])
        clip_model.gsr_encoder.load_state_dict(checkpoint["gsr_encoder_state_dict"])
    else:
        clip_model.load_state_dict(checkpoint["model_state_dict"])

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model


def run_epoch(model: AffectiveMLP, loader, device: torch.device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    for batch in loader:
        x = batch["feature"].to(device)
        y = batch["label"].to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += mae.item() * bs
        n += bs

    return {"mse": total_loss / max(n, 1), "mae": total_mae / max(n, 1)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    clip_model = load_pretrained_clip(args.encoder_ckpt, device)
    dataset = AffectiveEmbeddingDataset(args.data_dir, clip_model=clip_model, device=device, fusion=args.fusion)
    train_loader, val_loader, test_loader = build_affective_dataloaders(dataset, batch_size=args.batch_size, seed=args.seed)

    input_dim = dataset[0]["feature"].shape[-1]
    model = AffectiveMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        print(f"epoch={epoch:03d} train_mse={tr['mse']:.4f} train_mae={tr['mae']:.4f} val_mse={va['mse']:.4f} val_mae={va['mae']:.4f}")
        if va["mse"] < best_val:
            best_val = va["mse"]
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim, "args": vars(args)}, args.save_path)

    test = run_epoch(model, test_loader, device)
    print(f"test_mse={test['mse']:.4f} test_mae={test['mae']:.4f} total_time={time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
