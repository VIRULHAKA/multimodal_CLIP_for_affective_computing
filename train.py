"""Train PPG/GSR transformer encoders with a CLIP-style contrastive loss."""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from config import CFG, Config
from dataset import build_dataloaders
from model import PPGGSRCLIP, clip_contrastive_loss, pair_rank_metrics, retrieval_accuracy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP-style PPG/GSR transformer encoders")
    parser.add_argument("--data_dir", type=Path, default=CFG.data.data_dir)
    parser.add_argument("--window_size", type=int, default=CFG.data.window_size)
    parser.add_argument("--stride", type=int, default=CFG.data.stride)
    parser.add_argument("--batch_size", type=int, default=CFG.data.batch_size)
    parser.add_argument("--epochs", type=int, default=CFG.train.epochs)
    parser.add_argument("--lr", type=float, default=CFG.train.learning_rate)
    parser.add_argument("--device", type=str, default=CFG.train.device)
    parser.add_argument("--checkpoint_dir", type=Path, default=CFG.train.checkpoint_dir)
    parser.add_argument("--split_by_subject", action="store_true", help="Split train/val by subject (not by sample)")
    parser.add_argument("--use_ddp", action="store_true", help="Enable DistributedDataParallel (launch with torchrun)")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.data.data_dir = args.data_dir
    cfg.data.window_size = args.window_size
    cfg.data.stride = args.stride
    cfg.data.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.learning_rate = args.lr
    cfg.train.device = args.device
    cfg.train.checkpoint_dir = args.checkpoint_dir
    return cfg


def init_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    if not args.use_ddp:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        return False, 0, 1, 0, device

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA. Please run with GPUs and torchrun.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return True, local_rank, world_size, rank, device


def is_main_process(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0


def unwrap_model(model: PPGGSRCLIP | DDP) -> PPGGSRCLIP:
    return model.module if isinstance(model, DDP) else model


def run_epoch(
    model: PPGGSRCLIP | DDP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_ppg_acc = 0.0
    total_gsr_acc = 0.0
    total_ppg_mean_rank = 0.0
    total_ppg_median_rank = 0.0
    total_gsr_mean_rank = 0.0
    total_gsr_median_rank = 0.0
    total_batches = 0

    for batch_dix, batch in enumerate(loader):
        ppg = batch["ppg"].to(device, non_blocking=True)
        gsr = batch["gsr"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits, ppg_z, gsr_z = model(ppg, gsr)
            loss = clip_contrastive_loss(logits, ppg_z, gsr_z)
            ppg_acc, gsr_acc = retrieval_accuracy(logits)
            ppg_mean_rank, ppg_median_rank, gsr_mean_rank, gsr_median_rank = pair_rank_metrics(logits)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_ppg_acc += ppg_acc.item()
        total_gsr_acc += gsr_acc.item()
        total_ppg_mean_rank += ppg_mean_rank.item()
        total_ppg_median_rank += ppg_median_rank.item()
        total_gsr_mean_rank += gsr_mean_rank.item()
        total_gsr_median_rank += gsr_median_rank.item()
        total_batches += 1

        if batch_dix % 100 == 0:
            with torch.no_grad():
                sim_matrix = ppg_z @ ppg_z.T  # 同模态内部相似度
                off_diag = sim_matrix[~torch.eye(len(sim_matrix), dtype=bool)].mean()
                print(f"PPG intra-modal mean cosine sim: {off_diag:.4f}")

    return {
        "loss": total_loss / max(total_batches, 1),
        "ppg_to_gsr_acc": total_ppg_acc / max(total_batches, 1),
        "gsr_to_ppg_acc": total_gsr_acc / max(total_batches, 1),
        "ppg_mean_rank": total_ppg_mean_rank / max(total_batches, 1),
        "ppg_median_rank": total_ppg_median_rank / max(total_batches, 1),
        "gsr_mean_rank": total_gsr_mean_rank / max(total_batches, 1),
        "gsr_median_rank": total_gsr_median_rank / max(total_batches, 1),
    }


def save_checkpoint(
    path: Path,
    model: PPGGSRCLIP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        },
        path,
    )


def save_encoder_checkpoint(path: Path, model: PPGGSRCLIP, epoch: int, cfg: Config, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "ppg_encoder_state_dict": model.ppg_encoder.state_dict(),
            "gsr_encoder_state_dict": model.gsr_encoder.state_dict(),
            "model_config": asdict(cfg.model),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(CFG, args)
    distributed, local_rank, world_size, rank, device = init_distributed(args)
    set_seed(cfg.train.seed + rank)

    if is_main_process(distributed, rank):
        print("Building dataloaders...")
        print(f"Using device: {device}")
        if distributed:
            print(f"DDP enabled: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.data,
        seed=cfg.train.seed,
        split_by_subject=args.split_by_subject,
        val_ratio=0.1,
        test_ratio=0.1,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process(distributed, rank):
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        if test_loader:
            print(f"Test batches: {len(test_loader)}")

    raw_model = PPGGSRCLIP(cfg.model).to(device)
    if distributed:
        model: PPGGSRCLIP | DDP = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = raw_model

    if is_main_process(distributed, rank):
        print(f"Model initialized with {sum(p.numel() for p in raw_model.parameters())} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    best_val_loss = float("inf")
    start_time = time.time()
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None

    for epoch in range(1, cfg.train.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = None
        test_metrics = None

        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, device)

        if test_loader is not None:
            test_metrics = run_epoch(model, test_loader, device)

        if is_main_process(distributed, rank):
            message = (
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_ppg2gsr={train_metrics['ppg_to_gsr_acc']:.4f} "
                f"train_gsr2ppg={train_metrics['gsr_to_ppg_acc']:.4f} "
                f"train_ppg_mean_rank={train_metrics['ppg_mean_rank']:.2f} "
                f"train_ppg_median_rank={train_metrics['ppg_median_rank']:.2f} "
                f"train_gsr_mean_rank={train_metrics['gsr_mean_rank']:.2f} "
                f"train_gsr_median_rank={train_metrics['gsr_median_rank']:.2f}"
            )
            if val_metrics is not None:
                message += (
                    f" val_loss={val_metrics['loss']:.4f} "
                    f"val_ppg2gsr={val_metrics['ppg_to_gsr_acc']:.4f} "
                    f"val_gsr2ppg={val_metrics['gsr_to_ppg_acc']:.4f} "
                    f"val_ppg_mean_rank={val_metrics['ppg_mean_rank']:.2f} "
                    f"val_ppg_median_rank={val_metrics['ppg_median_rank']:.2f} "
                    f"val_gsr_mean_rank={val_metrics['gsr_mean_rank']:.2f} "
                    f"val_gsr_median_rank={val_metrics['gsr_median_rank']:.2f}"
                )
            if test_metrics is not None:
                message += (
                    f" test_loss={test_metrics['loss']:.4f} "
                    f"test_ppg2gsr={test_metrics['ppg_to_gsr_acc']:.4f} "
                    f"test_gsr2ppg={test_metrics['gsr_to_ppg_acc']:.4f} "
                    f"test_ppg_mean_rank={test_metrics['ppg_mean_rank']:.2f} "
                    f"test_ppg_median_rank={test_metrics['ppg_median_rank']:.2f} "
                    f"test_gsr_mean_rank={test_metrics['gsr_mean_rank']:.2f} "
                    f"test_gsr_median_rank={test_metrics['gsr_median_rank']:.2f}"
                )
            message += f" [time={time.time() - epoch_start_time:.2f}s]"
            print(message)

            model_to_save = unwrap_model(model)
            if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(cfg.train.checkpoint_dir / "best.pt", model_to_save, optimizer, epoch, cfg, val_metrics)
                save_encoder_checkpoint(cfg.train.checkpoint_dir / "best_encoder.pt", model_to_save, epoch, cfg, val_metrics)

            if epoch % cfg.train.save_every == 0:
                save_checkpoint(
                    cfg.train.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    model_to_save,
                    optimizer,
                    epoch,
                    cfg,
                    test_metrics or val_metrics or train_metrics,
                )

    if is_main_process(distributed, rank):
        save_checkpoint(
            cfg.train.checkpoint_dir / "last.pt",
            unwrap_model(model),
            optimizer,
            cfg.train.epochs,
            cfg,
            test_metrics or val_metrics or train_metrics,
        )
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"\nTraining completed in {hours}h {minutes}m {seconds:.2f}s")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
