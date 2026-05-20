"""Dataset utilities for affective computing using pretrained PPG/GSR encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class AffectiveEmbeddingDataset(Dataset):
    """Build fused embeddings + valence/arousal labels from paired PPG/GSR files."""

    def __init__(
        self,
        data_dir: str | Path,
        clip_model: torch.nn.Module,
        device: torch.device,
        fusion: str = "concat",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.clip_model = clip_model
        self.device = device
        self.fusion = fusion

        self.file_pairs = self._discover_files(self.data_dir)
        self.features: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self._build()

    @staticmethod
    def _discover_files(data_dir: Path) -> List[Tuple[Path, Path, Path]]:
        gsr_files = sorted(data_dir.rglob("*_GSR.npy"))
        pairs: List[Tuple[Path, Path, Path]] = []
        for gsr_path in gsr_files:
            stem = gsr_path.stem.replace("_GSR", "")
            ppg_path = gsr_path.with_name(f"{stem}_PPG.npy")
            label_path = gsr_path.with_name(f"{stem}_labels.npy")
            if ppg_path.exists() and label_path.exists():
                pairs.append((gsr_path, ppg_path, label_path))
        if not pairs:
            raise FileNotFoundError(f"No *_GSR/*_PPG/*_labels triplets found under {data_dir}")
        return pairs

    @staticmethod
    def _load_signal(path: Path) -> np.ndarray:
        arr = np.load(path, allow_pickle=True)
        if arr.ndim > 1:
            arr = arr[:, 1] if arr.shape[1] > 1 else arr.ravel()
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            raise ValueError(f"Empty signal after NaN filtering: {path}")
        return arr

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / (std + 1e-8)

    def _fuse(self, ppg_emb: torch.Tensor, gsr_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return torch.cat([ppg_emb, gsr_emb], dim=-1)
        if self.fusion == "sum":
            return ppg_emb + gsr_emb
        if self.fusion == "mean":
            return (ppg_emb + gsr_emb) / 2
        raise ValueError(f"Unsupported fusion: {self.fusion}")

    def _build(self) -> None:
        self.clip_model.eval()
        with torch.no_grad():
            for gsr_path, ppg_path, label_path in self.file_pairs:
                gsr = self._zscore(self._load_signal(gsr_path))
                ppg = self._zscore(self._load_signal(ppg_path))
                n = min(len(gsr), len(ppg))
                gsr_t = torch.from_numpy(gsr[:n]).unsqueeze(0).unsqueeze(-1).to(self.device)
                ppg_t = torch.from_numpy(ppg[:n]).unsqueeze(0).unsqueeze(-1).to(self.device)

                ppg_emb = self.clip_model.encode_ppg(ppg_t).squeeze(0).cpu()
                gsr_emb = self.clip_model.encode_gsr(gsr_t).squeeze(0).cpu()
                feat = self._fuse(ppg_emb, gsr_emb)

                label_arr = np.load(label_path, allow_pickle=True)
                if isinstance(label_arr, np.ndarray) and label_arr.ndim == 1 and label_arr.shape[0] >= 2:
                    valence, arousal = float(label_arr[0]), float(label_arr[1])
                else:
                    raise ValueError(f"Label file must be [valence, arousal]: {label_path}")

                self.features.append(feat.float())
                self.labels.append(torch.tensor([valence, arousal], dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"feature": self.features[idx], "label": self.labels[idx]}


def build_affective_dataloaders(
    dataset: AffectiveEmbeddingDataset,
    batch_size: int,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = len(dataset)
    val_n = max(1, int(n * val_ratio))
    test_n = max(1, int(n * test_ratio))
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("Not enough data for train split")
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n], generator=gen)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
