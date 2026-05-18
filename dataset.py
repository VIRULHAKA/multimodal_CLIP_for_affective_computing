"""Dataset utilities for npy files containing timestamp/GSR/PPG columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import DataConfig


@dataclass(frozen=True)
class WindowRecord:
    file_idx: int
    gsr_start: int
    gsr_end: int
    ppg_start: int
    ppg_end: int
    subject_id: str = ""


class PPGGSRNpyDataset(Dataset):
    """Sliding-window Dataset for paired separate GSR and PPG npy files (independent signals).

    Expected file naming: {name}_gsr.npy and {name}_ppg.npy
    Each npy file contains signal values (may be 1D or 2D).

    GSR and PPG are processed independently with separate sliding windows
    (no alignment needed). Each item returns:
        gsr: Tensor[window_size, 1]
        ppg: Tensor[window_size, 1]
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 400,
        stride: int = 200,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.file_pairs = self._discover_files(self.data_dir)
        self.gsr_arrays: List[np.ndarray] = []
        self.ppg_arrays: List[np.ndarray] = []
        self.records: List[WindowRecord] = []

        self._build_index()

    @staticmethod
    def _discover_files(data_dir: Path) -> List[Tuple[Path, Path]]:
        """Discover paired GSR/PPG npy files.
        
        Returns list of (gsr_path, ppg_path) tuples.
        Expects files named like: {name}_gsr.npy and {name}_ppg.npy
        """
        gsr_files = sorted(data_dir.rglob("*_GSR_20hz.npy"))
        ppg_map = {p.stem.replace("_PPG", ""): p for p in sorted(data_dir.rglob("*_PPG.npy"))}
        
        pairs = []
        for gsr_path in gsr_files:
            stem = gsr_path.stem.replace("_GSR_20hz", "")
            ppg_path = data_dir / f"{stem}_PPG.npy"
            if ppg_path.exists():
                pairs.append((gsr_path, ppg_path))
        
        if not pairs:
            raise FileNotFoundError(f"No paired GSR/PPG .npy files found under {data_dir}")
        return pairs

    @staticmethod
    def _load_paired_npy(gsr_path: Path, ppg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load separate GSR and PPG npy files.
        
        Returns:
            Tuple of (gsr_signal, ppg_signal) where each is shape (N,)
        """
        try:
            gsr_raw = np.load(gsr_path, mmap_mode="r")
            ppg_raw = np.load(ppg_path, mmap_mode="r")
        except ValueError:
            # Fallback for pickled files (memmap doesn't support pickle)
            gsr_raw = np.load(gsr_path, allow_pickle=True)
            ppg_raw = np.load(ppg_path, allow_pickle=True)
        
        # If 2D, take only the signal column (assume [timestamp, signal])
        if gsr_raw.ndim > 1:
            gsr_raw = gsr_raw[:, 1] if gsr_raw.shape[1] > 1 else gsr_raw.ravel()
        if ppg_raw.ndim > 1:
            ppg_raw = ppg_raw[:, 1] if ppg_raw.shape[1] > 1 else ppg_raw.ravel()
        
        # Convert to float32 and remove NaN
        gsr_signal = np.asarray(gsr_raw, dtype=np.float32)
        ppg_signal = np.asarray(ppg_raw, dtype=np.float32)
        
        gsr_signal = gsr_signal[~np.isnan(gsr_signal)]
        ppg_signal = ppg_signal[~np.isnan(ppg_signal)]
        
        if len(gsr_signal) == 0 or len(ppg_signal) == 0:
            raise ValueError(f"Empty signals after removing NaN from {gsr_path} and {ppg_path}")
        
        return gsr_signal, ppg_signal

    def _build_index(self) -> None:
        for gsr_path, ppg_path in self.file_pairs:
            # Extract subject_id from filename (e.g., "1001_session1_gsr.npy" -> "1001")
            subject_id = gsr_path.stem.split("_")[0]
            
            gsr_signal, ppg_signal = self._load_paired_npy(gsr_path, ppg_path)
            print(f"Loaded {gsr_path.name}: {len(gsr_signal)} samples, {ppg_path.name}: {len(ppg_signal)} samples (subject={subject_id})")
            
            if len(gsr_signal) < self.window_size or len(ppg_signal) < self.window_size:
                continue

            file_idx = len(self.gsr_arrays)
            self.gsr_arrays.append(gsr_signal)
            self.ppg_arrays.append(ppg_signal)
            # Generate windows for both signals independently
            # Generate paired windows (1-to-1 mapping by index)
            # This assumes GSR and PPG are roughly time-aligned
            gsr_windows = (len(gsr_signal) - self.window_size) // self.stride + 1
            ppg_windows = (len(ppg_signal) - self.window_size) // self.stride + 1
            max_windows = min(gsr_windows, ppg_windows)
            
            for i in range(max_windows):
                gsr_start = i * self.stride
                ppg_start = i * self.stride
                self.records.append(WindowRecord(
                    file_idx=file_idx,
                    gsr_start=gsr_start,
                    gsr_end=gsr_start + self.window_size,
                    ppg_start=ppg_start,
                    ppg_end=ppg_start + self.window_size,
                    subject_id=subject_id,
                ))

        if not self.records:
            raise ValueError(
                "No training windows were created. Try a smaller window_size/stride "
                "or check that the npy files contain enough rows."
            )

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        if std < 1e-8:
            return x - mean
        return (x - mean) / std
    
    def describe(self) -> Dict[str, int | float]:
        """Return a compact summary for checking dataset construction."""
        gsr_rows = [len(arr) for arr in self.gsr_arrays]
        ppg_rows = [len(arr) for arr in self.ppg_arrays]
        return {
            "num_file_pairs": len(self.file_pairs),
            "num_file_pairs_used": len(self.gsr_arrays),
            "num_windows": len(self.records),
            "window_size": self.window_size,
            "stride": self.stride,
            "min_gsr_samples": min(gsr_rows) if gsr_rows else 0,
            "max_gsr_samples": max(gsr_rows) if gsr_rows else 0,
            "min_ppg_samples": min(ppg_rows) if ppg_rows else 0,
            "max_ppg_samples": max(ppg_rows) if ppg_rows else 0,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        
        gsr_window = self.gsr_arrays[rec.file_idx][rec.gsr_start:rec.gsr_end].astype(np.float32)
        ppg_window = self.ppg_arrays[rec.file_idx][rec.ppg_start:rec.ppg_end].astype(np.float32)

        if self.normalize:
            gsr_window = self._zscore(gsr_window)
            ppg_window = self._zscore(ppg_window)

        return {
            "ppg": torch.from_numpy(ppg_window).unsqueeze(-1),
            "gsr": torch.from_numpy(gsr_window).unsqueeze(-1),
        }


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test by sample.
    
    Args:
        dataset: Dataset instance
        val_ratio: Validation ratio (default 0.1 for 10%)
        test_ratio: Test ratio (default 0.1 for 10%)
        seed: Random seed
    
    Returns:
        (train_subset, val_subset, test_subset)
    """
    if not 0.0 <= val_ratio + test_ratio < 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    n = len(dataset)
    val_len = int(n * val_ratio)
    test_len = int(n * test_ratio)
    train_len = n - val_len - test_len
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def split_dataset_by_subject(
    dataset: PPGGSRNpyDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset by subject: each subject goes entirely to train, val, or test.
    
    Args:
        dataset: PPGGSRNpyDataset instance
        val_ratio: Target ratio of validation subjects (default 0.1 for 10%)
        test_ratio: Target ratio of test subjects (default 0.1 for 10%)
        seed: Random seed for reproducibility
    
    Returns:
        (train_subset, val_subset, test_subset)
    """
    if not 0.0 <= val_ratio + test_ratio < 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")
    
    # Get unique subjects from records
    subject_ids = sorted(set(rec.subject_id for rec in dataset.records))
    print(f"Found {len(subject_ids)} unique subjects: {subject_ids}")
    
    # Split subjects (not samples)
    rng = np.random.RandomState(seed)
    n_subjects = len(subject_ids)
    val_subject_count = max(1, int(n_subjects * val_ratio))
    test_subject_count = max(1, int(n_subjects * test_ratio))
    
    # Randomly shuffle and split
    shuffled = rng.permutation(subject_ids)
    test_subjects = set(shuffled[:test_subject_count])
    val_subjects = set(shuffled[test_subject_count:test_subject_count + val_subject_count])
    train_subjects = set(shuffled[test_subject_count + val_subject_count:])
    
    # Create indices for train/val/test based on subject
    train_indices = [i for i, rec in enumerate(dataset.records) if rec.subject_id in train_subjects]
    val_indices = [i for i, rec in enumerate(dataset.records) if rec.subject_id in val_subjects]
    test_indices = [i for i, rec in enumerate(dataset.records) if rec.subject_id in test_subjects]
    
    print(f"Training subjects: {train_subjects} ({len(train_indices)} samples)")
    print(f"Validation subjects: {val_subjects} ({len(val_indices)} samples)")
    print(f"Test subjects: {test_subjects} ({len(test_indices)} samples)")
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return train_set, val_set, test_set


def build_dataloaders(
    cfg: DataConfig, 
    seed: int, 
    split_by_subject: bool = False,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Build train, validation, and test dataloaders.
    
    Args:
        cfg: DataConfig instance
        seed: Random seed
        split_by_subject: If True, split by subject. If False, split by sample.
        val_ratio: Validation ratio (default 0.1 for 10%)
        test_ratio: Test ratio (default 0.1 for 10%)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    print(cfg.data_dir)
    dataset = PPGGSRNpyDataset(
        data_dir=cfg.data_dir,
        window_size=cfg.window_size,
        stride=cfg.stride,
        normalize=cfg.normalize,
    )
    print(dataset.describe())
    
    # Choose split strategy
    if split_by_subject:
        train_set, val_set, test_set = split_dataset_by_subject(dataset, val_ratio, test_ratio, seed)
    else:
        train_set, val_set, test_set = split_dataset(dataset, val_ratio, test_ratio, seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
    )

    val_loader = None
    if len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
    
    test_loader = None
    if len(test_set) > 0:
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
