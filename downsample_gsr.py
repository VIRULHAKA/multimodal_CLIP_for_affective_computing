#!/usr/bin/env python3
"""Downsample GSR signals from 40Hz to 20Hz.

Processes all *_gsr.npy files in a directory tree and saves downsampled versions.
"""

import argparse
from pathlib import Path

import numpy as np


def downsample_signal(data: np.ndarray, downsample_factor: int = 2) -> np.ndarray:
    """Downsample signal by taking every nth sample (simple decimation).
    
    Args:
        data: Input signal array
        downsample_factor: Downsample factor (2 for 40Hz -> 20Hz)
    
    Returns:
        Downsampled signal
    """
    return data[::downsample_factor]


def downsample_gsr_files(data_dir: Path, downsample_factor: int = 2, suffix: str = "_20hz") -> None:
    """Find all GSR npy files and downsample them, saving to same directory with modified name.
    
    Args:
        data_dir: Root directory to search for *_gsr.npy files
        downsample_factor: Downsample factor (2 for 40Hz -> 20Hz)
        suffix: Suffix to add to downsampled filename (default: "_20hz")
    """
    gsr_files = sorted(data_dir.rglob("*_GSR.npy"))
    print(f"Found {len(gsr_files)} GSR files")
    
    if not gsr_files:
        print(f"No *_GSR.npy files found in {data_dir}")
        return
    
    for gsr_path in gsr_files:
        # Load signal (may be 2D with [timestamp, value] or 1D)
        subject_id = int(gsr_path.stem[:4])
        if subject_id < 3000: 
            print(f"Skipping {gsr_path} (subject ID {subject_id} < 3000)")
            continue
        data = np.load(gsr_path, mmap_mode="r")
        
        if data.ndim == 2:
            # [timestamp, signal] format
            timestamps = data[:, 0]
            signal_values = data[:, 1]
            
            # Downsample both
            ts_down = downsample_signal(timestamps, downsample_factor)
            sig_down = downsample_signal(signal_values, downsample_factor)
            
            # Recombine
            result = np.column_stack([ts_down, sig_down]).astype(np.float32)
        else:
            # 1D signal
            result = downsample_signal(data, downsample_factor).astype(np.float32)
        
        # Save in same directory with modified filename
        out_path = gsr_path.parent / f"{gsr_path.stem}{suffix}.npy"
        
        np.save(out_path, result)
        print(f"Saved {out_path}: {data.shape} -> {result.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample GSR signals from 40Hz to 20Hz")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root directory with *_GSR.npy files")
    parser.add_argument("--factor", type=int, default=2, help="Downsample factor (2 for 40Hz->20Hz)")
    parser.add_argument("--suffix", type=str, default="_20hz", help="Suffix for downsampled files")
    args = parser.parse_args()
    
    downsample_gsr_files(args.data_dir, args.factor, args.suffix)
    print("Done!")


if __name__ == "__main__":
    main()
