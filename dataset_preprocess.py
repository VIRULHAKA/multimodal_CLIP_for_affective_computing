#!/usr/bin/env python3
"""
Preprocess GSR and PPG data from .npy files: filter, clean, and prepare for model training.

Input format:
  data_dir/
    id_session_GSR.npy   (1D array)
    id_session_PPG.npy   (1D array)
    ...
  - GSR sampling rate: 40 Hz
  - PPG sampling rate: 20 Hz
  - Same start time for matched pairs (guaranteed by filename matching)

Features:
- Remove leading bad data from GSR (values <= 0.006) and corresponding PPG part
- Apply Butterworth bandpass filter to PPG
- Flexible output options (in-memory or save to .npy files)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class FilterConfig:
    """Configuration for Butterworth filter."""
    freq_range: Tuple[float, float] = (0.5, 0.8)  # Hz
    order: int = 4
    sample_rate: float = 20.0  # PPG sample rate


def _get_session_id_from_filename(filename: str) -> str:
    """Extract session_id from filename (remove _GSR or _PPG suffix)."""
    stem = Path(filename).stem
    for suffix in ("_GSR", "_PPG"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_npy_data(path: Path) -> np.ndarray:
    """Load 1D array from .npy file."""
    data = np.load(path)
    if data.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {data.shape}")
    return data


def _match_gsr_ppg_pairs(data_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Find matching GSR and PPG files in data_dir.
    
    Returns:
        Dict mapping session_id to (gsr_path, ppg_path)
    """
    gsr_files = sorted(data_dir.glob("*_GSR.npy"))
    ppg_files = sorted(data_dir.glob("*_PPG.npy"))
    
    ppg_map = {_get_session_id_from_filename(p.name): p for p in ppg_files}
    
    pairs = {}
    for gsr_path in gsr_files:
        session_id = _get_session_id_from_filename(gsr_path.name)
        ppg_path = ppg_map.get(session_id)
        if ppg_path is not None:
            pairs[session_id] = (gsr_path, ppg_path)
    
    return pairs


def _apply_butterworth_filter(
    ppg_data: np.ndarray,
    sample_rate: float,
    freq_range: Tuple[float, float],
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to PPG data.
    
    Args:
        ppg_data: PPG signal values
        sample_rate: Sampling rate in Hz
        freq_range: (low_freq, high_freq) in Hz
        order: Filter order
        
    Returns:
        Filtered PPG data
    """
    nyquist = sample_rate / 2
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    
    # Validate frequency range
    if low <= 0 or high >= 1:
        raise ValueError(
            f"Invalid frequency range: {freq_range} Hz with sample rate {sample_rate} Hz. "
            f"Normalized range [{low}, {high}] must be in (0, 1)."
        )
    
    # Design filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter (forward-backward to preserve phase)
    filtered = filtfilt(b, a, ppg_data)
    
    return filtered


def preprocess_pair(
    gsr_data: np.ndarray,
    ppg_data: np.ndarray,
    filter_config: FilterConfig,
    gsr_sample_rate: float = 40.0,
    ppg_sample_rate: float = 20.0,
    remove_leading_zeros: bool = True,
    apply_ppg_filter: bool = True,
    gsr_threshold: float = 0.006,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Preprocess a single GSR-PPG pair.
    
    Args:
        gsr_data: GSR signal (1D array)
        ppg_data: PPG signal (1D array)
        filter_config: Filter configuration
        gsr_sample_rate: GSR sampling rate (Hz), default 40
        ppg_sample_rate: PPG sampling rate (Hz), default 20
        remove_leading_zeros: Whether to remove leading GSR zeros and corresponding PPG data
        apply_ppg_filter: Whether to apply Butterworth filter
        gsr_threshold: GSR values <= threshold are considered bad
        
    Returns:
        (cleaned_gsr, cleaned_ppg, metadata_dict)
    """
    metadata = {
        "gsr_original_length": len(gsr_data),
        "ppg_original_length": len(ppg_data),
        "samples_removed": 0,
        "gsr_final_length": 0,
        "ppg_final_length": 0,
    }
    
    gsr_clean = gsr_data.copy()
    ppg_clean = ppg_data.copy()
    
    # Step 1: Remove leading bad GSR data and corresponding PPG data
    if remove_leading_zeros:
        # Find first valid GSR sample
        valid_gsr_idx = np.where(gsr_clean > gsr_threshold)[0]
        
        if len(valid_gsr_idx) == 0:
            print("Warning: All GSR values <= threshold")
            return np.array([]), np.array([]), {**metadata, "gsr_final_length": 0, "ppg_final_length": 0}
        
        first_valid_gsr_idx = valid_gsr_idx[0]
        
        if first_valid_gsr_idx > 0:
            # Calculate corresponding PPG index based on sample rates
            time_of_first_valid = first_valid_gsr_idx / gsr_sample_rate
            first_valid_ppg_idx = int(np.round(time_of_first_valid * ppg_sample_rate))
            first_valid_ppg_idx = min(first_valid_ppg_idx, len(ppg_clean) - 1)
            
            gsr_clean = gsr_clean[first_valid_gsr_idx:]
            ppg_clean = ppg_clean[first_valid_ppg_idx:]
            metadata["samples_removed"] = first_valid_gsr_idx
    
    if len(gsr_clean) == 0 or len(ppg_clean) == 0:
        print("Warning: Arrays are empty after cleanup")
        return gsr_clean, ppg_clean, {**metadata, "gsr_final_length": len(gsr_clean), "ppg_final_length": len(ppg_clean)}
    
    # Step 2: Apply Butterworth filter to PPG
    if apply_ppg_filter:
        try:
            ppg_clean = _apply_butterworth_filter(
                ppg_clean,
                sample_rate=filter_config.sample_rate,
                freq_range=filter_config.freq_range,
                order=filter_config.order
            )
        except ValueError as e:
            print(f"Warning: Filter failed: {e}. Skipping PPG filter.")
    
    metadata["gsr_final_length"] = len(gsr_clean)
    metadata["ppg_final_length"] = len(ppg_clean)
    
    return gsr_clean, ppg_clean, metadata


def preprocess_dataset(
    data_dir: Path,
    filter_config: FilterConfig,
    gsr_sample_rate: float = 40.0,
    ppg_sample_rate: float = 20.0,
    output_mode: str = "none",  # "none", "save"
    output_dir: Optional[Path] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Preprocess all GSR-PPG pairs in data_dir.
    
    Args:
        data_dir: Path to directory with *_GSR.npy and *_PPG.npy files
        filter_config: Filter configuration
        gsr_sample_rate: GSR sampling rate (Hz)
        ppg_sample_rate: PPG sampling rate (Hz)
        output_mode: "none" (in-memory only) or "save" (save to files)
        output_dir: Output directory (required if output_mode == "save")
        
    Returns:
        Dictionary mapping session_id to (processed_gsr, processed_ppg, metadata)
    """
    if output_mode == "save" and output_dir is None:
        raise ValueError("output_dir required when output_mode == 'save'")
    
    if output_mode == "save":
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching pairs
    pairs = _match_gsr_ppg_pairs(data_dir)
    
    if not pairs:
        print("Warning: No matching GSR/PPG pairs found!")
        return {}
    
    results = {}
    
    for session_id, (gsr_path, ppg_path) in pairs.items():
        print(f"Processing {session_id}...", end=" ")
        
        try:
            # Load .npy files
            gsr_data = _load_npy_data(gsr_path)
            ppg_data = _load_npy_data(ppg_path)
            
            # Preprocess
            gsr_processed, ppg_processed, metadata = preprocess_pair(
                gsr_data,
                ppg_data,
                filter_config=filter_config,
                gsr_sample_rate=gsr_sample_rate,
                ppg_sample_rate=ppg_sample_rate,
            )
            
            print(f"✓ (GSR: {metadata['gsr_original_length']}→{metadata['gsr_final_length']}, "
                  f"PPG: {metadata['ppg_original_length']}→{metadata['ppg_final_length']})")
            
            # Output handling
            if output_mode == "save":
                gsr_out_path = output_dir / f"{session_id}_GSR.npy"
                ppg_out_path = output_dir / f"{session_id}_PPG.npy"
                np.save(gsr_out_path, gsr_processed)
                np.save(ppg_out_path, ppg_processed)
            
            results[session_id] = (gsr_processed, ppg_processed, metadata)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GSR/PPG .npy data")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory with *_GSR.npy and *_PPG.npy files (~300 pairs)"
    )
    parser.add_argument(
        "--output_mode",
        choices=["none", "save"],
        default="none",
        help="Output mode: 'none' (in-memory only), 'save' (to .npy files)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (required if output_mode == 'save')"
    )
    parser.add_argument(
        "--freq_low",
        type=float,
        default=0.5,
        help="Butterworth filter lower frequency (Hz)"
    )
    parser.add_argument(
        "--freq_high",
        type=float,
        default=0.8,
        help="Butterworth filter upper frequency (Hz)"
    )
    parser.add_argument(
        "--filter_order",
        type=int,
        default=4,
        help="Butterworth filter order"
    )
    parser.add_argument(
        "--gsr_sample_rate",
        type=float,
        default=40.0,
        help="GSR sampling rate (Hz)"
    )
    parser.add_argument(
        "--ppg_sample_rate",
        type=float,
        default=20.0,
        help="PPG sampling rate (Hz)"
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Skip leading zero removal"
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip Butterworth filter on PPG"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    filter_config = FilterConfig(
        freq_range=(args.freq_low, args.freq_high),
        order=args.filter_order
    )
    
    print(f"Starting preprocessing...")
    print(f"  Input: {args.data_dir}")
    print(f"  GSR sample rate: {args.gsr_sample_rate} Hz")
    print(f"  PPG sample rate: {args.ppg_sample_rate} Hz")
    print(f"  Filter config: freq={filter_config.freq_range} Hz, order={filter_config.order}")
    print(f"  Output mode: {args.output_mode}")
    if args.output_mode == "save":
        print(f"  Output dir: {args.output_dir}")
    print()
    
    results = preprocess_dataset(
        data_dir=args.data_dir,
        filter_config=filter_config,
        gsr_sample_rate=args.gsr_sample_rate,
        ppg_sample_rate=args.ppg_sample_rate,
        output_mode=args.output_mode,
        output_dir=args.output_dir,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    if results:
        total_removed = sum(m["samples_removed"] for _, _, m in results.values())
        total_gsr_len = sum(m["gsr_final_length"] for _, _, m in results.values())
        total_ppg_len = sum(m["ppg_final_length"] for _, _, m in results.values())
        print(f"Total pairs processed: {len(results)}")
        print(f"Total leading bad samples removed: {total_removed}")
        print(f"Total GSR samples (after processing): {total_gsr_len}")
        print(f"Total PPG samples (after processing): {total_ppg_len}")
    else:
        print("No pairs processed.")
    print("="*60)


if __name__ == "__main__":
    main()
