#!/usr/bin/env python3
"""
Preprocess GSR and PPG data from .npy files.

Input format:
  data_dir/
    id_session_GSR.npy   (1D array, 20 Hz)
    id_session_PPG.npy   (1D array, 20 Hz)
    ...
  - GSR and PPG share the same sampling rate (20 Hz) and start time.

Steps:
  1. Remove leading bad GSR data (values <= GSR_THRESHOLD) and the
     corresponding PPG samples (same index, same sample rate).
  2. Apply Butterworth bandpass filter (0.5–8 Hz) to PPG.
  3. [Reserved] Motion artifact removal via ACC — not yet implemented.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GSR_THRESHOLD: float = 0.007
SAMPLE_RATE: float = 20.0  # Hz — both GSR and PPG


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """Configuration for PPG Butterworth bandpass filter."""
    freq_range: Tuple[float, float] = (0.5, 8.0)  # Hz
    order: int = 4
    sample_rate: float = SAMPLE_RATE


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _session_id(filename: str) -> str:
    """Strip _GSR_20hz / _PPG suffix to get session id."""
    stem = Path(filename).stem
    for suffix in ("_GSR_20hz", "_PPG"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_1d(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 1:
        if data.ndim == 2 and data.shape[1] > 1:
            data = data[:, 1]  # take second column if multiple channels
        else:
            raise ValueError(f"Expected 1D array, got shape {data.shape} in {path}")
    return data.astype(np.float32)


def _match_pairs(data_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Return {session_id: (gsr_path, ppg_path)} for all matched pairs."""
    ppg_map = {_session_id(p.name): p for p in sorted(data_dir.rglob("*_PPG.npy"))}
    pairs = {}
    for gsr_path in sorted(data_dir.rglob("*_GSR_20hz.npy")):
        sid = _session_id(gsr_path.name)
        if sid in ppg_map:
            pairs[sid] = (gsr_path, ppg_map[sid])
    return pairs


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def _remove_leading_bad_gsr(
    gsr: np.ndarray,
    ppg: np.ndarray,
    threshold: float = GSR_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Drop samples from the start where GSR <= threshold.
    Because GSR and PPG share the same sample rate, the cut index is identical
    for both signals.

    Returns (gsr_clean, ppg_clean, n_removed).
    """
    valid = np.where(gsr > threshold)[0]
    if len(valid) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), len(gsr)

    idx = valid[0]
    return gsr[idx:], ppg[idx:], idx


def _bandpass_ppg(ppg: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to PPG."""
    nyquist = cfg.sample_rate / 2.0
    low = cfg.freq_range[0] / nyquist
    high = cfg.freq_range[1] / nyquist

    if not (0 < low < high < 1):
        raise ValueError(
            f"Normalised frequency range [{low:.3f}, {high:.3f}] must be in (0, 1). "
            f"Check freq_range={cfg.freq_range} Hz vs sample_rate={cfg.sample_rate} Hz."
        )

    min_len = 3 * (3 * cfg.order + 1)
    if len(ppg) < min_len:
        raise ValueError(f"PPG signal too short for filtfilt ({len(ppg)} < {min_len} samples).")

    b, a = butter(cfg.order, [low, high], btype="band")
    return filtfilt(b, a, ppg).astype(np.float32)


def _remove_motion_artifacts_acc(
    gsr: np.ndarray,
    ppg: np.ndarray,
    acc: Optional[np.ndarray],  # noqa: ARG001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [Reserved] Remove motion artifacts using ACC signal.
    Not yet implemented — returns signals unchanged.
    """
    return gsr, ppg


# ---------------------------------------------------------------------------
# Per-pair and dataset entry points
# ---------------------------------------------------------------------------

def preprocess_pair(
    gsr: np.ndarray,
    ppg: np.ndarray,
    filter_config: FilterConfig,
    acc: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Full preprocessing pipeline for a single GSR/PPG pair.

    Returns (gsr_out, ppg_out, metadata).
    """
    meta = {
        "gsr_original_length": len(gsr),
        "ppg_original_length": len(ppg),
        "samples_removed": 0,
        "gsr_final_length": 0,
        "ppg_final_length": 0,
        "ppg_filtered": False,
    }

    # Step 1: remove leading bad GSR (and aligned PPG prefix)
    gsr, ppg, n_removed = _remove_leading_bad_gsr(gsr, ppg)
    meta["samples_removed"] = n_removed

    if len(gsr) == 0 or len(ppg) == 0:
        meta["gsr_final_length"] = len(gsr)
        meta["ppg_final_length"] = len(ppg)
        return gsr, ppg, meta

    # Step 2: bandpass filter PPG
    try:
        ppg = _bandpass_ppg(ppg, filter_config)
        meta["ppg_filtered"] = True
    except ValueError as e:
        print(f"  Warning: PPG filter skipped — {e}")

    # Step 3: ACC motion artifact removal (reserved)
    gsr, ppg = _remove_motion_artifacts_acc(gsr, ppg, acc)

    meta["gsr_final_length"] = len(gsr)
    meta["ppg_final_length"] = len(ppg)
    return gsr, ppg, meta


def preprocess_dataset(
    data_dir: Path,
    filter_config: FilterConfig,
    output_dir: Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Preprocess all matched GSR/PPG pairs under data_dir and save to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _match_pairs(data_dir)

    if not pairs:
        print("Warning: no matching GSR/PPG pairs found.")
        return {}

    results = {}
    for sid, (gsr_path, ppg_path) in pairs.items():
        print(f"Processing {sid} ...", end=" ", flush=True)
        try:
            gsr = _load_1d(gsr_path)
            ppg = _load_1d(ppg_path)
            gsr_out, ppg_out, meta = preprocess_pair(gsr, ppg, filter_config)
            np.save(output_dir / f"{sid}_GSR.npy", gsr_out)
            np.save(output_dir / f"{sid}_PPG.npy", ppg_out)
            print(
                f"✓  GSR {meta['gsr_original_length']}→{meta['gsr_final_length']}, "
                f"PPG {meta['ppg_original_length']}→{meta['ppg_final_length']}, "
                f"removed {meta['samples_removed']} leading samples"
            )
            results[sid] = (gsr_out, ppg_out, meta)
        except Exception as e:
            print(f"✗  {e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GSR/PPG .npy data")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Input directory with *_GSR.npy and *_PPG.npy files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to write processed .npy files")
    parser.add_argument("--freq_low", type=float, default=0.5,
                        help="PPG bandpass lower frequency in Hz (default: 0.5)")
    parser.add_argument("--freq_high", type=float, default=8.0,
                        help="PPG bandpass upper frequency in Hz (default: 8.0)")
    parser.add_argument("--filter_order", type=int, default=4,
                        help="Butterworth filter order (default: 4)")
    parser.add_argument("--sample_rate", type=float, default=SAMPLE_RATE,
                        help=f"Shared GSR/PPG sampling rate in Hz (default: {SAMPLE_RATE})")
    # Reserved for future ACC support
    parser.add_argument("--acc_dir", type=Path, default=None,
                        help="[Reserved] Directory with *_ACC.npy files for motion artifact removal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filter_config = FilterConfig(
        freq_range=(args.freq_low, args.freq_high),
        order=args.filter_order,
        sample_rate=args.sample_rate,
    )

    print("Preprocessing configuration")
    print(f"  Input dir   : {args.data_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Sample rate : {args.sample_rate} Hz (GSR + PPG)")
    print(f"  PPG filter  : {filter_config.freq_range[0]}–{filter_config.freq_range[1]} Hz, "
          f"order {filter_config.order}")
    if args.acc_dir:
        print(f"  ACC dir     : {args.acc_dir} (motion removal not yet implemented)")
    print()

    results = preprocess_dataset(args.data_dir, filter_config, args.output_dir)

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    if results:
        total_removed = sum(m["samples_removed"] for _, _, m in results.values())
        print(f"Pairs processed : {len(results)}")
        print(f"Leading samples removed (total) : {total_removed}")
    else:
        print("No pairs processed.")
    print("=" * 55)


if __name__ == "__main__":
    main()