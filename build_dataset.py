#!/usr/bin/env python3
"""Build paired PPG/GSR samples from raw subject CSV files.

Expected folder layout (root_dir):
  root_dir/
    1001/
      <time_range>_GSR.csv
      <time_range>_PPG.csv
      <time_range>_ACC.csv   (optional, ignored here)
    ...

CSV format assumptions:
  - First column: signal value
  - Second column: timestamp
  - No strict header requirement (script auto-detects)

Output:
    - Need approximately 120 GB of disk space for the aligned dataset. (Or 240 GB? If we also read ACC CSVs.)
    - out_dir/
        aligned/
            1001__<time_range>.npy
            ...
        metadata.csv
        summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


VALID_SUBJECT_PREFIXES = ("10", "20", "30")


@dataclass
class PairRecord:
    subject_id: str
    session_id: str
    start_time: float
    end_time: float
    n_gsr: int
    n_ppg: int


def _read_signal_csv(path: Path, value_name: str) -> pd.DataFrame:
    """Read CSV and normalize to columns: [timestamp, value_name]."""
    df = pd.read_csv(path)

    # Handle headerless files: fallback to raw read with numeric column ids. This might not be necessary and useful. 
    if df.shape[1] < 2:
        df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path} has fewer than 2 columns")

    col0, col1 = df.columns[:2]
    ts_raw = df[col1].astype(str).str.strip()
    ts_dt = pd.to_datetime(
    ts_raw,
    format='mixed',
    errors="coerce"
    )
    print(f"{ts_dt.isna().sum()} timestamps could not be parsed in {path.name}")


    timestamp = ts_dt

    out = pd.DataFrame(
        {
            value_name: pd.to_numeric(df[col0], errors="coerce"),
            "timestamp": timestamp,
        }
    ).dropna(subset=[value_name, "timestamp"])

    out = out.sort_values("timestamp", kind="mergesort")
    out = out[["timestamp", value_name]].reset_index(drop=True)
    print(f"  Read {len(out)} valid rows from {path.name}")
    return out


def _overlap_range(gsr: pd.DataFrame, ppg: pd.DataFrame) -> Optional[Tuple[float, float]]:
    print(gsr["timestamp"].iloc[0], gsr["timestamp"].iloc[-1])
    start = max(gsr["timestamp"].iloc[0], ppg["timestamp"].iloc[0])
    end = min(gsr["timestamp"].iloc[-1], ppg["timestamp"].iloc[-1])

    if end <= start:
        return None
    return start, end


def _crop_to_overlap(gsr: pd.DataFrame, ppg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Tuple[float, float]]]:
    overlap = _overlap_range(gsr, ppg)
    if overlap is None:
        return gsr.iloc[0:0], ppg.iloc[0:0], None

    start, end = overlap
    gsr_c = gsr[(gsr["timestamp"] >= start) & (gsr["timestamp"] <= end)].copy()
    ppg_c = ppg[(ppg["timestamp"] >= start) & (ppg["timestamp"] <= end)].copy()
    return gsr_c.reset_index(drop=True), ppg_c.reset_index(drop=True), overlap


def _align_on_ppg_timestamps(
    gsr_cropped: pd.DataFrame,
    ppg_cropped: pd.DataFrame,
    tolerance_s: float,
) -> pd.DataFrame:
    """Align GSR to PPG timeline using nearest timestamp within tolerance."""
    if gsr_cropped.empty or ppg_cropped.empty:
        return pd.DataFrame(columns=["timestamp", "ppg", "gsr"])

    # aligned = pd.merge_asof(
    #     ppg_cropped.sort_values("timestamp"),
    #     gsr_cropped.sort_values("timestamp"),
    #     on="timestamp",
    #     direction="nearest",
    #     tolerance=tolerance_s,
    # )

    aligned = pd.merge(
        ppg_cropped.sort_values("timestamp"),
        gsr_cropped.sort_values("timestamp"),
        on="timestamp",
    )
    aligned = aligned.dropna(subset=["ppg", "gsr"]).reset_index(drop=True)
    return aligned[["timestamp", "ppg", "gsr"]]


def _iter_subject_dirs(root_dir: Path) -> Iterable[Path]:
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and p.name.isdigit() and p.name.startswith(VALID_SUBJECT_PREFIXES):
            yield p


def _basename_without_modality(path: Path) -> str:
    stem = path.stem 
    for suffix in ("_GSR", "_PPG", "_ACC"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_dataset(root_dir: Path, out_dir: Path, tolerance_s: float) -> pd.DataFrame:
    print("-----------------------------------------")
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    records: List[PairRecord] = []
    print(f"Processing subjects in {root_dir}..." )

    for subj_dir in _iter_subject_dirs(root_dir):
        print("-----------------------------------------")
        subject_id = subj_dir.name
        print(subject_id)
        if int(subject_id) < 3025:
            print(f"Skipping subject {subject_id} (ID < 3025)")
            continue
        gsr_files = sorted(subj_dir.glob("*_GSR.csv")) # 这是一个gsr的文件list
        ppg_files = sorted(subj_dir.glob("*_PPG.csv"))
        ppg_map = {_basename_without_modality(p): p for p in ppg_files}

        for gsr_path in gsr_files:
            print("-----------------------------------------")
            session_id = _basename_without_modality(gsr_path)
            ppg_path = ppg_map.get(session_id)
            if ppg_path is None:
                continue

            gsr = _read_signal_csv(gsr_path, "gsr")
            ppg = _read_signal_csv(ppg_path, "ppg")
            gsr_c, ppg_c, overlap = _crop_to_overlap(gsr, ppg)

            if overlap is None:
                records.append(
                    PairRecord(subject_id, session_id, np.nan, np.nan, 0, 0, 0)
                )
                continue

            gsr_arr = gsr_c.to_numpy(dtype=np.float32)
            ppg_arr = ppg_c.to_numpy(dtype=np.float32)
            gsr_out_path = aligned_dir / f"{subject_id}__{session_id}_GSR.npy"
            ppg_out_path = aligned_dir / f"{subject_id}__{session_id}_PPG.npy"
            np.save(gsr_out_path, gsr_arr)
            np.save(ppg_out_path, ppg_arr)

            print(f"Subject {subject_id}, Session {session_id}: GSR={len(gsr)}, PPG={len(ppg)}, Overlap={len(gsr_c)} GSR & {len(ppg_c)} PPG")


            start, end = overlap
            try:
                formatted_start = pd.to_datetime(start, unit="s").strftime("%Y-%m-%d %H:%M:%S")
                formatted_end = pd.to_datetime(end, unit="s").strftime("%Y-%m-%d %H:%M:%S")
                print(f"  Overlap range: {formatted_start} to {formatted_end}")
            except:
                print(f"  Overlap range: {start:.3f} to {end:.3f} seconds since epoch")
            # aligned = _align_on_ppg_timestamps(gsr_c, ppg_c, tolerance_s=tolerance_s)
            # aligned["timestamp"] = aligned["timestamp"].astype("int64") / 1e9
            # aligned_arr = aligned.to_numpy(dtype=np.float32)

            # out_path = aligned_dir / f"{subject_id}__{session_id}.npy"
            # np.save(out_path, aligned_arr)

            records.append(
                PairRecord(
                    subject_id=subject_id,
                    session_id=session_id,
                    start_time=start,
                    end_time=end,
                    n_gsr=len(gsr_c),
                    n_ppg=len(ppg_c),
                )
            )

    meta = pd.DataFrame([r.__dict__ for r in records])
    meta.to_csv(out_dir / "metadata.csv", index=False)

    summary: Dict[str, float] = {
        "num_pairs": int(len(meta)),
        "num_pairs_with_overlap": int(meta["n_gsr"].gt(0).sum()) if len(meta) else 0,
        "mean_gsr_len": float(meta["n_gsr"].mean()) if len(meta) else 0.0,
        "mean_ppg_len": float(meta["n_ppg"].mean()) if len(meta) else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned PPG/GSR dataset from raw folders")
    parser.add_argument("--root_dir", type=Path, required=True, help="Root folder containing subject folders")
    parser.add_argument("--out_dir", type=Path, default=Path("dataset_output"), help="Output folder")
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=0.03,
        help="Max timestamp difference for nearest-neighbor alignment (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    print("Starting dataset building...")
    args = parse_args()
    meta = build_dataset(args.root_dir, args.out_dir, args.tolerance_s)
    print(f"Done. Built {len(meta)} subject-session pairs.")
    if len(meta):
        print(meta[["subject_id", "session_id", "n_gsr", "n_ppg"]].head())


if __name__ == "__main__":
    main()
