#!/usr/bin/env python
"""
Aggregate and visualize msprof op_summary data.

Usage:
    python scripts/visualize_profiler.py \
        --input export_only_prof_dir/orangepiaipro_18323_20251121182720764_ascend_pt/PROF_000001_20251121182720782_DLGNOIPHQIDIELOB/mindstudio_profiler_output \
        --top 20

Outputs (inside the input directory):
    - aggregated_op_summary.csv: grouped by (Op Type, Op Name), sorted by total time.
    - aggregated_op_summary_topN.png: bar chart of the top-N ops (if matplotlib is available).

The script is defensive about column names because msprof CSV headers vary slightly across versions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def find_time_columns(cols: Iterable[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Identify total time and call count columns (case/space insensitive).
    Returns (time_col, count_col).
    """
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in cols}
    time_col = None
    for key in (
        "totaltime(us)",
        "selftime(us)",
        "totaltime",
        "selftime",
        "taskduration(us)",  # msprof task timeline dump
    ):
        if key in normalized:
            time_col = normalized[key]
            break
    count_col = None
    for key in ("callcount", "opcount", "count", "runcount"):
        if key in normalized:
            count_col = normalized[key]
            break
    return time_col, count_col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces and lower-case column names for easier lookup."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def aggregate_op_summary(csv_paths: list[Path]) -> pd.DataFrame:
    if not csv_paths:
        raise FileNotFoundError("No op_summary*.csv files found under input directory.")
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df = normalize_columns(df)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    time_col, count_col = find_time_columns(df_all.columns)
    if time_col is None:
        raise ValueError(f"Could not find time column in {df_all.columns.tolist()}")

    # Prefer to keep Op Type/Name if present.
    op_type_col = next((c for c in df_all.columns if c.lower().startswith("op type")), None)
    op_name_col = next((c for c in df_all.columns if c.lower().startswith("op name")), None)
    if op_name_col is None:
        raise ValueError("Could not find 'Op Name' column in op_summary.")

    group_cols = [c for c in (op_type_col, op_name_col) if c is not None]

    agg = df_all.groupby(group_cols, dropna=False).agg(
        total_time_us=(time_col, "sum"),
        calls=(count_col, "sum") if count_col else (time_col, "count"),
        max_time_us=(time_col, "max"),
    )
    agg["avg_time_us"] = agg["total_time_us"] / agg["calls"].clip(lower=1)
    agg = agg.sort_values("total_time_us", ascending=False).reset_index()
    return agg


def plot_top_ops(df: pd.DataFrame, out_path: Path, top: int, metric: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[WARN] matplotlib not available; skipping chart export.")
        return

    top_df = df.head(top)
    labels = [
        f"{row.get('Op Type', '')}:{row.get('Op Name', '')}" if "Op Type" in row else row["Op Name"]
        for _, row in top_df.iterrows()
    ]
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_df)), top_df[metric], color="#4C72B0")
    plt.gca().invert_yaxis()
    plt.yticks(range(len(top_df)), labels, fontsize=8)
    plt.xlabel(metric.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved chart -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate msprof op_summary CSV files.")
    parser.add_argument("--input", required=True, type=Path, help="Path to mindstudio_profiler_output directory.")
    parser.add_argument("--top", type=int, default=20, help="Number of top ops to plot.")
    args = parser.parse_args()

    input_dir = args.input
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_paths = list(input_dir.glob("op_summary*.csv"))
    agg = aggregate_op_summary(csv_paths)

    # Sort by total time and avg time separately for easier comparison.
    agg_total = agg.sort_values("total_time_us", ascending=False).reset_index(drop=True)
    agg_avg = agg.sort_values("avg_time_us", ascending=False).reset_index(drop=True)

    out_csv_total = input_dir / "aggregated_op_summary.csv"
    agg_total.to_csv(out_csv_total, index=False)
    print(f"[INFO] Saved aggregated (by total) CSV -> {out_csv_total}")
    print("[INFO] Top 10 by total:\n", agg_total.head(10).to_string(index=False))

    out_csv_avg = input_dir / "aggregated_op_summary_by_avg.csv"
    agg_avg.to_csv(out_csv_avg, index=False)
    print(f"[INFO] Saved aggregated (by avg) CSV -> {out_csv_avg}")
    print("[INFO] Top 10 by avg:\n", agg_avg.head(10).to_string(index=False))

    plot_top_ops(
        agg_total,
        input_dir / f"aggregated_op_summary_top{args.top}_total.png",
        args.top,
        metric="total_time_us",
        title=f"Top {args.top} Ops by Total Time",
    )
    plot_top_ops(
        agg_avg,
        input_dir / f"aggregated_op_summary_top{args.top}_avg.png",
        args.top,
        metric="avg_time_us",
        title=f"Top {args.top} Ops by Avg Time",
    )


if __name__ == "__main__":
    main()
