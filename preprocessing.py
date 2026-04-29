from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
NIFTY500_PATH = PROJECT_ROOT / "nifty500.csv"
NIFTY50_PATH = PROJECT_ROOT / "nifty50.csv"


def load_nifty500(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("nifty500.csv should be a flat close-price matrix.")
    frame.index.name = "Date"
    return frame


def load_nifty50(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("nifty50.csv should be a flat close-price matrix.")
    if "NIFTY50" not in frame.columns and len(frame.columns) == 1:
        frame = frame.rename(columns={frame.columns[0]: "NIFTY50"})
    frame.index.name = "Date"
    return frame


def clean_nifty500(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")

    return cleaned.ffill()


def clean_nifty50(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.ffill()
    return cleaned


def main() -> int:
    try:
        nifty500_raw = load_nifty500(NIFTY500_PATH)
        nifty50_raw = load_nifty50(NIFTY50_PATH)
    except Exception as exc:
        print(f"[ERROR] Failed to load input files: {exc}")
        return 1

    n500_rows_before = len(nifty500_raw)
    n50_rows_before = len(nifty50_raw)
    n500_nulls_before = int(nifty500_raw.isna().sum().sum())
    n50_nulls_before = int(nifty50_raw.isna().sum().sum())

    nifty500 = clean_nifty500(nifty500_raw)
    nifty50 = clean_nifty50(nifty50_raw)

    valid_n50 = nifty50.dropna(how="all")
    if valid_n50.empty:
        print("[ERROR] nifty50.csv has no valid benchmark rows after cleaning.")
        return 2

    nifty50_start_date = valid_n50.index.min()
    cutoff_date = nifty50_start_date - pd.Timedelta(days=1000)
    nifty500 = nifty500[nifty500.index >= cutoff_date]

    n500_rows_after = len(nifty500)
    n50_rows_after = len(nifty50)
    n500_nulls_after = int(nifty500.isna().sum().sum())
    n50_nulls_after = int(nifty50.isna().sum().sum())

    nifty500.to_csv(NIFTY500_PATH)
    nifty50.to_csv(NIFTY50_PATH)

    print("Preprocessing completed.")
    print(f"NIFTY50 first valid date: {nifty50_start_date.date()}")
    print(f"NIFTY500 cutoff date: {cutoff_date.date()}")
    print(f"NIFTY500 rows: {n500_rows_before} -> {n500_rows_after}")
    print(f"NIFTY50 rows: {n50_rows_before} -> {n50_rows_after}")
    print(f"NIFTY500 nulls: {n500_nulls_before} -> {n500_nulls_after}")
    print(f"NIFTY50 nulls: {n50_nulls_before} -> {n50_nulls_after}")
    print(f"Overwritten: {NIFTY500_PATH}")
    print(f"Overwritten: {NIFTY50_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
