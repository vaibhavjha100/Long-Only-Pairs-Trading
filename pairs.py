"""Daily NIFTY500 pair discovery: correlation screen then Engle-Granger with freeze periods."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import _to_close_matrix, find_cointegrated_pairs, find_correlated_pairs

CORR_WINDOW = 250
CORR_THRESHOLD = 0.8
COINT_WINDOW = 500
COINT_ALPHA = 0.05
FREEZE_TRADING_DAYS = 30
TRIM_INITIAL_ROWS = 500


def load_nifty500(path: Path | None = None) -> pd.DataFrame:
    p = path or PROJECT_ROOT / "nifty500.csv"
    frame = pd.read_csv(p, index_col=0, parse_dates=[0])
    frame.index.name = "Date"
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return _to_close_matrix(frame)


def first_valid_dates(matrix: pd.DataFrame) -> pd.Series:
    """Per column, first index where value is non-null."""
    first: dict[str, pd.Timestamp] = {}
    for col in matrix.columns:
        s = matrix[col]
        idx = s.first_valid_index()
        if idx is not None:
            first[col] = pd.Timestamp(idx)
    return pd.Series(first, dtype="datetime64[ns]")


def pairs_to_serializable(pairs: list[tuple[str, str, float]]) -> list[list]:
    return [[a, b, float(p)] for a, b, p in pairs]


def run_pair_discovery(
    matrix: pd.DataFrame,
    *,
    corr_window: int = CORR_WINDOW,
    corr_threshold: float = CORR_THRESHOLD,
    coint_window: int = COINT_WINDOW,
    coint_alpha: float = COINT_ALPHA,
    freeze_days: int = FREEZE_TRADING_DAYS,
) -> pd.DataFrame:
    """
    Walk each trading day t with at least coint_window history rows.

    When not in a freeze, compute correlation (corr_window) then Engle-Granger
    (coint_window). After a non-empty cointegration list, copy that list for the
    next freeze_days trading sessions without re-running discovery.

    Returns a DataFrame indexed by Date with column cointegrated_pairs (list of
    (a, b, pvalue) tuples). Trim initial rows in main() before saving CSV.
    """
    matrix = matrix.sort_index()
    matrix = matrix[~matrix.index.duplicated(keep="last")]
    first_seen = first_valid_dates(matrix)

    rows: list[dict] = []
    freeze_remaining = 0
    frozen_pairs: list[tuple[str, str, float]] = []
    coint_found = False

    dates = matrix.index
    for i in range(coint_window - 1, len(dates)):
        t = dates[i]

        if freeze_remaining > 0:
            rows.append(
                {
                    "date": t,
                    "cointegrated_pairs": list(frozen_pairs),
                }
            )
            freeze_remaining -= 1
            continue

        valid_cols: list[str] = []
        for col in matrix.columns:
            if col not in first_seen.index:
                continue
            if first_seen[col] > t:
                continue
            if pd.isna(matrix.loc[t, col]):
                continue
            valid_cols.append(col)
        valid_cols.sort()

        if len(valid_cols) < 2:
            pairs_out: list[tuple[str, str, float]] = []
        else:
            sub = matrix[valid_cols].loc[:t]
            corr_candidates = find_correlated_pairs(
                sub, window=corr_window, threshold=corr_threshold
            )
            df_upto_t = matrix.loc[:t]
            pairs_out = find_cointegrated_pairs(
                df_upto_t,
                corr_candidates,
                window=coint_window,
                alpha=coint_alpha,
            )

        rows.append({"date": t, "cointegrated_pairs": pairs_out})

        if pairs_out:
            frozen_pairs = list(pairs_out)
            freeze_remaining = freeze_days
            coint_found = True
        else:
            coint_found = False

    out = pd.DataFrame(rows).set_index("date")
    out.index.name = "Date"
    out.attrs["coint_found"] = coint_found
    return out


def pairs_dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    export = df.copy()
    export["cointegrated_pairs"] = export["cointegrated_pairs"].apply(
        lambda xs: json.dumps(pairs_to_serializable(list(xs)), default=str)
    )
    export.to_csv(path, encoding="utf-8")


def main() -> int:
    matrix = load_nifty500()
    full = run_pair_discovery(matrix)
    trimmed = full.iloc[TRIM_INITIAL_ROWS:].copy()
    out_path = PROJECT_ROOT / "pairs.csv"
    pairs_dataframe_to_csv(trimmed, out_path)
    print(f"Wrote {out_path} with {len(trimmed)} rows (trimmed first {TRIM_INITIAL_ROWS} rows).")
    print(trimmed.head(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
