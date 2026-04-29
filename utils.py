from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


Pair = tuple[str, str]


def _validate_input(df: pd.DataFrame) -> None:
    """Ensure input has column labels and is not empty."""
    if df is None or df.empty or len(df.columns) == 0:
        raise ValueError("Expected non-empty DataFrame with ticker columns.")


def _get_price_field_for_ticker(df: pd.DataFrame, ticker: str) -> str:
    """Pick preferred price field for a ticker."""
    fields = df[ticker].columns
    if "Close" in fields:
        return "Close"
    if "Adj Close" in fields:
        return "Adj Close"
    raise ValueError(f"Ticker {ticker} has neither 'Close' nor 'Adj Close'.")


def _get_ticker_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract one ticker close series from a raw OHLCV multi-index frame."""
    field = _get_price_field_for_ticker(df, ticker)
    series = pd.to_numeric(df[(ticker, field)], errors="coerce")
    series.name = ticker
    return series.sort_index()


def _to_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize either raw OHLCV MultiIndex or flat close matrix to close matrix."""
    _validate_input(df)
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels != 2:
            raise ValueError("Expected 2-level MultiIndex for OHLCV input.")
        tickers = sorted(df.columns.get_level_values(0).unique())
        close_map: dict[str, pd.Series] = {}
        for ticker in tickers:
            try:
                close_map[ticker] = _get_ticker_series(df, ticker)
            except ValueError:
                continue
        if not close_map:
            raise ValueError("No tickers with Close/Adj Close fields found.")
        out = pd.DataFrame(close_map, index=df.index)
    else:
        out = df.copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _align_pair(df: pd.DataFrame, pair: Pair) -> pd.DataFrame:
    """Return aligned two-column frame for pair with null rows dropped."""
    a, b = pair
    matrix = _to_close_matrix(df)
    if a not in matrix.columns or b not in matrix.columns:
        return pd.DataFrame(columns=[a, b], dtype=float)

    joined = pd.concat([matrix[a], matrix[b]], axis=1)
    joined.columns = [a, b]
    return joined.dropna(how="any").sort_index()


def unique_tickers(df: pd.DataFrame) -> list[str]:
    """Return sorted unique ticker names from level-0 columns."""
    matrix = _to_close_matrix(df)
    return sorted(matrix.columns.tolist())


def find_correlated_pairs(
    df: pd.DataFrame, window: int = 250, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Return positively correlated candidate pairs above threshold."""
    matrix = _to_close_matrix(df)
    if window <= 1:
        raise ValueError("window must be > 1")

    tickers = sorted(matrix.columns.tolist())
    if len(tickers) < 2:
        return []

    candidates: list[tuple[str, str, float]] = []
    for a, b in combinations(tickers, 2):
        pair_df = pd.concat([matrix[a], matrix[b]], axis=1).dropna(how="any")
        if len(pair_df) < window:
            continue
        pair_window = pair_df.tail(window)
        corr = float(pair_window[a].corr(pair_window[b]))
        if np.isnan(corr):
            continue
        if corr >= threshold:
            candidates.append((a, b, corr))
    return candidates


def find_cointegrated_pairs(
    df: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str] | tuple[str, str, float]],
    window: int = 500,
    alpha: float = 0.05,
) -> list[tuple[str, str, float]]:
    """Run Engle-Granger test for candidate pairs and return significant results."""
    _to_close_matrix(df)
    if window <= 2:
        raise ValueError("window must be > 2")

    results: list[tuple[str, str, float]] = []
    for item in candidate_pairs:
        a, b = item[0], item[1]
        aligned = _align_pair(df, (a, b))
        if len(aligned) < window:
            continue
        aligned = aligned.tail(window)
        try:
            _, pvalue, _ = coint(aligned[a], aligned[b])
        except Exception:
            continue
        if np.isnan(pvalue):
            continue
        if pvalue < alpha:
            results.append((a, b, float(pvalue)))
    return results


def calculate_hedge_ratio(df: pd.DataFrame, pair: Pair, window: int = 250) -> float | None:
    """Compute hedge ratio by regressing pair[0] on pair[1]."""
    _to_close_matrix(df)
    if window <= 1:
        raise ValueError("window must be > 1")

    aligned = _align_pair(df, pair)
    if len(aligned) < window:
        return None
    aligned = aligned.tail(window)
    y = aligned[pair[0]].to_numpy(dtype=float)
    x = aligned[pair[1]].to_numpy(dtype=float)

    if np.allclose(x.std(ddof=0), 0.0):
        return None

    x_design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x_design, y, rcond=None)[0]
    return float(beta[1])


def calculate_spread(
    df: pd.DataFrame, pair: Pair, hedge_ratio: float, window: int = 90
) -> pd.Series:
    """Compute spread series over the latest window."""
    _to_close_matrix(df)
    if window <= 1:
        raise ValueError("window must be > 1")

    aligned = _align_pair(df, pair)
    if len(aligned) < window:
        return pd.Series(dtype=float, name="spread")

    aligned = aligned.tail(window)
    spread = aligned[pair[0]] - hedge_ratio * aligned[pair[1]]
    spread.name = "spread"
    return spread


def calculate_zscore(spread_series: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling z-score for spread series."""
    if window <= 1:
        raise ValueError("window must be > 1")

    spread = pd.to_numeric(spread_series, errors="coerce").dropna()
    if len(spread) < window:
        return pd.Series(dtype=float, name="zscore")

    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std = spread.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
    zscore = zscore.dropna()
    zscore.name = "zscore"
    return zscore
