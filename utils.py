from __future__ import annotations

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
    """Return positively correlated candidate pairs above threshold.

    Uses a full-window correlation matrix for speed on wide universes (equivalent
    to pairwise complete observations within the window for each pair).
    """
    matrix = _to_close_matrix(df)
    if window <= 1:
        raise ValueError("window must be > 1")

    tickers = sorted(matrix.columns.tolist())
    if len(tickers) < 2:
        return []

    tail = matrix.tail(window)
    if len(tail) < window:
        return []

    corr_mat = tail.corr()
    candidates: list[tuple[str, str, float]] = []
    for i, a in enumerate(tickers):
        if a not in corr_mat.columns:
            continue
        for b in tickers[i + 1 :]:
            if b not in corr_mat.columns:
                continue
            corr = float(corr_mat.loc[a, b])
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
    """Compute rolling z-score for spread series (trailing; preserves the input index)."""
    if window <= 1:
        raise ValueError("window must be > 1")

    spread = pd.to_numeric(spread_series, errors="coerce")
    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std = spread.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
    zscore.name = "zscore"
    return zscore


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing rolling z-score (alias aligned with strategy naming)."""
    return calculate_zscore(series, window)


def calculate_log_hedge_ratio(
    df: pd.DataFrame, pair: Pair, window: int = 250
) -> tuple[float, float] | None:
    """OLS on log(A) ~ alpha + beta*log(B); returns (alpha, beta) or None."""
    if window <= 1:
        raise ValueError("window must be > 1")

    aligned = _align_pair(df, pair)
    if len(aligned) < window:
        return None

    tail = aligned.tail(window)
    log_a = np.log(pd.to_numeric(tail[pair[0]], errors="coerce"))
    log_b = np.log(pd.to_numeric(tail[pair[1]], errors="coerce"))
    mask = log_a.notna() & log_b.notna()
    log_a = log_a[mask].to_numpy(dtype=float)
    log_b = log_b[mask].to_numpy(dtype=float)
    if len(log_a) < 2 or np.allclose(np.std(log_b, ddof=0), 0.0):
        return None

    x_design = np.column_stack([np.ones(len(log_b)), log_b])
    coef = np.linalg.lstsq(x_design, log_a, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def calculate_log_spread_series(
    df: pd.DataFrame,
    pair: Pair,
    alpha: float,
    beta: float,
    window: int | None = None,
) -> pd.Series:
    """Log spread log(A) - alpha - beta*log(B); optionally restrict to last `window` rows."""
    aligned = _align_pair(df, pair)
    if aligned.empty:
        return pd.Series(dtype=float, name="log_spread")

    log_a = np.log(pd.to_numeric(aligned[pair[0]], errors="coerce"))
    log_b = np.log(pd.to_numeric(aligned[pair[1]], errors="coerce"))
    spread = log_a - alpha - beta * log_b
    spread.name = "log_spread"
    if window is not None and window > 0:
        return spread.tail(window)
    return spread


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average (trailing)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return pd.to_numeric(series, errors="coerce").rolling(window=n, min_periods=n).mean()


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential smoothing (matches common TA implementations)."""
    if period <= 0:
        raise ValueError("period must be positive")

    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "rsi"
    return rsi


def realized_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling stdev of log returns (annualization left to caller if needed)."""
    if window <= 1:
        raise ValueError("window must be > 1")

    s = pd.to_numeric(series, errors="coerce")
    lr = np.log(s / s.shift(1))
    rv = lr.rolling(window=window, min_periods=window).std(ddof=0)
    rv.name = "rv"
    return rv


def rolling_percentile_rank(series: pd.Series, window: int | None = None) -> pd.Series:
    """Percentile rank of each point vs strictly prior history (expanding if window is None).

    Values lie in [0, 1]; NaN until two usable observations exist for expanding mode.
    """
    x = pd.to_numeric(series, errors="coerce")
    if window is None:
        out = pd.Series(index=x.index, dtype=float)
        prior_vals: list[float] = []
        for dt in x.index:
            v = float(x.loc[dt]) if pd.notna(x.loc[dt]) else np.nan
            if not prior_vals or np.isnan(v):
                out.loc[dt] = np.nan
            else:
                arr = np.array(prior_vals)
                out.loc[dt] = float(np.mean(arr < v))
            if not np.isnan(v):
                prior_vals.append(v)
        out.name = "pct_rank"
        return out

    if window <= 1:
        raise ValueError("window must be > 1")

    def _pct_last(win: pd.Series) -> float:
        if len(win) < window or win.isna().all():
            return np.nan
        cur = win.iloc[-1]
        if np.isnan(cur):
            return np.nan
        hist = win.iloc[:-1].dropna()
        if hist.empty:
            return np.nan
        return float(np.mean(hist.to_numpy(dtype=float) < float(cur)))

    return x.rolling(window=window, min_periods=window).apply(_pct_last, raw=False)


def high_vol_regime_flag(
    rv_series: pd.Series, quantile: float = 0.8, min_periods: int = 40
) -> pd.Series:
    """True when today's RV is >= `quantile` of the expanding distribution of prior RV values."""
    x = pd.to_numeric(rv_series, errors="coerce")
    thr = x.shift(1).expanding(min_periods=min_periods).quantile(quantile)
    flag = (x >= thr) & thr.notna()
    flag.name = "high_vol"
    return flag
