from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from utils import (
    Pair,
    calculate_log_hedge_ratio,
    calculate_log_spread_series,
    find_cointegrated_pairs,
    find_correlated_pairs,
    high_vol_regime_flag,
    realized_volatility,
    rolling_zscore,
    rsi_wilder,
    sma,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# India cash-market style fee stack (illustrative; GST applies to fee portion only).
INDIA_EQ_COSTS: dict[str, float] = {
    "brokerage_pct": 0.0003,
    "exchange_txn_pct": 0.0000325,
    "sebi_pct": 0.000001,
    "stamp_duty_buy_pct": 0.00003,
    "stt_sell_pct": 0.001,
    "gst_rate": 0.18,
}


def _trade_costs(side: Literal["buy", "sell"], notional: float, c: dict[str, float]) -> tuple[float, float]:
    """Return (transaction_costs_excluding_gst, gst) for one trade leg."""
    if notional <= 0 or np.isnan(notional):
        return 0.0, 0.0
    br = notional * c["brokerage_pct"]
    ex = notional * c["exchange_txn_pct"]
    sebi = notional * c["sebi_pct"]
    stamp = notional * c["stamp_duty_buy_pct"] if side == "buy" else 0.0
    stt = notional * c["stt_sell_pct"] if side == "sell" else 0.0
    txn = br + ex + sebi + stamp + stt
    gst_base = br + ex + sebi
    gst = c["gst_rate"] * gst_base
    return float(txn), float(gst)


def _prev_trading_day(cal: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    loc = cal.get_loc(d)
    if isinstance(loc, slice):
        loc = loc.start
    i = int(loc)
    if i == 0:
        raise ValueError("No previous trading day.")
    return cal[i - 1]


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _z_at_date(z: pd.Series, d: pd.Timestamp) -> float | None:
    if d not in z.index:
        return None
    v = _safe_float(z.loc[d])
    return v


@dataclass
class OpenLeg:
    """Long position in `symbol` opened from pair (A,B) with fixed ordering."""

    symbol: str
    partner: str
    pair: Pair
    leg: Literal["A", "B"]
    entry_style: Literal["low_z", "high_z"]
    shares: int


@dataclass
class BacktestParams:
    corr_window: int = 250
    corr_threshold: float = 0.8
    coint_window: int = 500
    coint_alpha: float = 0.05
    max_corr_pairs_for_coint: int = 400
    hedge_window: int = 250
    spread_window: int = 90
    z_window: int = 60
    z_entry: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_cross_lookback: int = 5
    sma_fast: int = 50
    sma_slow: int = 200
    rv_window: int = 20
    vol_quantile: float = 0.8
    vol_regime_min_periods: int = 40
    risk_frac: float = 0.025
    per_share_risk_frac: float = 0.025
    max_position_frac: float = 0.40
    starting_cash: float = 100_000.0


def pair_z_on_date(
    df_upto_s: pd.DataFrame,
    pair: Pair,
    hedge_window: int,
    z_window: int,
    s: pd.Timestamp,
) -> float | None:
    """Z at signal date `s` (OLS + spread use only a tail slice for speed)."""
    coef = calculate_log_hedge_ratio(df_upto_s, pair, hedge_window)
    if coef is None:
        return None
    alpha, beta = coef
    tail_rows = max(hedge_window, z_window) + 40
    df_tail = df_upto_s.tail(tail_rows)
    spread = calculate_log_spread_series(df_tail, pair, alpha, beta, window=None)
    z = rolling_zscore(spread, z_window)
    return _z_at_date(z, s)


def run_backtest(
    universe: pd.DataFrame,
    benchmark: pd.DataFrame,
    params: BacktestParams | None = None,
    costs: dict[str, float] | None = None,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Daily walk-forward long-only pairs backtest; signal at `s`, execute at `t` close."""
    p = params or BacktestParams()
    c = costs or INDIA_EQ_COSTS

    bench_col = "NIFTY50" if "NIFTY50" in benchmark.columns else benchmark.columns[0]
    universe = universe.sort_index().copy()
    benchmark = benchmark.sort_index().copy()
    universe = universe[~universe.index.duplicated(keep="last")]
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")]

    cal_full = universe.index.intersection(benchmark.index).sort_values()
    if len(cal_full) < 3:
        raise ValueError("Insufficient overlapping calendar dates.")

    min_start = max(p.coint_window, p.corr_window, p.hedge_window, p.z_window, p.sma_slow, p.rv_window) + 5
    if len(cal_full) <= min_start:
        raise ValueError("Full overlapping calendar too short for configured windows.")

    cash = float(p.starting_cash)
    positions: dict[str, OpenLeg] = {}
    rows: list[dict[str, Any]] = []

    loop_dates = cal_full[min_start + 1 :]
    if start_date is not None:
        loop_dates = loop_dates[loop_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        loop_dates = loop_dates[loop_dates <= pd.Timestamp(end_date)]
    if len(loop_dates) == 0:
        raise ValueError("No trading days in requested range after warm-up window.")

    for t in loop_dates:
        s = _prev_trading_day(cal_full, t)

        df_u = universe.loc[:s]
        bench_s = benchmark.loc[:s, bench_col]

        # --- Portfolio value at s close (equity_s) ---
        equity_s = cash
        for sym, open_leg in list(positions.items()):
            px = _safe_float(df_u.loc[s, sym]) if sym in df_u.columns else None
            if px is None:
                continue
            equity_s += open_leg.shares * px

        # --- Benchmark regimes at s ---
        close_b = pd.to_numeric(bench_s, errors="coerce")
        sma50 = sma(close_b, p.sma_fast)
        sma200 = sma(close_b, p.sma_slow)
        rv = realized_volatility(close_b, p.rv_window)
        hi_vol = high_vol_regime_flag(rv, quantile=p.vol_quantile, min_periods=p.vol_regime_min_periods)

        cs = _safe_float(close_b.loc[s]) if s in close_b.index else None
        bull = False
        if cs is not None and s in sma50.index and s in sma200.index:
            s50 = _safe_float(sma50.loc[s])
            s200 = _safe_float(sma200.loc[s])
            if s50 is not None and s200 is not None:
                bull = cs > s200 and s50 > s200

        high_vol = bool(hi_vol.loc[s]) if s in hi_vol.index else False

        # --- Screening at s ---
        corr_pairs = find_correlated_pairs(df_u, window=p.corr_window, threshold=p.corr_threshold)
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        corr_trimmed = corr_pairs[: p.max_corr_pairs_for_coint]
        coint_pairs = find_cointegrated_pairs(
            df_u, corr_trimmed, window=p.coint_window, alpha=p.coint_alpha
        )

        # Candidate entries: symbol -> (abs_z, pair tuple ordered as screen, z_s, leg)
        candidates: dict[str, tuple[float, Pair, float, Literal["A", "B"]]] = {}

        for a, b, _pv in coint_pairs:
            pair: Pair = (a, b)
            z_s = pair_z_on_date(df_u, pair, p.hedge_window, p.z_window, s)
            if z_s is None or np.isnan(z_s):
                continue
            if z_s <= -p.z_entry:
                sym, leg = a, "A"
                key = abs(z_s)
                prev = candidates.get(sym)
                if prev is None or key > prev[0]:
                    candidates[sym] = (key, pair, float(z_s), leg)
            if z_s >= p.z_entry:
                sym, leg = b, "B"
                key = abs(z_s)
                prev = candidates.get(sym)
                if prev is None or key > prev[0]:
                    candidates[sym] = (key, pair, float(z_s), leg)

        # --- Exit intents at s (execute at t) ---
        exit_symbols: set[str] = set()
        for sym, ol in list(positions.items()):
            a, b = ol.pair
            z_s = pair_z_on_date(df_u, (a, b), p.hedge_window, p.z_window, s)
            if z_s is None:
                continue

            # Z exit
            if ol.entry_style == "low_z" and z_s >= 0:
                exit_symbols.add(sym)
                continue
            if ol.entry_style == "high_z" and z_s <= 0:
                exit_symbols.add(sym)
                continue

            # Regime stop
            if not bull or high_vol:
                exit_symbols.add(sym)
                continue

            # Cointegration break
            still = find_cointegrated_pairs(df_u, [(a, b)], window=p.coint_window, alpha=p.coint_alpha)
            if not still:
                exit_symbols.add(sym)

        # --- Entry intents after filters (signal still from s) ---
        entry_requests: list[tuple[str, Pair, Literal["A", "B"], float]] = []
        held = set(positions.keys())

        for sym, (_absz, pair, z_val, leg) in candidates.items():
            if sym in held:
                continue
            if sym not in df_u.columns:
                continue

            # RSI + oversold history + crossover at s
            px_series = pd.to_numeric(df_u[sym], errors="coerce")
            rsi = rsi_wilder(px_series, p.rsi_period)
            if s not in rsi.index:
                continue
            tail = rsi.loc[:s].tail(p.rsi_cross_lookback)
            if tail.isna().all() or not (tail < p.rsi_oversold).any():
                continue
            s_prev = _prev_trading_day(cal_full, s)
            r_prev = _safe_float(rsi.loc[s_prev]) if s_prev in rsi.index else None
            r_s = _safe_float(rsi.loc[s])
            if r_prev is None or r_s is None:
                continue
            if not (r_prev < p.rsi_oversold and r_s >= p.rsi_oversold):
                continue

            if not bull or high_vol:
                continue

            entry_requests.append((sym, pair, leg, z_val))

        # ========= Trading at t close: sell first =========
        txn_day = 0.0
        gst_day = 0.0
        trades_t: list[dict[str, Any]] = []

        px_t_all = universe.loc[t] if t in universe.index else None

        for sym in list(exit_symbols):
            open_leg = positions.pop(sym, None)
            if open_leg is None or px_t_all is None or sym not in px_t_all.index:
                continue
            px = _safe_float(px_t_all[sym])
            if px is None:
                continue
            notional = open_leg.shares * px
            txn, gst = _trade_costs("sell", notional, c)
            cash += notional - txn - gst
            txn_day += txn
            gst_day += gst
            trades_t.append(
                {
                    "action": "sell",
                    "symbol": sym,
                    "shares": open_leg.shares,
                    "price": px,
                    "notional": notional,
                    "transaction_costs": txn,
                    "gst": gst,
                    "reason": "exit",
                }
            )

        # --- Entries: sizing from equity_s, price at t ---
        equity_s = cash
        for sym, open_leg in positions.items():
            px_s = _safe_float(df_u.loc[s, sym]) if sym in df_u.columns else None
            if px_s is None:
                continue
            equity_s += open_leg.shares * px_s

        for sym, pair, leg, _zv in entry_requests:
            if sym in positions:
                continue
            if px_t_all is None or sym not in px_t_all.index:
                continue
            px_t = _safe_float(px_t_all[sym])
            if px_t is None or px_t <= 0:
                continue

            risk_dollars = equity_s * p.risk_frac
            per_share_risk = px_t * p.per_share_risk_frac
            if per_share_risk <= 0:
                continue
            shares = int(np.floor(risk_dollars / per_share_risk))
            max_sh = int(np.floor(p.max_position_frac * equity_s / px_t))
            shares = max(0, min(shares, max_sh))
            if shares <= 0:
                continue

            notional = shares * px_t
            txn, gst = _trade_costs("buy", notional, c)
            if cash < notional + txn + gst:
                continue

            cash -= notional + txn + gst
            txn_day += txn
            gst_day += gst

            partner = pair[1] if leg == "A" else pair[0]
            style: Literal["low_z", "high_z"] = "low_z" if leg == "A" else "high_z"
            positions[sym] = OpenLeg(
                symbol=sym,
                partner=partner,
                pair=pair,
                leg=leg,
                entry_style=style,
                shares=shares,
            )
            trades_t.append(
                {
                    "action": "buy",
                    "symbol": sym,
                    "shares": shares,
                    "price": px_t,
                    "notional": notional,
                    "transaction_costs": txn,
                    "gst": gst,
                    "reason": "entry",
                    "pair": list(pair),
                    "leg": leg,
                }
            )

        # --- Mark to market at t (cash already net of today's fees) ---
        equity_end = cash
        holdings: dict[str, int] = {}
        if px_t_all is not None:
            for sym, open_leg in positions.items():
                if sym not in px_t_all.index:
                    continue
                px = _safe_float(px_t_all[sym])
                if px is None:
                    continue
                equity_end += open_leg.shares * px
                holdings[sym] = open_leg.shares

        net_t = float(equity_end)
        gross_t = net_t + txn_day + gst_day
        prev_net = float(rows[-1]["net_portfolio_value"]) if rows else float(p.starting_cash)
        daily_pnl = net_t - prev_net
        daily_ret = daily_pnl / prev_net if prev_net > 0 else 0.0

        rows.append(
            {
                "date": t,
                "gross_portfolio_value": gross_t,
                "transaction_costs": txn_day,
                "tax": gst_day,
                "net_portfolio_value": net_t,
                "cash": cash,
                "holdings": holdings,
                "trades": trades_t,
                "num_open_positions": len(positions),
                "daily_pnl": daily_pnl,
                "daily_return": daily_ret,
                "market_regime": bull,
                "vol_regime": high_vol,
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    return out


def assert_results_sanity(df: pd.DataFrame) -> None:
    required = (
        "gross_portfolio_value",
        "transaction_costs",
        "tax",
        "net_portfolio_value",
        "cash",
        "holdings",
        "trades",
        "num_open_positions",
        "daily_pnl",
        "daily_return",
        "market_regime",
        "vol_regime",
    )
    missing = set(required) - set(df.columns)
    if missing:
        raise AssertionError(f"results missing columns: {sorted(missing)}")
    if not (df["transaction_costs"] >= -1e-9).all():
        raise AssertionError("negative transaction_costs")
    if not (df["tax"] >= -1e-9).all():
        raise AssertionError("negative tax")
    if not (df["num_open_positions"] >= 0).all():
        raise AssertionError("negative position count")


def results_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Serialize dict/list columns as JSON for CSV."""
    export = df.copy()
    for col in ("holdings", "trades"):
        if col in export.columns:
            export[col] = export[col].apply(lambda x: json.dumps(x, default=str))
    export.to_csv(path)


def load_inputs(
    universe_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load flat CSV close matrices from project root."""
    u_path = universe_path or PROJECT_ROOT / "nifty500.csv"
    b_path = benchmark_path or PROJECT_ROOT / "nifty50.csv"
    universe = pd.read_csv(u_path, index_col=0, parse_dates=[0])
    benchmark = pd.read_csv(b_path, index_col=0, parse_dates=[0])
    return universe, benchmark


def main() -> int:
    universe, benchmark = load_inputs()
    # Short recent slice: daily Engle–Granger on many pairs is expensive.
    end = universe.index.max()
    start = end - pd.Timedelta(days=14)
    slim = BacktestParams(max_corr_pairs_for_coint=20)
    bt = run_backtest(universe, benchmark, params=slim, start_date=start, end_date=end)
    assert_results_sanity(bt)
    out_path = PROJECT_ROOT / "results.csv"
    results_to_csv(bt, out_path)
    print(f"Wrote {out_path} with {len(bt)} rows.")
    print(bt.tail(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
