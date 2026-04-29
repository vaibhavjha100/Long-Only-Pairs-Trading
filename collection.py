from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent
SYMBOLS_CSV = PROJECT_ROOT / "ind_nifty500list.csv"
NIFTY500_OUTPUT = PROJECT_ROOT / "nifty500.csv"
NIFTY50_OUTPUT = PROJECT_ROOT / "nifty50.csv"
NIFTY50_TICKER = "^NSEI"


def load_nifty500_symbols(csv_path: Path) -> list[str]:
    """Load, clean, and deduplicate NIFTY500 symbols from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing symbols file: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "Symbol" not in frame.columns:
        raise ValueError("Expected a 'Symbol' column in ind_nifty500list.csv")

    symbols = (
        frame["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return symbols


def to_yahoo_ticker(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance symbol."""
    return f"{symbol}.NS"


def _extract_close_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to a simple close-price matrix."""
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        level1 = set(data.columns.get_level_values(1))
        # Handle both yfinance variants:
        # 1) (Ticker, Field) when group_by="ticker"
        # 2) (Field, Ticker) default for some single-ticker downloads
        if "Close" in level0 or "Adj Close" in level0:
            field = "Close" if "Close" in level0 else "Adj Close"
            close_df = data.xs(field, axis=1, level=0)
        elif "Close" in level1 or "Adj Close" in level1:
            tickers = sorted(level0)
            close_map: dict[str, pd.Series] = {}
            for ticker in tickers:
                fields = list(data[ticker].columns)
                field = "Close" if "Close" in fields else "Adj Close" if "Adj Close" in fields else None
                if field is None:
                    continue
                close_map[ticker] = pd.to_numeric(data[(ticker, field)], errors="coerce")
            close_df = pd.DataFrame(close_map, index=data.index)
        else:
            raise ValueError("Could not find Close/Adj Close fields in downloaded data.")
    else:
        field = "Close" if "Close" in data.columns else "Adj Close" if "Adj Close" in data.columns else None
        if field is None:
            raise ValueError("Could not find Close/Adj Close in benchmark download.")
        close_df = pd.DataFrame({"NIFTY50": pd.to_numeric(data[field], errors="coerce")}, index=data.index)

    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    return close_df


def download_nifty500(symbols: list[str], output_path: Path) -> tuple[bool, str]:
    """Download full daily NIFTY500 close data into one flat CSV."""
    if not symbols:
        return False, "No symbols found in source CSV."

    yahoo_tickers = [to_yahoo_ticker(symbol) for symbol in symbols]
    print(f"Downloading NIFTY500 constituents: {len(yahoo_tickers)} tickers")

    raw = yf.download(
        tickers=yahoo_tickers,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        return False, "NIFTY500 download returned no data."

    data = _extract_close_matrix(raw)
    if data.empty:
        return False, "NIFTY500 close extraction returned no data."

    data.to_csv(output_path, index_label="Date")
    return True, f"Saved {output_path.name} with shape={data.shape}"


def download_nifty50(output_path: Path) -> tuple[bool, str]:
    """Download full daily NIFTY50 benchmark close data."""
    print(f"Downloading benchmark: {NIFTY50_TICKER}")
    raw = yf.download(
        tickers=NIFTY50_TICKER,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return False, "NIFTY50 benchmark download returned no data."

    data = _extract_close_matrix(raw)
    if "NIFTY50" not in data.columns and len(data.columns) == 1:
        data = data.rename(columns={data.columns[0]: "NIFTY50"})
    data.to_csv(output_path)
    return True, f"Saved {output_path.name} with shape={data.shape}"


def main() -> int:
    print("Starting data collection...")
    print(f"Project root: {PROJECT_ROOT}")

    try:
        symbols = load_nifty500_symbols(SYMBOLS_CSV)
        print(f"Loaded {len(symbols)} symbols from {SYMBOLS_CSV.name}")
    except Exception as exc:
        print(f"[ERROR] Could not read symbols CSV: {exc}")
        return 1

    success_500 = False
    success_50 = False

    try:
        success_500, msg_500 = download_nifty500(symbols, NIFTY500_OUTPUT)
        print(f"[NIFTY500] {msg_500}")
    except Exception as exc:
        print(f"[NIFTY500] Failed with exception: {exc}")

    try:
        success_50, msg_50 = download_nifty50(NIFTY50_OUTPUT)
        print(f"[NIFTY50] {msg_50}")
    except Exception as exc:
        print(f"[NIFTY50] Failed with exception: {exc}")

    print("\nSummary")
    print(f"- NIFTY500 output: {'ok' if success_500 else 'failed'} ({NIFTY500_OUTPUT})")
    print(f"- NIFTY50 output: {'ok' if success_50 else 'failed'} ({NIFTY50_OUTPUT})")

    if success_500 and success_50:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
