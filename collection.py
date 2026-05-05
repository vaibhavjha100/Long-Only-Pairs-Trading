'''
Collection module

Collect close price data for stocks in NIFTY500 and the NIFTY50 index.
Data Source: Yahoo Finance through yfinance library.

Collect sector data for each stock in NIFTY500.

Saves the data to csv files: nifty500.csv and nifty50.csv.
Saves sector data to pickle file: sector_data.pkl.
'''

import yfinance as yf
import pandas as pd
import pickle

nifty500_list = pd.read_csv('ind_nifty500list.csv')
if 'Symbol' not in nifty500_list.columns:
    raise ValueError("ind_nifty500list.csv must contain a 'Symbol' column.")
tickers = nifty500_list['Symbol'].dropna().astype(str).tolist()

# Add '.NS' to the end of each ticker
tickers = [ticker + '.NS' for ticker in tickers]

# Collect data for all tickers in a df
# Daily data with max period
# Only include Close price with ticker as column name

def _download_close_series(symbol: str):
    """Return Close as Series named ``symbol``, or None if missing/empty."""
    data = yf.download(
        symbol,
        period='max',
        interval='1d',
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if data is None or getattr(data, 'empty', True):
        return None
    if 'Close' not in data.columns:
        return None
    close = data['Close'].copy()
    close.name = symbol
    return close


_close_cols = []
for ticker in tickers:
    s = _download_close_series(ticker)
    if s is not None:
        _close_cols.append(s)

if not _close_cols:
    raise ValueError('No NIFTY500 close data downloaded; check tickers and network.')

df = pd.concat(_close_cols, axis=1, sort=True)

nifty50_data = yf.download(
    '^NSEI',
    period='max',
    interval='1d',
    auto_adjust=True,
    progress=False,
    multi_level_index=False,
)
if nifty50_data is None or nifty50_data.empty or 'Close' not in nifty50_data.columns:
    raise ValueError('Failed to download ^NSEI (nifty50 proxy); check network and symbol.')

# Print for both dfs
print(df.head())
print(nifty50_data.head())

print(df.info())
print(nifty50_data.info())

# Collect sector data for each stock in NIFTY500
sector_data = {}
for ticker in tickers:
    info = yf.Ticker(ticker).info
    if not isinstance(info, dict):
        info = {}
    sector_data[ticker] = info.get('sector')

# Save sector data to pickle file
with open('sector_data.pkl', 'wb') as f:
    pickle.dump(sector_data, f)

# Save the data to csv files
df.to_csv('nifty500.csv')
nifty50_data.to_csv('nifty50.csv')

print("Data collection complete")