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
tickers = nifty500_list['Symbol'].tolist()

# Add '.NS' to the end of each ticker
tickers = [ticker + '.NS' for ticker in tickers]

# Collect data for all tickers in a df
# Daily data with max period
# Only include Close price with ticker as column name

df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, period='max', interval='1d', auto_adjust=True, multi_level_index=False)
    data = data['Close']
    data.name = ticker
    df = pd.concat([df, data], axis=1, sort=True)

nifty50_data = yf.download('^NSEI', period='max', interval='1d', auto_adjust=True, multi_level_index=False)

# Print for both dfs
print(df.head())
print(nifty50_data.head())

print(df.info())
print(nifty50_data.info())

# Collect sector data for each stock in NIFTY500
sector_data = {}
for ticker in tickers:
    sector_data[ticker] = yf.Ticker(ticker).info['sector']

# Save sector data to pickle file
with open('sector_data.pkl', 'wb') as f:
    pickle.dump(sector_data, f)

# Save the data to csv files
df.to_csv('nifty500.csv')
nifty50_data.to_csv('nifty50.csv')

print("Data collection complete")