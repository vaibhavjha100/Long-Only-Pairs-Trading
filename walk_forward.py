'''
Walk forward module

Walk forward the strategy and calculates the portfolio metrics.

Returns:
- portfolio returns dataframe
- index is the date
- multiple columns:
    - gross portfolio value
    - transaction cost
    - tax
    - net portfolio value
    - gross return
    - net return
    - holdings
    - cash
    - trades
    - pnl
    - market regime
    - num_open_positions
    - num_trades
    - trade_turnover

Saves the backtest results to a csv file: backtest_results.csv.
'''

import pandas as pd
import numpy as np

# Load signals.csv
signals = pd.read_csv('signals.csv', index_col=0, parse_dates=[0])

# Load nifty500.csv
nifty500 = pd.read_csv('nifty500.csv', index_col=0, parse_dates=[0])

# Load nifty50.csv
nifty50 = pd.read_csv('nifty50.csv', index_col=0, parse_dates=[0])

# Slice nifty500 from same start date as nifty50
nifty500 = nifty500.loc[nifty50.index[0]:]

def market_regime(nifty50):
    # Calculate 200 day simple moving average for the last date
    sma200 = nifty50.iloc[-1].rolling(200).mean()
    price = nifty50['Close'].iloc[-1]

    if price > sma200:
        return 'Up'
    else:
        return 'Down'



for i in nifty500.index:
    if i not in signals.index:
        continue