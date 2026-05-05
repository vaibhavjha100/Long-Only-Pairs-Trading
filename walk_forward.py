'''
Walk forward module

Walk forward the strategy and calculates the portfolio metrics.

Use this cost structure:
  brokerage_rate:       0.0003    (0.03% per side)
  brokerage_max:        20.0      (INR cap per order)
  minimum of brokerage rat and max is taken as the brokerage cost
  stt:              0.001
  exchange_txn_charge:  0.0000335
  sebi_charge:          0.000001
  stamp_duty_buy:       0.00015   (buy side only)
  stamp_duty_sell:      0.0
  gst_rate:             0.18      (on brokerage + exchange_charge + sebi)
  dp_sell_charge:       13.5      (fixed INR per scrip per sell, CDSL)

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

# Cost Parameters
brokerage_rate = 0.0003
brokerage_max = 20.0
stt = 0.001
exchange_txn_charge = 0.0000335
sebi_charge = 0.000001
stamp_duty_buy = 0.00015
stamp_duty_sell = 0.0
gst_rate = 0.18
dp_sell_charge = 13.5

initial_capital = 100000

# Initialize portfolio dataframe
portfolio = pd.DataFrame(index=nifty500.index, columns=['gross_portfolio_value', 'transaction_cost', 'tax', 'net_portfolio_value', 'gross_return', 'net_return', 'holdings', 'cash', 'trades', 'pnl', 'market_regime', 'num_open_positions', 'num_trades', 'trade_turnover'])
portfolio['gross_portfolio_value'] = initial_capital
portfolio['transaction_cost'] = 0
portfolio['tax'] = 0
portfolio['net_portfolio_value'] = initial_capital
portfolio['gross_return'] = 0
portfolio['net_return'] = 0

for i in nifty500.index:
    if i not in signals.index:
        continue