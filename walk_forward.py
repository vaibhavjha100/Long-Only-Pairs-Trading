'''
Walk forward module

Walk forward the strategy and calculates the portfolio metrics.

Use this cost structure:
  transaction cost:     0.6%     (0.6% of turnover)
  dp_sell_charge:       13.5      (fixed INR per scrip per sell, CDSL)
  

Returns:
- portfolio returns dataframe
- index is the date
- multiple columns:
    - gross portfolio value
    - transaction cost
    - net portfolio value
    - gross return
    - net return
    - holdings
    - cash
    - trades
    - gross pnl
    - net pnl
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

# Shift signals by 1 day to avoid look-ahead bias
signals = signals.shift(1)

def market_regime(nifty50):
    # Calculate 200 day simple moving average for the last date
    sma200 = nifty50.iloc[-1].rolling(200).mean()
    price = nifty50['Close'].iloc[-1]

    if price > sma200:
        return 'Up'
    else:
        return 'Down'

def check_holdings(holdings: dict[str, int]) -> bool:
    for ticker, quantity in holdings.items():
        if quantity > 0:
            return True
    return False

# Cost Parameters
transaction_cost = 0.155
dp_sell_charge = 13.5

initial_capital = 100000

# Initialize portfolio dataframe
portfolio = pd.DataFrame(index=nifty500.index, columns=['gross_portfolio_value', 'transaction_cost', 'net_portfolio_value', 'gross_return', 'net_return', 'holdings', 'cash', 'trades', 'gross_pnl', 'net_pnl', 'market_regime', 'num_open_positions', 'num_trades', 'trade_turnover'])


for i in nifty500.index:
    if i not in signals.index:
        # Take all siganls as 0
        signals.loc[i] = 0
    mreg = market_regime(nifty50)
    # Calculate cash available as previous day's cash or initial capital
    cash = portfolio.loc[i-1, 'cash'] if i-1 in portfolio.index else initial_capital

    holdings = portfolio.loc[i-1, 'holdings'] if i-1 in portfolio.index else {ticker: 0 for ticker in signals.columns}
    trades = 0
    trade_turnover = 0
    transaction_cost = 0

    # All tickers with positive signals
    buy_tickers = signals.loc[i, signals.loc[i] > 0].index

    # All tickers with negative signals
    sell_tickers = signals.loc[i, signals.loc[i] < 0].index

    # We sell first to get cash
    
    # Check if there are any holdings to sell
    if check_holdings(holdings):
        if len(sell_tickers) > 0:
            # Sell the holdings
            for ticker in sell_tickers:
                if ticker in holdings:
                    # Sell the holding
                    cash += holdings[ticker] * nifty500.loc[i, ticker]
                    holdings[ticker] = 0
                    trades += 1
                    trade_turnover += holdings[ticker] * nifty500.loc[i, ticker]
                    transaction_cost += trade_turnover * transaction_cost + dp_sell_charge
        if mreg == 'Down':
            # Sell all holdings
            for ticker, quantity in holdings.items():
                if quantity > 0:
                    # Sell the holding
                    cash += quantity * nifty500.loc[i, ticker]
                    holdings[ticker] = 0
                    trades += 1
                    trade_turnover += quantity * nifty500.loc[i, ticker]
                    transaction_cost += trade_turnover * transaction_cost + dp_sell_charge
    if len(buy_tickers) > 0:
        pass # TODO: Fill buy orders