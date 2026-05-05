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
    - transaction cost
    - portfolio value
    - daily return
    - holdings
    - cash
    - trades
    - daily pnl
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
transaction_cost_rate = 0.006
dp_sell_charge = 13.5

initial_capital = 100000

# Initialize portfolio dataframe
portfolio = pd.DataFrame(index=nifty500.index, columns=['portfolio_value', 'transaction_cost', 'daily_return', 'holdings', 'cash', 'trades', 'daily_pnl', 'market_regime', 'num_open_positions', 'num_trades', 'trade_turnover'])


for i in nifty500.index:
    # Skip 1st 200 days for market regime calculation
    if i < nifty50.index[200]:
        continue
    if i not in signals.index:
        # Take all siganls as 0
        signals.loc[i] = 0

    # Calculate market regime for the slice of nifty50 till the current date
    mreg = market_regime(nifty50.loc[nifty50.index <= i])
    # Calculate cash available as previous day's cash or initial capital
    cash = portfolio.loc[i-1, 'cash'].copy() if i-1 in portfolio.index else initial_capital

    holdings = portfolio.loc[i-1, 'holdings'].copy() if i-1 in portfolio.index else {ticker: 0 for ticker in signals.columns}
    trades = {ticker: 0 for ticker in signals.columns}
    trade_turnover = 0
    transaction_cost = portfolio.loc[i-1, 'transaction_cost'].copy() if i-1 in portfolio.index else 0

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
                    sell_amount = holdings[ticker] * nifty500.loc[i, ticker]
                    cash += sell_amount - sell_amount * transaction_cost_rate - dp_sell_charge
                    trade_turnover += sell_amount
                    transaction_cost += sell_amount * transaction_cost_rate + dp_sell_charge
                    trades[ticker] = -holdings[ticker]
                    holdings[ticker] = 0
        if mreg == 'Down':
            # Sell all holdings
            for ticker, quantity in holdings.items():
                if quantity > 0:
                    # Sell the holding
                    sell_amount = quantity * nifty500.loc[i, ticker]
                    cash += sell_amount - sell_amount * transaction_cost_rate - dp_sell_charge
                    trade_turnover += sell_amount
                    transaction_cost += sell_amount * transaction_cost_rate + dp_sell_charge
                    trades[ticker] = -quantity
                    holdings[ticker] = 0
    if len(buy_tickers) > 0 and mreg == 'Up':
        valid_buy_tickers = []
        for ticker in buy_tickers:
            if ticker not in holdings or ticker not in nifty500.columns:
                continue
            if pd.isna(nifty500.loc[i, ticker]) or nifty500.loc[i, ticker] <= 0:
                continue
            if pd.isna(signals.loc[i, ticker]) or signals.loc[i, ticker] <= 0:
                continue
            valid_buy_tickers.append(ticker)

        signal_sum = signals.loc[i, valid_buy_tickers].sum() if len(valid_buy_tickers) > 0 else 0

        while len(valid_buy_tickers) > 0 and signal_sum > 0:
            target_base = cash
            current_values = {}
            gaps = {}

            for ticker in valid_buy_tickers:
                current_values[ticker] = holdings[ticker] * nifty500.loc[i, ticker]
                target_base += current_values[ticker]

            for ticker in valid_buy_tickers:
                target_weight = signals.loc[i, ticker] / signal_sum
                target_value = target_base * target_weight
                gaps[ticker] = target_value - current_values[ticker]

            buy_candidates = []
            for ticker in valid_buy_tickers:
                buy_amount = nifty500.loc[i, ticker]
                buy_cost = buy_amount * transaction_cost_rate
                if gaps[ticker] > 0 and cash >= buy_amount + buy_cost:
                    buy_candidates.append(ticker)

            if len(buy_candidates) == 0:
                break

            ticker = max(buy_candidates, key=lambda x: gaps[x])
            buy_amount = nifty500.loc[i, ticker]
            buy_cost = buy_amount * transaction_cost_rate

            trades[ticker] += 1
            holdings[ticker] += 1
            trade_turnover += buy_amount
            transaction_cost += buy_cost
            cash -= buy_amount + buy_cost