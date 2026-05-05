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
nifty500 = nifty500.sort_index()

# Load nifty50.csv
nifty50 = pd.read_csv('nifty50.csv', index_col=0, parse_dates=[0])
nifty50 = nifty50.sort_index()

# Slice nifty500 from same start date as nifty50
nifty500 = nifty500.loc[nifty50.index[0]:]

# Shift signals by 1 day to avoid look-ahead bias
signals = signals.shift(1)
# Align to backtest calendar without mutating signals inside the loop
signals = signals.reindex(nifty500.index, fill_value=0)

def _valid_trade_price(px) -> bool:
    return pd.notna(px) and float(px) > 0

def market_regime(nifty50_slice):
    # 200-day SMA of Close at the last row of the slice (walk-forward safe)
    closes = nifty50_slice['Close']
    if len(closes) < 200:
        return 'Down'
    sma200 = closes.rolling(200).mean().iloc[-1]
    price = closes.iloc[-1]
    if pd.isna(sma200) or pd.isna(price):
        return 'Down'
    if price > sma200:
        return 'Up'
    return 'Down'

def check_holdings(holdings: dict[str, int]) -> bool:
    for ticker, quantity in holdings.items():
        if quantity > 0:
            return True
    return False


def holdings_market_value(i, holdings: dict) -> float:
    mv = 0.0
    for t, q in holdings.items():
        if q <= 0 or t not in nifty500.columns:
            continue
        px = nifty500.loc[i, t]
        if _valid_trade_price(px):
            mv += q * float(px)
    return mv


def last_filled_portfolio_pos(pos: int):
    """Index position of latest row at or before pos-1 with a computed portfolio_value."""
    for j in range(pos - 1, -1, -1):
        idx = nifty500.index[j]
        pv = portfolio.loc[idx, 'portfolio_value']
        if pd.notna(pv):
            return j
    return None


# Cost Parameters
transaction_cost_rate = 0.006
dp_sell_charge = 13.5

initial_capital = 100000

# Initialize portfolio dataframe
portfolio = pd.DataFrame(index=nifty500.index, columns=['portfolio_value', 'transaction_cost', 'daily_return', 'holdings', 'cash', 'trades', 'daily_pnl', 'market_regime', 'num_open_positions', 'num_trades', 'trade_turnover'])

# Skip dates before index position 200 on nifty50 (warm-up for SMA200), if history exists
_regime_skip_until = nifty50.index[200] if len(nifty50.index) > 200 else nifty50.index[0]

for i in nifty500.index:
    if i < _regime_skip_until:
        continue

    pos = nifty500.index.get_loc(i)
    if not isinstance(pos, int):
        continue

    prev_filled_pos = last_filled_portfolio_pos(pos)
    prev_filled_idx = nifty500.index[prev_filled_pos] if prev_filled_pos is not None else None

    # Calculate market regime for the slice of nifty50 till the current date
    mreg = market_regime(nifty50.loc[nifty50.index <= i])

    # Carry forward last computed portfolio state (handles skipped warmup rows)
    if prev_filled_idx is None:
        cash = initial_capital
        holdings = {ticker: 0 for ticker in signals.columns}
    else:
        prev_cash = portfolio.loc[prev_filled_idx, 'cash']
        cash = float(prev_cash) if pd.notna(prev_cash) else initial_capital
        prev_h = portfolio.loc[prev_filled_idx, 'holdings']
        if isinstance(prev_h, dict):
            holdings = dict(prev_h)
        else:
            holdings = {ticker: 0 for ticker in signals.columns}

    trades = {ticker: 0 for ticker in signals.columns}
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
                    px = nifty500.loc[i, ticker]
                    if not _valid_trade_price(px):
                        continue
                    # Sell the holding
                    sell_amount = holdings[ticker] * float(px)
                    cash += sell_amount - sell_amount * transaction_cost_rate - dp_sell_charge
                    trade_turnover += sell_amount
                    transaction_cost += sell_amount * transaction_cost_rate + dp_sell_charge
                    trades[ticker] = -holdings[ticker]
                    holdings[ticker] = 0
        if mreg == 'Down':
            # Sell all holdings
            for ticker, quantity in list(holdings.items()):
                if quantity > 0:
                    px = nifty500.loc[i, ticker]
                    if not _valid_trade_price(px):
                        continue
                    sell_amount = quantity * float(px)
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
                px_t = nifty500.loc[i, ticker]
                current_values[ticker] = holdings[ticker] * float(px_t)
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

    mv = holdings_market_value(i, holdings)
    portfolio_value = cash + mv

    if portfolio_value <= 0:
        raise ValueError(f"Portfolio value is less than 0 at {i}")

    if prev_filled_idx is None:
        prev_portfolio_value = initial_capital
    else:
        pv_prev = portfolio.loc[prev_filled_idx, 'portfolio_value']
        prev_portfolio_value = float(pv_prev) if pd.notna(pv_prev) else initial_capital

    daily_pnl = portfolio_value - prev_portfolio_value
    if pd.notna(prev_portfolio_value) and prev_portfolio_value != 0:
        daily_return = daily_pnl / prev_portfolio_value
    else:
        daily_return = np.nan

    num_open_positions = sum(1 for quantity in holdings.values() if quantity > 0)
    # Count of tickers with any fill this day; total share lots = sum(abs(v) for v in trades.values())
    num_trades = sum(1 for v in trades.values() if v != 0)

    portfolio.at[i, 'portfolio_value'] = portfolio_value
    portfolio.at[i, 'transaction_cost'] = transaction_cost
    portfolio.at[i, 'daily_return'] = daily_return
    portfolio.at[i, 'holdings'] = dict(holdings)
    portfolio.at[i, 'cash'] = float(cash)
    portfolio.at[i, 'trades'] = dict(trades)
    portfolio.at[i, 'daily_pnl'] = daily_pnl
    portfolio.at[i, 'num_open_positions'] = int(num_open_positions)
    portfolio.at[i, 'num_trades'] = int(num_trades)
    portfolio.at[i, 'trade_turnover'] = float(trade_turnover)
    portfolio.at[i, 'market_regime'] = mreg

print(portfolio.head())
print(portfolio.info())
portfolio.to_csv('backtest_results.csv')