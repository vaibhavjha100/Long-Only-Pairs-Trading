'''
Signal module

Generate signals for each pair based on the z-scores of the spread.

Buy signal hits:
For a pair (A,B):
- When z score of spread is greater than 2, add 1 to A and subtract 1 from B
- When z score of spread is less than -2, subtract 1 from A and add 1 to B
- When z score of spread is between -2 and 2, do nothing

Returns:
- signals dataframe
- index is the date
- column are the tickers of the pairs
- column values buy signal hits

Run: python generate_signals.py
(Do not name this file signal.py — it shadows the stdlib signal module and breaks pandas.)
'''

import pandas as pd
import numpy as np

for _spread_path in ('spread_z_scores.csv', 'spread.csv'):
    try:
        spread = pd.read_csv(_spread_path, index_col=0, parse_dates=[0])
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError('Expected spread_z_scores.csv or spread.csv')

# Load nifty 500 list
nifty500 = pd.read_csv('ind_nifty500list.csv')

all_tickers = nifty500['Symbol'].tolist()

# Add '.NS' to the end of each ticker
all_tickers = [ticker + '.NS' for ticker in all_tickers]

# Create a dataframe to store the signals (zeros so += works)
signals = pd.DataFrame(0, index=spread.index, columns=all_tickers, dtype=np.int64)

for c in spread.columns:
    parts = c.split('|')
    if len(parts) != 2:
        continue
    left, right = parts
    if left not in all_tickers or right not in all_tickers:
        continue
    z = spread[c]
    hi = z > 2
    lo = z < -2
    signals[left] = signals[left] + hi.astype(np.int64) - lo.astype(np.int64)
    signals[right] = signals[right] - hi.astype(np.int64) + lo.astype(np.int64)

print(signals.head())
print(signals.info())
print(signals.describe())

signals.to_csv('signals.csv')
