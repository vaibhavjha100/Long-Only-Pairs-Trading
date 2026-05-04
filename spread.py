'''
Spread module

- Calculate daily hedge ratios for each pair (250 trading-day window: cov(y,x)/var(x))
- Calculate daily spreads for each pair (y - beta*x using that hedge ratio)
- Calculate daily z-scores of the spread (60 trading-day rolling mean and std)

Returns:
- z scores dataframe (also written to spread.csv)
- index is the date (aligned to nifty500)
- columns are all pairs that ever appear in pairs.csv, named "TICKER_A|TICKER_B"
- column values are the z scores; NaN when the pair is not in the cointegrated list that day
'''

import ast
import pandas as pd
import numpy as np

# Load pairs.csv
pairs = pd.read_csv('pairs.csv', index_col=0, parse_dates=[0])
pairs['pairs'] = pairs['pairs'].apply(ast.literal_eval)

# Load nifty500.csv
nifty500 = pd.read_csv('nifty500.csv', index_col=0, parse_dates=[0])
nifty500 = nifty500.sort_index()

hedge_window = 250
spread_z_window = 60


def pair_to_col(p):
    return p[0] + '|' + p[1]


all_pairs = set()
for row_pairs in pairs['pairs']:
    for pair in row_pairs:
        all_pairs.add(tuple(pair))

pair_cols = sorted(pair_to_col(p) for p in all_pairs)

# --- Rolling hedge ratio (250 trading days), spread, z (60 on spread), updated daily ---
hedge_ratios = pd.DataFrame(index=nifty500.index, columns=pair_cols, dtype=float)
spreads = pd.DataFrame(index=nifty500.index, columns=pair_cols, dtype=float)
z_full = pd.DataFrame(index=nifty500.index, columns=pair_cols, dtype=float)

for p in all_pairs:
    a, b = p[0], p[1]
    c = pair_to_col(p)
    y = nifty500[a]
    x = nifty500[b]
    x_var = x.rolling(hedge_window, min_periods=hedge_window).var()
    beta = y.rolling(hedge_window, min_periods=hedge_window).cov(x) / x_var
    hedge_ratios[c] = beta
    spr = y - beta * x
    spreads[c] = spr
    mu = spr.rolling(spread_z_window, min_periods=spread_z_window).mean()
    sig = spr.rolling(spread_z_window, min_periods=spread_z_window).std(ddof=1)
    z_full[c] = (spr - mu) / sig

# Pairs not in the cointegrated list for a date -> NaN for that date
active = pd.DataFrame(False, index=nifty500.index, columns=pair_cols)
for dt in pairs.index:
    if dt not in active.index:
        continue
    lst = pairs.loc[dt, 'pairs']
    if not isinstance(lst, list):
        continue
    for p in lst:
        c = pair_to_col(tuple(p))
        if c in active.columns:
            active.loc[dt, c] = True

z_scores = z_full.where(active)

# Save z-score panel (optional downstream use)
z_scores.to_csv('spread.csv')

print(z_scores.info())
print('Spread module complete')
