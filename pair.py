'''
Pair module

Find pairs of stocks that are cointegrated.

Pair discovery:
- Calculate rolling correlation between all pairs of stocks
- Only include pairs with rolling correlation >= threshold
- Engle-Granger cointegration test
- Only include pairs with p-value < alpha

Correlation Window: 250 trading days
Correlation Threshold: 0.8
Engle-Granger Window: 500 trading days
Alpha: 0.05

Return:
- pairs dataframe
- index is the date
- column is pairs
- pairs column is a list of tuples of pairs

Saves the pairs to a csv file: pairs.csv.
'''

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

# Load nifty500.csv
nifty500 = pd.read_csv('nifty500.csv', index_col=0, parse_dates=[0])

# Calculate start dates for each ticker in nifty500
start_dates = {}
for ticker in nifty500.columns:
    start_dates[ticker] = nifty500[ticker].first_valid_index()

# Parameters
correlation_window = 250
correlation_threshold = 0.85
engle_granger_window = 500
alpha = 0.05

pairs = pd.DataFrame(columns=['pairs'])

lock_counter = 0

for i in nifty500.index:
    if lock_counter > 0:
        lock_counter -= 1
        # Add pairs from previous date to pairs dataframe current date
        pairs.loc[i, 'pairs'] = pairs.loc[i - 1, 'pairs'].copy()
        continue
    pos = nifty500.index.get_loc(i)
    if not isinstance(pos, int):
        continue
    # Check if there are enough trading rows for a 500-row window ending at the current date
    if pos < engle_granger_window - 1:
        continue
    # Get the 500 trading rows before the current date
    window = nifty500.iloc[pos - engle_granger_window + 1 : pos + 1]
    # Slice for valid tickers in window
    # A valid ticker is one that has start date before or equal to the 1st date in the window
    valid_tickers = [ticker for ticker in window.columns if start_dates[ticker] <= window.index[0]]
    window = window[valid_tickers]

    # Calculate sub window for correlation calculation
    # The sub window should be the last 250 trading days before the current date
    sub_window = window.iloc[-correlation_window:]
    # Slice for valid tickers in sub window
    # A valid ticker is one that has start date before or equal to the 1st date in the sub window
    valid_tickers = [ticker for ticker in sub_window.columns if start_dates[ticker] <= sub_window.index[0]]
    sub_window = sub_window[valid_tickers]

    # Calculate the rolling correlation between all pairs of stocks
    corr_mat = sub_window.corr()
    # Create a list of tuples of pairs with rolling correlation >= threshold
    candidate_pairs = [(ticker1, ticker2) for ticker1 in valid_tickers for ticker2 in valid_tickers if ticker1 < ticker2 and corr_mat.loc[ticker1, ticker2] >= correlation_threshold]

    valid_pairs = []
    # Engle-Granger cointegration test
    for pair in candidate_pairs:
        _, pvalue, _ = coint(window[pair[0]], window[pair[1]])
        if pvalue < alpha:
            valid_pairs.append(pair)
    
    if len(valid_pairs) > 0:
        lock_counter = 30
        
    print("Number of valid pairs: ", len(valid_pairs), "at date: ", i)
    pairs.loc[i, 'pairs'] = valid_pairs

# Save the pairs to a csv file
pairs.to_csv('pairs.csv')
print("Pair discovery complete")