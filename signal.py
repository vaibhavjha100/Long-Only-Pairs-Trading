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
'''

import pandas as pd

# Load spread.csv
spread = pd.read_csv('spread.csv', index_col=0, parse_dates=[0])

# Load nifty 500 list
nifty500 = pd.read_csv('ind_nifty500list.csv')

all_tickers = nifty500['Symbol'].tolist()

# Add '.NS' to the end of each ticker
all_tickers = [ticker + '.NS' for ticker in all_tickers]

# Create a dataframe to store the signals
signals = pd.DataFrame(columns=all_tickers, dtype=int)

for i in spread.index:
    for c in spread.columns:
        left, right = c.split('|')
        if left not in all_tickers or right not in all_tickers:
            continue
        if pd.isna(spread.loc[i, c]):
            continue
        if spread.loc[i, c] > 2:
            signals.loc[i, left] += 1
            signals.loc[i, right] -= 1
        elif spread.loc[i, c] < -2:
            signals.loc[i, left] -= 1
            signals.loc[i, right] += 1
        
print(signals.head())
print(signals.info())
print(signals.describe())



