'''
Preprocessing module

Preprocess the data collected from the collection module.
- Sort the data by date
- FFill missing values with the previous day's value
- Start nifty500 data 500 trading days before the start of nifty50 data


Saves the data to csv files: nifty500.csv and nifty50.csv.
'''

import pandas as pd

# Load nifty500.csv
nifty500 = pd.read_csv('nifty500.csv', index_col=0, parse_dates=[0])

# Load nifty50.csv
nifty50 = pd.read_csv('nifty50.csv', index_col=0, parse_dates=[0])

if nifty500.empty:
    raise ValueError('nifty500.csv is empty.')
if nifty50.empty:
    raise ValueError('nifty50.csv is empty.')

# Sort the data by date
nifty500 = nifty500.sort_index()
nifty50 = nifty50.sort_index()

_dup500 = nifty500.index.duplicated()
if _dup500.any():
    raise ValueError(
        f'nifty500.csv has {int(_dup500.sum())} duplicate index label(s); fix dates before preprocessing.'
    )
_dup50 = nifty50.index.duplicated()
if _dup50.any():
    raise ValueError(
        f'nifty50.csv has {int(_dup50.sum())} duplicate index label(s); fix dates before preprocessing.'
    )

# FFill missing values with the previous day's value
nifty500 = nifty500.ffill()
nifty50 = nifty50.ffill()

# Start nifty500 data 500 trading days before the start of nifty50 data
# Integer position of nifty50's first date in nifty500's index; slice 500 rows before it
# Dont use timedelta, use index position
first_nifty50_date = nifty50.index[0]
align_pos = nifty500.index.get_indexer([first_nifty50_date])[0]
if align_pos == -1:
    raise ValueError(
        f"nifty500 has no row for nifty50 start date {first_nifty50_date}"
    )
if align_pos == -2:
    raise ValueError(
        f"ambiguous alignment: duplicate nifty500 rows for nifty50 start date {first_nifty50_date}"
    )
start_pos = max(0, align_pos - 500)
nifty500 = nifty500.iloc[start_pos:]

# Save the data to csv files
nifty500.to_csv('nifty500.csv')
nifty50.to_csv('nifty50.csv')

print("Preprocessing complete")