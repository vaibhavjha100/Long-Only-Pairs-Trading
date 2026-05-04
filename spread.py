'''
Spread module

- Calculate daily hedge ratios for each pair
- Calculate daily spreads for each pair
- Calculate daily z-scores for each pair

Returns:
- z scores dataframe
- index is the date
- columns are the pairs
- column values are the z scores
'''

import pandas as pd
import numpy as np
import pickle

# Load pairs.csv
pairs = pd.read_csv('pairs.csv', index_col=0, parse_dates=[0])

# Load nifty500.csv
nifty500 = pd.read_csv('nifty500.csv', index_col=0, parse_dates=[0])

