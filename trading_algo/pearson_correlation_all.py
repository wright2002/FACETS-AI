# Kyle Wright

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 0)

# === Load and preprocess the dataset ===
df = pd.read_csv('RSI_MES_2_MIN_LOAD_MACD1250.csv', parse_dates=['time'])
df.columns = df.columns.str.strip().str.replace(" ", "_")
df = df.sort_values('time').reset_index(drop=True)

# === Add MACD-Signal Difference ===
df['MACD_Signal_Diff'] = df['MACD'] - df['Signal']

# === Track steps since last crossover ===
steps_since = []
count = None
for i in range(len(df)):
    if i == 0 or (
        (df['MACD'].iloc[i - 1] < df['Signal'].iloc[i - 1] and df['MACD'].iloc[i] > df['Signal'].iloc[i]) or
        (df['MACD'].iloc[i - 1] > df['Signal'].iloc[i - 1] and df['MACD'].iloc[i] < df['Signal'].iloc[i])
    ):
        count = 0
    elif count is not None:
        count += 1
    else:
        count = np.nan
    steps_since.append(count)

df['Steps_Since_Crossover'] = steps_since

def label_future_movement(idx):
    if idx + 5 >= len(df):
        return np.nan
    entry = df.loc[idx, 'close']
    future_closes = df.loc[idx + 1:idx + 5, 'close']
    return max(future_closes) - entry

df['Future_Movement'] = df.index.map(label_future_movement)

# === Define features ===
features = ['open', 'high', 'low', 'close', 'VWAP', 'Upper_Band_#1',
       'Lower_Band_#1', 'Upper_Band_#2', 'Lower_Band_#2', 'Upper_Band_#3',
       'Lower_Band_#3', 'Volume', 'RSI', 'RSI-based_MA', 'Histogram', 'MACD',
       'Signal', 'ATR', 'MACD_Signal_Diff', 'Steps_Since_Crossover', 'Future_Movement']
X = df[features]

# === Compute and export Pearson correlation matrix ===
correlation_matrix = X.corr(method='pearson')
correlation_matrix.to_csv("correlation_matrix.csv")
