# Kyle Wright

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# === Define features ===
features = ['open', 'high', 'low', 'close', 'VWAP', 'Upper_Band_#1',
       'Lower_Band_#1', 'Upper_Band_#2', 'Lower_Band_#2', 'Upper_Band_#3',
       'Lower_Band_#3', 'Volume', 'RSI', 'RSI-based_MA', 'Histogram', 'MACD',
       'Signal', 'ATR', 'MACD_Signal_Diff', 'Steps_Since_Crossover']
X = df[features]

# === Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Run PCA ===
pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

# === Plot cumulative explained variance ===
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Optional: Display PCA loadings ===
pca_components = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)
print("\nTop PCA Loadings:")
print(pca_components.head(5))
pca_components.to_csv("pca_output.csv")
