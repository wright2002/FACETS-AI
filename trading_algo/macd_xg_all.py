# Kyle Wright
# Last Mod: 5/27/25

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

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

# === Label: price movement within 5 bars ===
MIN_MOVE = 2.5
def label_future_movement(idx):
    if idx + 5 >= len(df):
        return np.nan
    entry = df.loc[idx, 'close']
    future_closes = df.loc[idx + 1:idx + 5, 'close']
    return int(any(abs(close - entry) >= MIN_MOVE for close in future_closes))

df['label'] = df.index.map(label_future_movement)
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# === Define features ===
features = ['open', 'high', 'low', 'close', 'VWAP', 'Upper_Band_#1',
       'Lower_Band_#1', 'Upper_Band_#2', 'Lower_Band_#2', 'Upper_Band_#3',
       'Lower_Band_#3', 'Volume', 'RSI', 'RSI-based_MA', 'Histogram', 'MACD',
       'Signal', 'ATR', 'MACD_Signal_Diff', 'Steps_Since_Crossover']
X = df[features]
y = df['label']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# === Train model ===
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === SHAP Analysis ===
print("🔍 Generating SHAP summary...")
shap.initjs()
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)
