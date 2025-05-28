import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import os

def main():
    # === Setup ===
    DATA_PATH = 'C:/Users/kylem/OneDrive/Desktop/crypto_ai/RSI_MES_2_MIN_LOAD_MACD1250.csv'
    EXPORT_DIR = 'C:/Users/kylem/OneDrive/Desktop/crypto_ai/'
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # === STEP 1: Load & Clean Data ===
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    df.to_csv(os.path.join(EXPORT_DIR, 'step_1_loaded_cleaned.csv'), index=False)
    print("✅ Step 1 complete: Data loaded and cleaned")

    # === STEP 2: Detect MACD Crossovers ===
    def detect_macd_crossovers(df):
        crossover = (
            ((df['MACD'].shift(1) < df['Signal'].shift(1)) & (df['MACD'] > df['Signal'])) |
            ((df['MACD'].shift(1) > df['Signal'].shift(1)) & (df['MACD'] < df['Signal']))
        )
        df['MACD_Crossover'] = crossover.astype(int)
        return df

    df = detect_macd_crossovers(df)
    df.to_csv(os.path.join(EXPORT_DIR, 'step_2_macd_crossovers.csv'), index=False)
    print("✅ Step 2 complete: MACD crossovers detected")

    # === STEP 3: Label Each Crossover (3-point move logic) ===
    MIN_MOVE = 2.5
    def label_macd_signal(idx):
        if idx + 5 >= len(df) or idx - 1 < 0:
            return np.nan
        entry = df.loc[idx, 'close']
        prev_macd = df.loc[idx - 1, 'MACD']
        prev_signal = df.loc[idx - 1, 'Signal']
        curr_macd = df.loc[idx, 'MACD']
        curr_signal = df.loc[idx, 'Signal']
        future_closes = df.loc[idx + 1:idx + 5, 'close']

        if prev_macd < prev_signal and curr_macd > curr_signal:
            if any(close - entry >= MIN_MOVE for close in future_closes):
                return 1
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            if any(entry - close >= MIN_MOVE for close in future_closes):
                return -1
        return np.nan

    df['label'] = df.index.map(lambda idx: label_macd_signal(idx) if df.loc[idx, 'MACD_Crossover'] == 1 else np.nan)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    df.to_csv(os.path.join(EXPORT_DIR, 'step_3_macd_labeled.csv'), index=False)
    print("✅ Step 3 complete: Labels assigned to crossovers")

    # === STEP 4: Prepare tsfresh Input Format ===
    FEATURE_COLS = ['open', 'high', 'low', 'close', 'VWAP', 'Volume', 'RSI', 'RSI-based_MA', 'Histogram', 'MACD', 'Signal', 'ATR']
    df['ts_id'] = df.index.astype(str)

    melted = pd.melt(
        df[['ts_id', 'time'] + FEATURE_COLS],
        id_vars=['ts_id', 'time'],
        value_vars=FEATURE_COLS,
        var_name='feature',
        value_name='value'
    )

    melted.rename(columns={'ts_id': 'id'}, inplace=True)
    tsfresh_input = melted[['id', 'time', 'feature', 'value']]
    tsfresh_input.to_csv(os.path.join(EXPORT_DIR, 'step_4_tsfresh_input.csv'), index=False)
    print("✅ Step 4 complete: tsfresh input prepared")

    # === STEP 5: Feature Extraction ===
    X_extracted = extract_features(
        tsfresh_input,
        column_id='id',
        column_sort='time',
        column_kind='feature',
        column_value='value',
        default_fc_parameters=ComprehensiveFCParameters(),
        n_jobs=0
    )
    X_extracted.to_csv(os.path.join(EXPORT_DIR, 'step_5_features_extracted.csv'))
    print("✅ Step 5 complete: Features extracted")

    # === STEP 6: Group & Align with Labels ===
    X_grouped = X_extracted.copy()
    X_grouped.index = X_grouped.index.astype(int)
    df.set_index(df.index.astype(int), inplace=True)
#y = df.loc[X_grouped.index, 'label'].astype(int)
    #new one line below.

    y = df.loc[X_grouped.index, 'label'].map({-1: 0, 1: 1}).astype(int)

    X_grouped.to_csv(os.path.join(EXPORT_DIR, 'step_6_features_grouped.csv'))
    y.to_csv(os.path.join(EXPORT_DIR, 'step_6_labels.csv'))
    print("✅ Step 6 complete: Features grouped and labels aligned")

    # === STEP 7: Impute & Select Features ===
    X_grouped = X_grouped.select_dtypes(include=[np.number])
    impute(X_grouped)
    X_grouped.to_csv(os.path.join(EXPORT_DIR, 'step_7_features_imputed.csv'))

    if y.nunique() < 2:
        X_selected = X_grouped.copy()
    else:
        X_selected = select_features(X_grouped, y)

    X_selected.to_csv(os.path.join(EXPORT_DIR, 'step_8_features_selected.csv'))
    print("✅ Step 7 complete: Features imputed and selected")

    # === STEP 8: Train/Test Split & Model Training ===
 #   X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=42)

    X_train = X_selected.iloc[:int(len(X_selected) * 0.75)]
    X_test = X_selected.iloc[int(len(X_selected) * 0.75):]
    y_train = y.iloc[:int(len(y) * 0.75)]
    y_test = y.iloc[int(len(y) * 0.75):]

  #  model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',  # required for newer versions
        device='cuda:0'  # tells XGBoost to use your first GPU
    )

    print(f"✅ Number of features passed to the model: {X_selected.shape[1]}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ✅ Save the trained model
    model.save_model("xgb_FE_model.json")

    print("✅ Step 8 complete: Model trained and evaluated")

    # Add predicted labels to test set
    results_df = X_test.copy()
    results_df['true_label'] = y_test.values
    results_df['predicted_label'] = y_pred

    # Optional: Add predicted probabilities (e.g., probability of class 1)
    y_proba = model.predict_proba(X_test)
    results_df['prob_class_1'] = y_proba[:, 1]  # probability of label 1

    # Merge back time and price from original df
    original_subset = df.loc[results_df.index, ['time', 'close']]
    results_df = pd.concat([original_subset.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # Save
    results_df.to_csv(os.path.join(EXPORT_DIR, 'step_9_test_results_with_time.csv'), index=False)
    print("📤 Test results (with time & price) exported to step_9_test_results_with_time.csv")

    # === Output Classification Report ===
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))


# === Entry Point ===
if __name__ == "__main__":
    main()
