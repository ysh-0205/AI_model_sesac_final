import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import gc

# Try importing tabulate for nice table formatting
try:
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# --- Configuration --- #
DATA_PATH = 'accidents.csv'
RANDOM_STATE = 42
MODEL_FILENAME = 'enhanced_accident_severity_pipeline.joblib'

# ------------------------------------------------------------------------
# 💡 User-defined thresholds for the 4-tier risk mapping
# P(Sev 3) 기준 임계값
T_HIGH = 0.25
T_LOW = 0.10
# P(Sev 2) 기준 임계값 (Tier 0 판단을 위한 추가적인 안전 확신도)
N_SAFETY = 0.90
# ------------------------------------------------------------------------

SAMPLES_PER_SEVERITY = 500

FEATURES = [
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Temperature(F)',
    'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Is_Rush_Hour', 'Is_Weekend', 'Icy_Road'
]


# --- Helper Function: Load Data, Feature Engineering, Sampling ---
def load_and_sample_data(samples_per_severity):
    """
    Loads data, creates 3 derived features (Rush Hour, Weekend, Icy Road),
    merges severity levels, and samples data.
    """
    # ... (데이터 로딩 및 전처리 함수 내용은 이전과 동일) ...
    print(f"Loading data from {DATA_PATH} and engineering features...")
    try:
        df = pd.read_csv(DATA_PATH)

        # 💡 [파생 변수 생성 로직 추가] (학습 코드와 동일해야 함)
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['Hour'] = df['Start_Time'].dt.hour
        df['Weekday'] = df['Start_Time'].dt.weekday

        # Rush Hour: 06~09, 16~19
        df['Is_Rush_Hour'] = df['Hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)

        # Weekend: 토(5), 일(6)
        df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

        # Icy Road: Temp <= 32F AND Precip > 0
        temp = df['Temperature(F)'].fillna(50)
        precip = df['Precipitation(in)'].fillna(0)
        df['Icy_Road'] = ((temp <= 32) & (precip > 0)).astype(int)

        required_cols = FEATURES + ['Severity']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        df_clean = df[required_cols].dropna().copy()
        df_clean['Severity'] = df_clean['Severity'].replace({1: 2, 4: 3})

        sampled_rows = []
        target_severities = [2, 3]

        for sev in target_severities:
            sev_df = df_clean[df_clean['Severity'] == sev]
            n_sample = min(len(sev_df), samples_per_severity)
            if n_sample > 0:
                sampled_rows.append(sev_df.sample(n=n_sample, random_state=RANDOM_STATE))
            else:
                print(f"Warning: Not enough data for Severity {sev}. Found {len(sev_df)} rows.")

        if not sampled_rows:
            raise ValueError("No data rows could be sampled.")

        sample_df = pd.concat(sampled_rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        total_samples = len(sample_df)
        print(f"Successfully sampled {total_samples} rows (Targeting {samples_per_severity} per Severity).")

        X_test = sample_df[FEATURES]
        y_test_true = sample_df['Severity'].values

        del df, df_clean, sampled_rows
        gc.collect()

        return X_test, y_test_true

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Data file '{DATA_PATH}' not found.")
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        raise


def analyze_false_negatives(df: pd.DataFrame, t_high: float, t_low: float, n_safety: float):
    """
    Analyzes False Negative cases where Actual Severity 3 was underestimated.
    Checks for Tier 0 (Critical Miss), Tier 1 (Moderate Miss), and Tier 2 (Buffer Zone).
    """
    print("\n\n--- 7. Targeted False Negative Analysis (Actual Severity 3) ---")
    print("Finding cases where Actual Severity 3 was underestimated.")
    print(f"(T_HIGH={t_high:.2f}, T_LOW={t_low:.2f}, N_SAFETY={n_safety:.2f})")

    # 1. Actual Severity 3 -> Predicted Risk Tier 0 (Critical Miss)
    sev3_to_tier0 = df[
        (df['Actual Severity(2/3)'] == 3) &
        (df['Predicted Risk Level'] == 0)
        ].reset_index(drop=True)

    print(f"\n[Case A: Actual 3 -> Predicted Tier 0 (LOWEST RISK)] - Count: {len(sev3_to_tier0)}")
    if len(sev3_to_tier0) > 0:
        print(f"**CRITICAL MISS:** Model predicted LOWEST Risk (Tier 0) for a HIGH Severity (3) case.")
        if TABULATE_AVAILABLE:
            print(tabulate(sev3_to_tier0.head(20), headers='keys', tablefmt='pipe', showindex=True))
            print(f"... (Total {len(sev3_to_tier0)} rows) ...")
        else:
            print(sev3_to_tier0.head(20).to_string(index=True))
    else:
        print("=> **SUCCESS**: No Actual Severity 3 cases were rated as Tier 0.")

    # 2. Actual Severity 3 -> Predicted Risk Tier 1 (Moderate Miss/Watchlist)
    sev3_to_tier1 = df[
        (df['Actual Severity(2/3)'] == 3) &
        (df['Predicted Risk Level'] == 1)
        ].reset_index(drop=True)

    print(f"\n[Case B: Actual 3 -> Predicted Tier 1 (MILD RISK)] - Count: {len(sev3_to_tier1)}")
    if len(sev3_to_tier1) > 0:
        print(f"**MODERATE MISS:** High Severity (3) cases classified as Mild Risk (Tier 1). (Watchlist for Re-review)")
        if TABULATE_AVAILABLE:
            print(tabulate(sev3_to_tier1.head(20), headers='keys', tablefmt='pipe', showindex=True))
            print(f"... (Total {len(sev3_to_tier1)} rows) ...")
        else:
            print(sev3_to_tier1.head(20).to_string(index=True))
    else:
        print("=> **SUCCESS**: No Actual Severity 3 cases were rated as Tier 1.")

    # 3. Actual Severity 3 -> Predicted Risk Tier 2 (Buffer Zone)
    sev3_to_tier2 = df[
        (df['Actual Severity(2/3)'] == 3) &
        (df['Predicted Risk Level'] == 2)
        ].reset_index(drop=True)

    print(f"\n[Case C: Actual 3 -> Predicted Tier 2 (MEDIUM RISK)] - Count: {len(sev3_to_tier2)}")
    if len(sev3_to_tier2) > 0:
        print(f"**BUFFER SALVAGE:** High Severity (3) cases successfully salvaged to Tier 2 (Medium Risk).")
        if TABULATE_AVAILABLE:
            print(tabulate(sev3_to_tier2.head(20), headers='keys', tablefmt='pipe', showindex=True))
            print(f"... (Total {len(sev3_to_tier2)} rows) ...")
        else:
            print(sev3_to_tier2.head(20).to_string(index=True))
    else:
        print("=> **ERROR**: No Actual Severity 3 cases were rated as Tier 2. Check T_LOW/T_HIGH boundary.")

    print("\n--- End of Targeted Analysis ---")


def evaluate_saved_model():
    print("--- 1. Loading Model and Data ---")

    global T_HIGH, T_LOW, N_SAFETY

    try:
        pipeline = joblib.load(MODEL_FILENAME)
        print(f"Pipeline '{MODEL_FILENAME}' successfully loaded.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
        return

    try:
        X_test, y_test_true = load_and_sample_data(samples_per_severity=SAMPLES_PER_SEVERITY)
    except Exception as e:
        print(e)
        return

    print(f"X_test shape: {X_test.shape}")

    # --- 3. Prediction & 4-Tier Risk Mapping ---
    print("\n--- 3. Prediction & 4-Tier Risk Mapping ---")

    # [:, 0] = P(Sev 2), [:, 1] = P(Sev 3)
    y_proba = pipeline.predict_proba(X_test)
    y_proba_sev2 = y_proba[:, 0]
    y_proba_sev3 = y_proba[:, 1]

    print(f"Applying new 4-Tier logic (T_HIGH={T_HIGH:.2f}, T_LOW={T_LOW:.2f}, N_SAFETY={N_SAFETY:.2f})")

    # 💡 4-Tier 로직 구현
    y_risk_pred = np.zeros(len(X_test), dtype=int)

    # Tier 3 (High) - P(Sev 3) >= T_HIGH
    condition_for_tier3 = (y_proba_sev3 >= T_HIGH)
    y_risk_pred[condition_for_tier3] = 3

    # Tier 0 (Low) - P(Sev 3) < T_LOW AND P(Sev 2) >= N_SAFETY
    condition_for_tier0 = (y_proba_sev3 < T_LOW) & (y_proba_sev2 >= N_SAFETY)
    y_risk_pred[condition_for_tier0] = 0

    # Tier 1 (Mild) - P(Sev 3) < T_LOW AND P(Sev 2) < N_SAFETY (남은 P(Sev 3) < T_LOW 케이스)
    condition_for_tier1 = (y_proba_sev3 < T_LOW) & (y_proba_sev2 < N_SAFETY)
    y_risk_pred[condition_for_tier1] = 1

    # Tier 2 (Medium) - T_LOW <= P(Sev 3) < T_HIGH (Gray Zone)
    condition_for_tier2 = (y_proba_sev3 >= T_LOW) & (y_proba_sev3 < T_HIGH)
    y_risk_pred[condition_for_tier2] = 2

    # --- 5. Performance (2-Tier Basis) ---
    print(f"\n--- 5. 2-Tier Performance Report (Threshold {T_HIGH}) ---")
    y_pred_sev23 = np.where(y_proba_sev3 >= T_HIGH, 3, 2)

    print(classification_report(y_test_true, y_pred_sev23, digits=4, target_names=['Severity 2', 'Severity 3']))
    print(f"Weighted F1: {f1_score(y_test_true, y_pred_sev23, average='weighted'):.4f}")
    print("-----------------------------------------------------------------")

    # --- 6. Table & Analysis ---
    print(f"\n--- 6. Risk Level Analysis ({len(y_test_true)} samples) ---")

    risk_map = {0: 'Tier 0 (Lowest)', 1: 'Tier 1 (Mild)', 2: 'Tier 2 (Medium)', 3: 'Tier 3 (High)'}

    evaluation_result = pd.DataFrame({
        'Actual Severity(2/3)': y_test_true,
        'P(Sev 2)': y_proba_sev2.round(4),
        'P(Sev 3)': y_proba_sev3.round(4),
        'Predicted Risk Level': y_risk_pred,
        'Risk Level Interpretation': np.vectorize(risk_map.get)(y_risk_pred)
    })

    if TABULATE_AVAILABLE:
        # Show first 20 rows for brevity
        print(tabulate(evaluation_result.head(20), headers='keys', tablefmt='pipe', showindex=False))
        print(f"... (Total {len(evaluation_result)} rows) ...")
    else:
        print(evaluation_result.head(20).to_string(index=False))

    analyze_false_negatives(evaluation_result, T_HIGH, T_LOW, N_SAFETY)


if __name__ == "__main__":
    evaluate_saved_model()