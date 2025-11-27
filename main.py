import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import warnings
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os  # <-- 경로 디버깅을 위해 os 모듈 추가

# -----------------
# 1. Configuration (설정값 및 모델 로직)
# -----------------

# 모델 파일명
MODEL_FILENAME = 'enhanced_accident_severity_pipeline.joblib'

# 3-Tier Risk Mapping Thresholds
T_HIGH = 0.70  # Tier 2 (High) 결정 기준: P(Sev 3) >= T_HIGH (실제 경로 예측값을 기준으로 상향 조정)
T_LOW = 0.15  # Low Risk 영역 시작 기준: P(Sev 3) < T_LOW ('안전한 조건' 테스트 결과를 반영하여 상향)
N_SAFETY = 0.50  # Tier 0 (Low) 확정 기준: P(Sev 2) >= N_SAFETY (P(Sev3)이 낮아도 P(Sev2)가 높으면 안전으로 판단)

# 모델 학습 시 사용된 최종 10개 피처 리스트 (순서 중요!)
FEATURES = [
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Temperature(F)',
    'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Is_Rush_Hour', 'Is_Weekend', 'Icy_Road'
]


# -----------------
# 2. Pydantic 데이터 모델 (API 입력 데이터 형식 정의)
# -----------------
class FeatureData(BaseModel):
    # 날짜/시간 정보 (파생 변수 생성에 사용)
    Start_Time: str = Field(description="사고 시작 시간 (예: '2023-11-25 08:30:00').")

    # 원시 기상 조건 피처 7개
    Visibility_mi: float = Field(description="가시성 (마일).")
    Wind_Speed_mph: float = Field(description="풍속 (mph).")
    Precipitation_in: float = Field(description="강수량 (인치).")
    Temperature_F: float = Field(description="온도 (화씨).")
    Wind_Chill_F: float = Field(description="체감 온도 (화씨).")
    Humidity_percent: float = Field(description="습도 (%).")
    Pressure_in: float = Field(description="기압 (인치).")


# -----------------
# 3. FastAPI 및 Model Loading
# -----------------

app = FastAPI(
    title="AI Core Accident Risk Prediction Server",
    description="FastAPI server for 3-Tier accident severity risk prediction (Safety-focused)."
)
MODEL_PIPELINE = None


@app.on_event("startup")
def load_model():
    """서버 시작 시점에 모델을 한 번만 로드하여 메모리에 유지합니다."""
    global MODEL_PIPELINE

    # --- 🔍 모델 로드 디버깅을 위한 시각적 정보 출력 ---
    current_cwd = os.getcwd()
    model_path = os.path.join(current_cwd, MODEL_FILENAME)
    print("--- 🔍 모델 로드 디버깅 정보 ---")
    print(f"현재 작업 디렉토리 (CWD): {current_cwd}")
    print(f"모델 파일명: {MODEL_FILENAME}")
    print(f"시도하는 전체 경로: {model_path}")
    print("------------------------------")
    # ---------------------------------------------

    try:
        # joblib.load는 상대 경로로 시도합니다.
        MODEL_PIPELINE = joblib.load(MODEL_FILENAME)
        print(f"✅ AI Core: '{MODEL_FILENAME}' 모델 파이프라인 로드 완료.")
    except FileNotFoundError:
        # 모델 파일 경로를 찾을 수 없을 때 발생하는 오류 처리
        print(f"❌ ERROR: 모델 파일 '{MODEL_FILENAME}'을 찾을 수 없습니다. 예측을 수행할 수 없습니다.")
        MODEL_PIPELINE = None
    except Exception as e:
        print(f"❌ ERROR: 모델 로드 중 오류 발생: {e}")
        MODEL_PIPELINE = None


# --------------------------------------------------------------------------------------
# 4. Feature Engineering Helper (파생 변수 생성 함수)
# --------------------------------------------------------------------------------------

def engineer_features(raw_data: FeatureData) -> Dict[str, Any]:
    """
    원시 입력 데이터를 받아 파생 변수를 생성하고, 모델에 필요한 10가지 피처를
    정확한 순서로 담은 딕셔너리를 반환합니다.
    """
    # 1. Start_Time 파싱
    try:
        start_time = datetime.strptime(raw_data.Start_Time, '%Y-%m-%d %H:%M:%S')
        hour = start_time.hour
        weekday = start_time.weekday()  # 월요일=0, 일요일=6
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Start_Time format. Use 'YYYY-MM-DD HH:MM:SS'.")

    # 2. 파생 변수 계산 (분석 코드와 동일)
    # Rush Hour: 06~09, 16~19
    is_rush_hour = 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0

    # Weekend: 토(5), 일(6)
    is_weekend = 1 if weekday >= 5 else 0

    # Icy Road: Temp <= 32F AND Precip > 0 (결측치 처리 없이 입력값을 그대로 사용)
    temp_f = raw_data.Temperature_F
    precip_in = raw_data.Precipitation_in
    icy_road = 1 if (temp_f <= 32) and (precip_in > 0) else 0

    # 3. 10개 피처 딕셔너리 생성 (FEATURES 리스트 순서와 일치하도록 보장)
    # Pydantic 필드 이름과 FEATURES 리스트의 이름 매핑
    feature_dict = {
        'Visibility(mi)': raw_data.Visibility_mi,
        'Wind_Speed(mph)': raw_data.Wind_Speed_mph,
        'Precipitation(in)': raw_data.Precipitation_in,
        'Temperature(F)': raw_data.Temperature_F,
        'Wind_Chill(F)': raw_data.Wind_Chill_F,
        'Humidity(%)': raw_data.Humidity_percent,
        'Pressure(in)': raw_data.Pressure_in,
        'Is_Rush_Hour': float(is_rush_hour),
        'Is_Weekend': float(is_weekend),
        'Icy_Road': float(icy_road)
    }

    # FEATURES 리스트 순서대로 최종 값 리스트 생성
    final_input_values = [feature_dict[key] for key in FEATURES]

    return final_input_values


# --------------------------------------------------------------------------------------
# 5. API Endpoint
# --------------------------------------------------------------------------------------

@app.post("/predict", summary="Accident Risk Tier Prediction", response_model=Dict[str, Any])
def predict_risk(data: FeatureData):
    """
    입력 피처와 시간을 받아 AI 모델 예측을 수행하고 3-Tier 위험도를 반환합니다.
    Tier 0 (Low)는 가장 안전하다고 확신할 때만 부여됩니다.
    """
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Prediction unavailable.")

    # 1. 파생 변수 생성 및 10개 피처 값 준비
    try:
        final_input_values = engineer_features(data)
    except HTTPException as e:
        return {'error': e.detail}

    # 2. 모델 입력 형식으로 변환 (Pandas DataFrame)
    # 10개 피처의 순서와 이름이 모델 학습 시와 일치해야 합니다.
    X_input = pd.DataFrame([final_input_values], columns=FEATURES)

    # 3. 모델 확률 예측
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning
        )
        try:
            # 예측 수행: [ [P(Sev 2), P(Sev 3)] ] 형태의 결과를 반환
            y_proba = MODEL_PIPELINE.predict_proba(X_input)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed inside model: {e}")

    p_sev2 = y_proba[0]  # Severity 2 확률
    p_sev3 = y_proba[1]  # Severity 3 확률

    # --- 💡 리턴 전 중간값/최종값 터미널 출력 (디버깅용) ---
    print("\n[AI CORE LOG: 예측 전송 데이터 확인]")
    print(f"1. 최종 입력 피처 (10개): {X_input.values.tolist()[0]}")
    print(f"2. Raw 예측 확률: P(Sev 2)={p_sev2:.4f}, P(Sev 3)={p_sev3:.4f}")
    print(f"3. 적용된 임계값: T_HIGH={T_HIGH}, T_LOW={T_LOW}, N_SAFETY={N_SAFETY}")

    # 4. 3-Tier Risk Level 로직 적용 (안전 최우선 로직)
    tier_reason = ""
    if p_sev3 >= T_HIGH:
        predicted_risk_tier = 2
        tier_reason = f"P(Sev 3)({p_sev3:.4f}) >= T_HIGH({T_HIGH})"
    elif p_sev3 < T_LOW:
        if p_sev2 >= N_SAFETY:
            predicted_risk_tier = 0
            tier_reason = f"P(Sev 3)({p_sev3:.4f}) < T_LOW({T_LOW}) and P(Sev 2)({p_sev2:.4f}) >= N_SAFETY({N_SAFETY})"
        else:
            predicted_risk_tier = 1
            tier_reason = f"P(Sev 3)({p_sev3:.4f}) < T_LOW({T_LOW}) but P(Sev 2)({p_sev2:.4f}) < N_SAFETY({N_SAFETY})"
    else: # T_LOW <= p_sev3 < T_HIGH
        predicted_risk_tier = 1
        tier_reason = f"T_LOW({T_LOW}) <= P(Sev 3)({p_sev3:.4f}) < T_HIGH({T_HIGH})"


    print(f"4. Tier 결정 로직: {tier_reason}")
    print(f"5. 할당된 최종 Risk Tier: {predicted_risk_tier}")
    print("[AI CORE LOG: 데이터 확인 완료]")
    # -------------------------------------------------------------

    # 5. 결과 반환
    return {
        'P_Severity_2': float(p_sev2),
        'P_Severity_3': float(p_sev3),
        'predicted_risk_tier': int(predicted_risk_tier),
        'tier_interpretation': {
            0: "Tier 0: Low Risk (Extreme Confidence in Safety - No Action)",
            1: "Tier 1: Observation/Medium Risk (Gray Zone or Low Confidence in Safety - Needs Review)",
            2: "Tier 2: High Risk (Immediate Action Required)"
        }[predicted_risk_tier]
    }