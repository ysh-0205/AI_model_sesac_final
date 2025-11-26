# AI Core: 기상 조건 기반 사고 위험 예측 서버 (FastAPI)

이 프로젝트는  LightGBM  모델을 사용하여 실시간 기상 조건 및 시간대 데이터를 기반으로 사고의 심각도(Severity)를 예측하고, 이를  3단계 위험 티어 (3-Tier Risk Level) 로 해석하여 반환하는 FastAPI 서버입니다. 기상 정보와 시간대에 따라 사고의 심각도를 예측하여 위험 수준을 판단합니다.

## 🌟 주요 기능 및 특징

-  머신러닝 기반 예측 : `enhanced_accident_severity_pipeline.joblib` 모델을 사용하여 사고 발생 시 심각도 2 (Severity 2)와 심각도 3 (Severity 3이 발생할 확률을 예측합니다.
-  파생 변수 자동 생성 : 입력된 `Start_Time`을 기반으로 러시 아워(Is_Rush_Hour), 주말 여부(Is_Weekend), 빙판길 조건(Icy_Road) 등의 주요 피처를 자동으로 생성합니다.
-  3-Tier 위험 결정 로직 : 예측된 확률을 기반으로 안전 중심의 비즈니스 로직을 적용하여 최종 위험 티어(0, 1, 2)를 결정합니다.

## 📊 3-Tier 위험 결정 로직 (Safety-First)

모델이 반환하는 심각도 3 확률(P_Severity_3)과 심각도 2 확률(P_Severity_2)을 기반으로 다음의 기준에 따라 3단계 위험 티어를 결정합니다.

| 위험 티어 | 이름              | 결정 기준                                                           | 해석                                                                   |
|----------|------------------|--------------------------------------------------------------------|------------------------------------------------------------------------|
|  Tier 2  |  High Risk  (즉각 조치) | `P(Sev 3) >= 0.25` (T_HIGH)                                          | 심각한 사고(Sev 3) 발생 확률이 높으므로, 즉각적인 경고 및 조치가 필요합니다. |
|  Tier 1  |  Observation  (관찰 필요) | `Tier 2` 또는 `Tier 0`에 해당하지 않는 모든 경우                       | 'Gray Zone'에 해당하거나, 심각도 3 확률은 낮지만 안전 확신(P(Sev 2))이 부족한 경우입니다. 지속적인 관찰이 필요합니다. |
|  Tier 0  |  Low Risk  (안전 확신) | `P(Sev 3) < 0.10` (T_LOW) AND `P(Sev 2) >= 0.90` (N_SAFETY)           | 심각도 3 확률이 낮고, 동시에 낮은 심각도 2 확률에 대한 확신이 매우 높을 때만 부여되는 가장 안전한 상태입니다. |

## 🛠️ API 엔드포인트 및 응답 형식

### `POST /predict`

이 엔드포인트는 예측을 수행하고, 아래 4가지 정보를 포함하는 단일 JSON 객체를 반환합니다.


요청 본문 (예시)
```json
{
  "Start_Time": "2023-11-25 08:30:00",
  "Visibility_mi": 5.0,
  "Wind_Speed_mph": 10.0,
  "Precipitation_in": 0.1,
  "Temperature_F": 35.0,
  "Wind_Chill_F": 30.0,
  "Humidity_percent": 75.0,
  "Pressure_in": 30.0
}
```
응답 본문 (예시)
```json
{
  "P_Severity_2": 0.9512,
  "P_Severity_3": 0.0488,
  "predicted_risk_tier": 1,
  "tier_interpretation": "Tier 1: Observation/Medium Risk (Gray Zone or Low Confidence in Safety - Needs Review)"
}
```
