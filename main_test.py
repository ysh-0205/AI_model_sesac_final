from fastapi.testclient import TestClient
from datetime import datetime
import os
import json
from typing import Dict, Any

# main.py에서 FastAPI 인스턴스, 모델 파일명, 그리고 load_model 함수를 가져옵니다.
# 💡 load_model 함수를 명시적으로 호출하여 테스트 환경에서 모델 로드를 강제합니다.
try:
    from main import app, MODEL_FILENAME, load_model
except ImportError:
    print("오류: main.py를 찾을 수 없습니다. main.py가 같은 디렉토리에 있는지 확인하십시오.")
    exit(1)

# -----------------------------------------------------------
# 💡 핵심 수정: 테스트를 시작하기 전에 모델 로딩 함수를 명시적으로 호출합니다.
print("--- 🛠️ 모델 로드 강제 시작 (TestClient 환경) ---")
load_model()
print("--- 🛠️ 모델 로드 시도 완료 ---")
# -----------------------------------------------------------


# FastAPI 테스트 클라이언트 초기화
client = TestClient(app)


def run_prediction_test(payload: Dict[str, Any], scenario_name: str):
    """
    주어진 페이로드를 사용하여 /predict 엔드포인트를 호출하고 결과를 출력합니다.
    """

    print(f"\n=======================================================")
    print(f"--- 예측 엔드포인트 테스트 시작: {scenario_name} ---")
    print(f"=======================================================")
    print(f"입력 페이로드: {json.dumps(payload, indent=2)}")

    # 엔드포인트 호출
    # TestClient를 사용하면 실제 네트워크 호출 없이 app 객체에 직접 요청합니다.
    response = client.post("/predict", json=payload)

    print("\n--- 응답 결과 ---")
    print(f"상태 코드 (Status Code): {response.status_code}")

    try:
        data = response.json()
        print("응답 JSON 본문:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        if response.status_code == 200:
            print("\n✅ 예측 성공: 모델이 로드되었으며, 유효한 결과를 반환했습니다.")
            # 예측 티어 및 해석 출력
            tier = data.get('predicted_risk_tier')
            interpretation = data.get('tier_interpretation')
            print(f"최종 예측 티어: Tier {tier} ({interpretation})")

        elif response.status_code == 503:
            print(f"\n❌ 오류: 503 Service Unavailable")
            print(f"이 오류는 '{MODEL_FILENAME}' 모델 파일이 현재 위치에 없어서 예측 파이프라인이 로드되지 않았기 때문에 발생합니다.")
        elif response.status_code == 422:
            print("\n❌ 오류: 422 Unprocessable Entity")
            print("Pydantic 유효성 검사 실패. 입력 페이로드에 필수 필드가 누락되었는지 확인하세요.")
        else:
            print(f"\n❓ 알 수 없는 오류 상태 코드: {response.status_code}")

    except json.JSONDecodeError:
        print("오류: 서버로부터 유효한 JSON 응답을 받지 못했습니다.")
        print(f"응답 텍스트: {response.text}")


def run_all_prediction_examples():
    """모든 정의된 시나리오에 대해 예측 테스트를 실행합니다."""

    # 1. 기본 시나리오: 맑고 따뜻한 조건 (Tier 0 또는 Tier 1 예상)
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload_safe = {
        "Start_Time": current_time_str,
        "Visibility_mi": 10.0,
        "Wind_Speed_mph": 5.0,
        "Precipitation_in": 0.0,
        "Temperature_F": 65.0,
        "Wind_Chill_F": 65.0,
        "Humidity_percent": 60.0,
        "Pressure_in": 30.0
    }
    run_prediction_test(payload_safe, "1. 기본 시나리오 (현재 시각/맑음)")

    # 2. 위험 시나리오: 빙판길 및 러시 아워 (Tier 2 예상)
    # Start_Time의 날짜는 현재 날짜를 사용하고 시간만 07:30:00으로 설정
    rush_hour_time_str = datetime.now().strftime('%Y-%m-%d') + " 07:30:00"
    payload_risk = {
        "Start_Time": rush_hour_time_str,
        "Visibility_mi": 0.5,  # 낮은 가시성
        "Wind_Speed_mph": 15.0,  # 높은 풍속
        "Precipitation_in": 0.2,  # 강수량 (빙판길 조건 충족)
        "Temperature_F": 20.0,  # 빙점 이하 (빙판길 조건 충족)
        "Wind_Chill_F": 10.0,
        "Humidity_percent": 85.0,  # 높은 습도
        "Pressure_in": 29.5
    }
    run_prediction_test(payload_risk, "2. 위험 시나리오 (빙판길/러시 아워)")

    # 3. 안전 시나리오: 맑고 따뜻한 한낮 (Tier 0 예상)
    # Start_Time의 날짜는 현재 날짜를 사용하고 시간만 14:00:00으로 설정
    safe_time_str = datetime.now().strftime('%Y-%m-%d') + " 14:00:00"
    payload_very_safe = {
        "Start_Time": safe_time_str,
        "Visibility_mi": 10.0,
        "Wind_Speed_mph": 3.0,
        "Precipitation_in": 0.0,
        "Temperature_F": 75.0,
        "Wind_Chill_F": 75.0,
        "Humidity_percent": 40.0,
        "Pressure_in": 30.5
    }
    run_prediction_test(payload_very_safe, "3. 안전 시나리오 (맑음/비러시 아워)")


if __name__ == "__main__":
    run_all_prediction_examples()