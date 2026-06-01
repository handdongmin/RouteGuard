# EscapeRoute Vision

EscapeRoute Vision은 실내 비상 대피 경로에 놓인 장애물과 잠재적 위험 요소를 탐지하는 영상 기반 컴퓨터비전 프로젝트입니다.

사용자가 방에서 출입구 방향으로 이동하며 촬영한 짧은 스마트폰 영상을 업로드하면, 영상 속 객체와 상대적인 깊이를 분석하고 통로를 방해할 가능성이 있는 물체를 찾아냅니다. 분석 결과는 주석이 표시된 영상, 시간대별 위험 요소 목록, 대피 경로 점수로 제공합니다.

이 프로젝트는 자취방이나 기숙사뿐만 아니라 연구실, 동아리방, 소형 사무실과 같은 실내 공간에서도 활용할 수 있습니다.

## 주요 기능

- 짧은 실내 대피 경로 영상 업로드
- 가방, 상자, 의자, 전선 등 통행을 방해할 수 있는 객체 탐지
- 단안 깊이 추정 모델을 활용한 객체별 상대 거리 분석
- 통로 후보 영역과 장애물의 겹침 정도 계산
- 시간대별 위험 요소 목록 및 대피 경로 점수 생성
- 위험 객체가 표시된 결과 영상 제공
- 정리 전후 대피 경로 점수 비교

## 처리 과정

```text
영상 업로드
  -> 프레임 추출
  -> 객체 탐지
  -> 상대 깊이 추정
  -> 통로 후보 영역 분석
  -> 위험도 점수 계산
  -> 주석 영상 및 리포트 생성
```

## 프로젝트 구조

```text
escape-route-vision/
  app.py                    # Streamlit 웹 애플리케이션 진입점
  src/
    config.py               # 공통 설정
    detector.py             # 객체 탐지
    depth_estimator.py      # 상대 깊이 추정
    path_analyzer.py        # 통로 후보 영역 분석
    risk_rules.py           # 위험도 점수 계산 규칙
    visualizer.py           # 결과 영상 및 리포트 생성
  data/
    samples/                # 데모용 짧은 영상
  assets/                   # README 스크린샷 및 데모 미디어
  outputs/                  # 생성된 결과 파일 (Git 제외)
  tests/                    # 위험도 계산 규칙 테스트
```

## 실행 방법

### 1. 가상 환경 생성

```bash
python -m venv .venv
```

### 2. 가상 환경 활성화

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 애플리케이션 실행

```bash
streamlit run app.py
```

## 기술 스택

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- Depth Anything V2
- NumPy
- Pillow

## 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
