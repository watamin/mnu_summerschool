# 우리 학교 급식 데이터 AI 탐험대

남악고등학교 NEIS 공개 급식 데이터와 익명 취향 입력을 이용해 개인별 메뉴를 추천하는 중학교 2학년용 15시간 프로젝트입니다. 3시간씩 5회 동안 제작하고, 6회차 3시간은 발표와 상호 체험에 사용합니다.

## 권장 수업판: 분할 Jupyter 교과서

설명이 충분하고 학생이 장별로 따라 하기 쉬운 새 판은 `jupyter_course/README.md`에서 시작합니다.

- 00~08의 9개 노트북으로 분할
- 각 장마다 개념·비유·예상 결과·코드 해설·학생 실습·학습 확인·정리 제공
- 로컬 Jupyter Notebook 7과 ipywidgets 사용
- 외부 공유 링크 없이 현재 PC에서 추천 화면 실행
- 역할 분담표나 점수식 평가표 없이 학습 결과와 설명으로 확인

처음에는 `jupyter_course/00_설치_준비.md`를 열어 가상환경 생성과 requirements 설치 명령을 한 줄씩 실행합니다. 기존 단일 Colab 노트북은 백업·비교용으로 그대로 보존되어 있습니다.

## 내일 검토할 파일

1. `jupyter_course/00_설치_준비.md` — 가상환경·requirements·Jupyter 수동 설치
2. `jupyter_course/README.md` — 학생용 실행·학습 안내
3. `jupyter_course/chapters/00_시작하기.ipynb` — 설치 확인과 교과서판 시작
4. `jupyter_course/교사용_운영안.md` — 6회 분 단위 운영과 장별 확인 기준
5. `jupyter_course/verification-report.md` — 9개 장 새 커널 실행 결과
6. `notebooks/우리학교_급식_AI_개인추천기_학생용.ipynb` — 기존 Colab 통합판

## 완성 서비스

학생은 다음 익명 취향만 입력합니다.

- 좋아하는 재료·메뉴 최대 5개
- 피하고 싶은 재료·메뉴 최대 5개
- 밥·면·국물·튀김·디저트 중 선호 유형
- 매운맛 선호도 1~5
- 수업용 가상 알레르기 주의 번호 1~19(선택)

서비스는 NEIS 메뉴를 문자 n-gram TF-IDF로 숫자화하고 코사인 유사도와 명시적 가감점을 합쳐 상위 메뉴를 추천합니다. 점수식은 `0~100으로 제한(20 + 70×유사도 + 8×좋아함 일치 + 5×유형 일치 - 18×기피 일치 - 3×매운맛 차이)`입니다. 20점 기준점은 감점 효과가 0점 하한에서 사라지지 않게 합니다. 영양 수치는 완전한 영양 행이 3개 이상일 때만 작은 K-Means로 상대적 패턴을 묶고, 부족하면 `데이터 부족`으로 표시합니다. 점수는 건강 점수나 실제 만족도 예측이 아닙니다.

## 개인정보와 안전

- 이름, 학번, 반, 연락처, 체중, 질병명은 입력하거나 저장하지 않습니다.
- 코드 셀에는 가상 취향 프로필과 빈 알레르기 번호만 둡니다. UI 입력은 파일에 저장하지 않습니다.
- Colab의 Gradio 화면은 접속 가능한 임시 공개 링크를 만드므로 실제 알레르기·질병 정보는 입력하지 않습니다.
- 수업용 가상 알레르기 번호가 표시된 메뉴는 추천 후보에서 제외하는 동작만 시험합니다.
- 실제 메뉴와 알레르기 정보는 학교 급식표와 영양사 안내를 반드시 다시 확인합니다.
- 유료 LLM API, 로그인, 데이터베이스는 필요하지 않습니다.

## 기존 Colab 통합판 실행 방법

1. [Google Colab](https://colab.research.google.com/)을 엽니다.
2. `파일 → 노트 업로드`를 선택합니다.
3. `notebooks/우리학교_급식_AI_개인추천기_학생용.ipynb`를 업로드합니다.
4. `런타임 → 모두 실행`을 누릅니다.
5. 마지막 Gradio 셀에 생성된 임시 링크에서 개인추천기 화면을 사용합니다.

노트북은 실시간 NEIS 조회를 먼저 시도하고 NEIS 연결 오류가 나면 요청 날짜와 겹치는 내장 남악고 예비 데이터로 전환합니다. 예비 데이터의 실제 날짜는 `2026-06-24~2026-06-30`이며, 날짜가 겹치지 않으면 사용 가능 기간을 안내하고 멈춥니다. NEIS 응답이 1,000행을 넘을 때는 전체 개수만큼 다음 페이지도 자동 조회합니다.

## 로컬 개발 실행

Python 3.11 이상 환경에서 다음 명령을 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\build_colab.py
.\.venv\Scripts\python.exe scripts\verify_colab.py
.\.venv\Scripts\python.exe app.py
```

Jupyter 교과서판은 패키지 목록을 `requirements-jupyter.txt`에 한 줄씩 관리합니다. 학생은 `jupyter_course/00_설치_준비.md`를 보며 다음 명령을 순서대로 직접 실행합니다.

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-jupyter.txt
.\.venv\Scripts\python.exe -m notebook
```

## 폴더 구조

```text
neis-meal-ai/
├─ app.py                         Gradio 서비스
├─ data/                          공식 NEIS 예비 데이터
├─ docs/                          교사용·학생용 수업 자료
├─ jupyter_course/                권장 Jupyter 교과서판과 9개 장
├─ notebooks/                     Colab 학생용 파일
├─ requirements-jupyter.txt       Jupyter 수업판 패키지 목록
├─ scripts/                       데이터·노트북 생성과 검증
├─ src/neis_meal_ai/              API·전처리·추천 핵심 코드
└─ tests/                         자동 테스트
```

## 참고한 공개 자료의 역할

- [teddylee777/machine-learning](https://github.com/teddylee777/machine-learning): Python, Pandas, 데이터 시각화 학습 흐름
- [Microsoft AI for Beginners](https://github.com/microsoft/AI-For-Beginners): AI 입문, 초보 예제, 책임 있는 AI 구성
- [bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book): 에이전트의 `모델 + 맥락 + 도구` 개념을 확장 설명에 사용
- [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch): 만들기·테스트·설명서·검증까지 포함하는 프로젝트 방식
