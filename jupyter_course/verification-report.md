# Jupyter 노트북 실행 확인

마지막 확인일: 2026년 7월 28일

이 기록은 문서의 완성도를 평가한 결과가 아니라, 저장소의 테스트와 아홉 개 노트북이 현재 환경에서 끝까지 실행되었는지 확인한 내용입니다.

## 확인한 환경

- Windows
- Python 3.12.13
- Jupyter Notebook 7.6.1
- ipywidgets 8.1.8
- nbclient 0.11.0
- nbformat 5.10.4
- ipykernel 7.3.0

## 실행한 명령

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\build_jupyter_textbook.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_jupyter_textbook.py
```

결과는 다음과 같습니다.

- 패키지 연결 오류 없음
- 자동 테스트 81개 통과
- 노트북 9개 통과, 실패 0개
- 새 커널 실행 합계 11.11초

## 장별 새 커널 실행

아래 실행 시간은 2026년 7월 28일에 한 번 측정한 값입니다. PC 상태에 따라 조금씩 달라질 수 있습니다.

| 장 | 파일 | 코드 셀 | 설명 셀 | 실행 시간 | 결과 |
|---:|---|---:|---:|---:|---|
| 00 | `00_시작하기.ipynb` | 4 | 18 | 1.46초 | PASS |
| 01 | `01_NEIS_API와_JSON.ipynb` | 5 | 21 | 1.20초 | PASS |
| 02 | `02_급식데이터_정리와_그래프.ipynb` | 5 | 21 | 1.46초 | PASS |
| 03 | `03_TFIDF_글자를_숫자로.ipynb` | 5 | 21 | 1.20초 | PASS |
| 04 | `04_유사도와_식단군집.ipynb` | 4 | 18 | 1.21초 | PASS |
| 05 | `05_개인추천_점수설계.ipynb` | 4 | 18 | 1.22초 | PASS |
| 06 | `06_Jupyter_추천화면.ipynb` | 5 | 21 | 1.25초 | PASS |
| 07 | `07_테스트와_모델카드.ipynb` | 4 | 18 | 1.23초 | PASS |
| 08 | `08_발표와_체험.ipynb` | 4 | 18 | 0.88초 | PASS |

각 장은 앞 장의 변수나 실행 상태를 넘겨받지 않고 별도의 Python 커널에서 실행했습니다.

## 자동으로 확인하는 내용

- 프로젝트 폴더와 `.venv` Python 경로
- 남악고등학교 예비 급식 5행의 구조
- 실시간 NEIS 조회 실패 시 같은 기간의 예비 자료 사용
- 9개 파일명, 셀 ID, Jupyter 메타데이터
- TF-IDF, 군집, 추천 결과의 필수 값
- 06장 버튼 콜백과 잘못된 입력 안내
- 07장의 네 가지 시험과 모델 카드
- 노트북을 다시 생성해도 같은 파일이 만들어지는지 여부
- Windows와 다른 운영체제에서 같은 LF 줄바꿈을 쓰는지 여부

## 수업 전에 다시 볼 것

자동 실행이 통과해도 학교 PC의 인터넷, Python 설치 권한, NEIS 서버 상태는 달라질 수 있습니다. 첫 수업 전에 00장과 06장을 학생 PC 한 대에서 다시 실행합니다. 실제 식단과 알레르기 정보는 추천 결과가 아니라 학교 급식표와 영양사 안내로 확인합니다.
