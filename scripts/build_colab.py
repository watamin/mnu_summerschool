"""소스와 예비 데이터를 내장한 독립 실행형 학생용 Colab 노트북을 만든다."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "notebooks" / "우리학교_급식_AI_개인추천기_학생용.ipynb"
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "namak_meals_sample.json"


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def _code(source: str, *, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.strip() + "\n",
    }


def _module_source(filename: str, *, remove_relative_imports: bool = False) -> str:
    source = (PROJECT_ROOT / "src" / "neis_meal_ai" / filename).read_text(encoding="utf-8")
    if remove_relative_imports:
        source = "\n".join(line for line in source.splitlines() if not line.startswith("from ."))
    return source


def build_notebook(output_path: str | Path = DEFAULT_OUTPUT, sample_path: str | Path = DEFAULT_SAMPLE) -> Path:
    output = Path(output_path)
    sample = Path(sample_path)
    payload = json.loads(sample.read_text(encoding="utf-8"))
    embedded_payload = repr(payload)

    cells = [
        _markdown(
            """
# 🍱 우리 학교 급식 데이터 AI 탐험대
## 나에게 맞는 급식 개인추천기 — 학생용 완성 프로젝트

이 노트북 하나만 Colab에 업로드하면 실행할 수 있습니다. 5회 동안 각 단계를 이해하고 바꾸면서 우리 팀의 서비스를 완성합니다.

### 완성 후 설명할 수 있어야 하는 것

1. NEIS API가 학교 급식 데이터를 주는 과정
2. 메뉴 문자열을 분석 가능한 표로 바꾸는 전처리
3. TF-IDF가 한국어 메뉴를 숫자로 바꾸는 방식
4. 코사인 유사도가 취향과 메뉴를 비교하는 방식
5. K-Means가 비슷한 식단을 묶는 방식
6. AI 추천이 정답이 아니라는 점과 알레르기 안전 원칙

> 개인정보 약속: 이름·학번·반·연락처·체중·질병명은 입력하지 않습니다. 코드 셀에는 가상 프로필만 둡니다. Colab 서비스는 임시 공개 링크이므로 실제 알레르기·질병 정보는 입력하지 않고 수업용 가상 번호만 사용합니다.
"""
        ),
        _markdown(
            """
## 0단계: 실행 환경 확인

Colab에는 대부분의 라이브러리가 준비되어 있습니다. 서비스 화면용 Gradio가 없을 때만 설치합니다.

- 예상 결과: `환경 준비 완료`가 출력됩니다.
- 확인 질문: 라이브러리는 직접 모든 코드를 쓰지 않고도 검증된 기능을 사용할 수 있게 해 줍니다.
"""
        ),
        _code(
            """
import importlib.util
import os
import subprocess
import sys

VERIFY_MODE = os.getenv("NEIS_MEAL_AI_VERIFY", "0") == "1"
if not VERIFY_MODE and importlib.util.find_spec("gradio") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<7"], check=True)
print("환경 준비 완료", "(자동 검증 모드)" if VERIFY_MODE else "(Colab 학습 모드)")
"""
        ),
        _markdown(
            """
## 1단계: NEIS API 이해하기

API는 다른 서비스가 공개한 정보를 정해진 주소와 규칙으로 요청하는 창구입니다. 아래 코드는 학교명으로 학교 코드를 찾고, 그 코드로 급식 행을 가져옵니다.

- 입력: 학교명, 시작일, 종료일
- 출력: 메뉴·열량·영양·알레르기 번호가 들어 있는 JSON 행
- 학생 도전: `pSize`를 찾아 한 번에 몇 행을 요청하는지 확인하세요.
"""
        ),
        _code(_module_source("neis.py")),
        _markdown(
            """
## 2단계: 실시간 데이터와 예비 데이터

먼저 NEIS에 접속하고, 네트워크나 서버 문제가 생기면 노트북 안에 포함된 실제 남악고 공개 급식 5건을 사용합니다. 예비 자료를 쓰면 출처 메시지가 달라집니다.

- 예상 결과: 데이터 출처와 원본 행 수가 출력됩니다.
- 확인 질문: 예비 데이터가 없으면 수업 중 API 장애가 전체 프로젝트 중단으로 이어질 수 있습니다.
"""
        ),
        _code(f"EMBEDDED_SAMPLE_PAYLOAD = {embedded_payload}\nEMBEDDED_SAMPLE_ROWS = EMBEDDED_SAMPLE_PAYLOAD['rows']"),
        _code(
            """
SCHOOL_NAME = "남악고등학교"
QUERY_START = "20260101"
QUERY_END = "20261231"

if VERIFY_MODE:
    raw_rows = EMBEDDED_SAMPLE_ROWS
    data_source = "내장 NEIS 예비 데이터"
else:
    try:
        school = search_school(SCHOOL_NAME)
        raw_rows = fetch_meals(school, QUERY_START, QUERY_END)
        if not raw_rows:
            raise NeisApiError("선택 기간에 데이터가 없습니다.")
        data_source = "실시간 NEIS 데이터"
    except NeisApiError as error:
        raw_rows = [
            row for row in EMBEDDED_SAMPLE_ROWS
            if QUERY_START <= str(row.get("MLSV_YMD", "")) <= QUERY_END
        ]
        if not raw_rows:
            sample_dates = sorted(str(row.get("MLSV_YMD", "")) for row in EMBEDDED_SAMPLE_ROWS)
            raise NeisApiError(
                f"요청 기간과 겹치는 예비 데이터가 없습니다. "
                f"수업용 예비 데이터 기간: {sample_dates[0]}~{sample_dates[-1]}"
            ) from error
        data_source = f"내장 NEIS 예비 데이터 (사유: {error})"

print("데이터 출처:", data_source)
print("원본 급식 행:", len(raw_rows))
"""
        ),
        _markdown(
            """
## 3단계: 메뉴 문자열 전처리

원본에는 `<br/>`, 알레르기 번호, `Kcal` 같은 표기가 섞여 있습니다. AI가 비교할 수 있도록 메뉴 목록과 숫자 열로 분리합니다.

- 예상 결과: 날짜, 메뉴 문장, 열량, 영양 수치가 있는 표
- 학생 도전: `split_dishes`에서 HTML 줄바꿈이 어떻게 처리되는지 찾아 표시하세요.
"""
        ),
        _code(_module_source("cleaning.py")),
        _code(
            """
meal_df = meals_to_frame(raw_rows)
if meal_df.empty:
    raise RuntimeError("분석할 급식 데이터가 없습니다.")
print(meal_df[["date", "menu_text", "calories", "protein_g"]].head().to_string(index=False))
print("\\n분석 가능한 행:", len(meal_df))
"""
        ),
        _markdown(
            """
## 4단계: 데이터 탐색과 시각화

AI를 만들기 전에 데이터의 범위와 빠진 값을 확인합니다. 그래프는 정답을 주는 장식이 아니라 데이터의 특징과 오류를 찾는 도구입니다.

- 학생 도전: 가장 열량이 높은 식단과 낮은 식단의 메뉴를 비교하세요.
- 주의: 열량이 높거나 낮다는 사실만으로 건강함을 판정하지 않습니다.
"""
        ),
        _code(
            """
summary_columns = ["calories", "carbs_g", "protein_g", "fat_g", "dish_count"]
print(meal_df[summary_columns].describe().round(1).to_string())

if importlib.util.find_spec("matplotlib") is not None and not VERIFY_MODE:
    import matplotlib.pyplot as plt
    plot_df = meal_df.sort_values("date")
    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["date"], plot_df["calories"], color="#4F8BF9")
    plt.title("남악고 급식 열량 비교 — 상대 비교용")
    plt.ylabel("Kcal")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("자동 검증에서는 표로 확인했습니다. Colab에서는 막대그래프가 표시됩니다.")
"""
        ),
        _markdown(
            """
## 5단계: TF-IDF와 코사인 유사도

TF-IDF는 각 메뉴에서 특징적인 글자 조각에 더 큰 값을 줍니다. 코사인 유사도는 취향 벡터와 메뉴 벡터가 같은 방향을 가리키는 정도를 0~1로 비교합니다.

이 수업에서는 라이브러리 한 줄로 숨기지 않고 문자 n-gram TF-IDF와 작은 K-Means를 직접 구현해 내부 원리를 관찰합니다.

- 확인 질문: 모든 메뉴에 자주 나오는 글자와 특정 메뉴에만 나오는 글자 중 어느 쪽이 구별에 유리할까요?
"""
        ),
        _code(_module_source("recommender.py")),
        _markdown(
            """
## 6단계: K-Means로 식단 유형 찾기

K-Means는 정답표 없이 비슷한 숫자 패턴을 가까운 중심점에 묶습니다. 열량·탄수화물·단백질·지방·메뉴 수를 사용합니다.

군집명은 데이터 안에서의 상대적 차이일 뿐 `건강함`이나 `좋음`을 뜻하지 않습니다.
"""
        ),
        _code(
            """
clustered_df = cluster_meals(meal_df, max_clusters=3)
print(clustered_df[["date", "calories", "protein_g", "cluster_name"]].to_string(index=False))
"""
        ),
        _markdown(
            """
## 7단계: 익명 개인 취향 프로필 만들기

아래 값은 알고리즘 시험용 **가상 학생 프로필**입니다. 이름이나 학번은 만들지 않습니다. 좋아하는 메뉴를 바꾸어 실험할 수 있지만, 실제 알레르기 정보는 코드 셀에 적지 말고 `allergy_codes=()`를 유지하세요.

- `likes`: 좋아하는 재료·메뉴 최대 5개
- `avoids`: 피하고 싶은 재료·메뉴 최대 5개
- `preferred_types`: 밥, 면, 국물, 튀김, 디저트
- `spice_level`: 1~5
- `allergy_codes`: NEIS 알레르기 주의 번호 1~19, 선택 입력
"""
        ),
        _code(
            """
demo_profile = PreferenceProfile(
    likes=("파스타", "피자"),
    avoids=("오이",),
    preferred_types=("면", "디저트"),
    spice_level=2,
    allergy_codes=(),
)
validate_profile(demo_profile)
print("가상 취향 프로필 준비 완료 (개인 식별 정보·실제 알레르기 입력 없음)")
"""
        ),
        _markdown(
            """
## 8단계: 개인별 추천 결과와 이유

점수는 아래 수식을 0~100점으로 자른 상대 점수입니다. 20점 기준점은 감점이 0점 아래로 즉시 사라지지 않게 합니다.

`20 + 70×텍스트유사도 + 8×좋아하는키워드수 + 5×선호유형수 - 18×기피키워드수 - 3×매운맛차이`

알레르기 주의 번호가 겹치면 점수 계산 전에 추천 후보에서 제외합니다.

- 학생 도전: 취향을 바꾸고 1위가 왜 달라졌는지 `추천 이유`로 설명하세요.
"""
        ),
        _code(_module_source("service.py", remove_relative_imports=True)),
        _code(
            """
recommendation_summary, recommendation_result = run_recommendation(
    meal_df,
    likes_text=", ".join(demo_profile.likes),
    avoids_text=", ".join(demo_profile.avoids),
    preferred_types=demo_profile.preferred_types,
    spice_level=demo_profile.spice_level,
    allergy_codes=demo_profile.allergy_codes,
    top_n=3,
)
print(recommendation_summary)
print(recommendation_result.to_string(index=False))
"""
        ),
        _markdown(
            """
## 9단계: 추천 결과 시험하기

AI 서비스는 한 번 실행되는 것보다 입력을 바꿔도 규칙대로 움직이는지가 중요합니다. 서로 다른 두 취향의 1위를 비교합니다.

- 시험 A: 면·피자 선호
- 시험 B: 밥·국물 선호
- 성공 기준: 두 결과의 이유를 데이터로 설명할 수 있다.
"""
        ),
        _code(
            """
profile_a = PreferenceProfile(("파스타", "피자"), (), ("면",), 2, ())
profile_b = PreferenceProfile(("밥", "국"), (), ("밥", "국물"), 3, ())
result_a = recommend_menus(meal_df, profile_a, top_n=1)
result_b = recommend_menus(meal_df, profile_b, top_n=1)
print("A 프로필 1위:", result_a.iloc[0]["menu_text"], "/", result_a.iloc[0]["reason"])
print("B 프로필 1위:", result_b.iloc[0]["menu_text"], "/", result_b.iloc[0]["reason"])
"""
        ),
        _code(_module_source("ui.py")),
        _markdown(
            """
## 10단계: Gradio 서비스 화면

아래 셀을 실행하면 입력 상자와 추천 버튼이 있는 서비스가 열립니다. Colab에서는 접속 가능한 임시 공개 링크가 만들어집니다. 이름·학번은 입력하지 말고, 알레르기 번호도 실제 정보 대신 발표용 가상 번호로만 시험합니다.

발표에서는 `입력 → 데이터 → AI 비교 → 추천 결과 → 한계` 순서로 시연하세요.
"""
        ),
        _code(
            """
def build_colab_demo(current_frame):
    import gradio as gr

    def recommend_ui(likes, avoids, menu_types, spice, allergies):
        summary, table = run_recommendation(
            current_frame,
            likes_text=likes,
            avoids_text=avoids,
            preferred_types=menu_types,
            spice_level=int(spice),
            allergy_codes=[int(code) for code in allergies],
        )
        return summary, table

    with gr.Blocks(title="우리 학교 급식 AI 개인추천기") as demo:
        gr.Markdown(
            "# 🍱 우리 학교 급식 AI 개인추천기\\n"
            "이름·학번은 입력하지 않습니다. Colab 링크는 임시 공개 링크입니다.\\n\\n"
            "⚠️ 실제 알레르기·질병 정보는 입력하지 말고 수업용 가상 번호만 사용하세요."
        )
        likes = gr.Textbox(label="좋아하는 재료·메뉴", value="파스타, 피자")
        avoids = gr.Textbox(label="피하고 싶은 재료·메뉴", value="오이")
        menu_types = gr.CheckboxGroup(["밥", "면", "국물", "튀김", "디저트"], label="선호 유형")
        spice = gr.Slider(1, 5, value=3, step=1, label="매운맛 선호도")
        allergies = gr.CheckboxGroup(
            [str(code) for code in range(1, 20)],
            label="알레르기 주의 번호(가상 시연 전용)",
        )
        button = gr.Button("나에게 맞는 급식 찾기", variant="primary")
        message = gr.Markdown()
        table = gr.Dataframe(interactive=False)
        button.click(recommend_ui, [likes, avoids, menu_types, spice, allergies], [message, table])
        gr.Markdown("⚠️ 실제 식단과 알레르기 정보는 학교 급식표와 영양사 안내를 다시 확인하세요.")
    return demo

if VERIFY_MODE:
    print("Gradio 화면 함수 정의 완료 - 자동 검증에서는 서버를 열지 않습니다.")
else:
    demo = build_colab_demo(meal_df)
    demo.launch(**launch_options(is_colab=is_google_colab()))
""",
            tags=["service-ui"],
        ),
        _markdown(
            """
## 11단계: 모델 카드와 발표 준비

모델 카드는 AI가 무엇을 하고, 어떤 데이터를 쓰며, 어디에서 틀릴 수 있는지 공개하는 설명서입니다.

### 반드시 말할 한계

- 추천 점수는 취향 유사도이지 건강 점수가 아니다.
- 메뉴명에 없는 재료는 AI가 알 수 없다.
- 알레르기 번호 누락이나 식단 변경 가능성이 있다.
- 현재 수집한 작은 표본이 전체 사용자를 대표하지 않는다.
- 생성형 AI로 발표문을 만들었다면 NEIS 원본과 다시 비교한다.
"""
        ),
        _code(
            """
model_card = {
    "서비스 목적": "익명 취향과 NEIS 메뉴 텍스트를 비교한 상대 추천",
    "사용 데이터": data_source,
    "AI 방법": "문자 n-gram TF-IDF, 코사인 유사도, 작은 K-Means",
    "수집하지 않는 정보": "이름, 학번, 반, 연락처, 체중, 질병명",
    "안전 문구": SAFETY_NOTICE,
    "핵심 한계": "메뉴명과 선택한 취향만 비교하므로 실제 만족도나 건강 적합도를 예측하지 않음",
}
for key, value in model_card.items():
    print(f"- {key}: {value}")
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "우리학교_급식_AI_개인추천기_학생용.ipynb",
                "provenance": [],
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


def main() -> int:
    path = build_notebook()
    print(f"created={path} cells={len(json.loads(path.read_text(encoding='utf-8'))['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
