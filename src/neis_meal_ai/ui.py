"""급식 추천기를 노트북과 로컬 웹에서 함께 쓰기 위한 화면 구성."""

from __future__ import annotations

import sys
from collections.abc import Iterable

import pandas as pd


RESULT_COLUMNS = [
    "순위",
    "날짜",
    "추천 점수",
    "메뉴",
    "추천 이유",
    "식단 군집",
    "알레르기 번호",
]


def is_google_colab() -> bool:
    """현재 코드가 Google Colab에서 실행 중인지 확인한다."""

    return "google.colab" in sys.modules


def launch_options(*, is_colab: bool) -> dict[str, object]:
    """실행 환경에 맞는 안전한 Gradio 공개 범위를 반환한다."""

    options: dict[str, object] = {
        "share": bool(is_colab),
        "debug": False,
        "show_error": True,
    }
    if not is_colab:
        options.update(
            {
                "server_name": "127.0.0.1",
                "inbrowser": True,
            }
        )
    return options


def _allergy_numbers(values: Iterable[str | int] | None) -> tuple[int, ...]:
    """체크박스에서 받은 알레르기 번호를 추천기 입력 형식으로 바꾼다."""

    return tuple(int(value) for value in (values or ()))


def build_demo(frame: pd.DataFrame, data_source: str):
    """주어진 급식표로 바로 사용할 수 있는 학생용 Gradio 화면을 만든다."""

    import gradio as gr

    from .recommender import MENU_TYPE_KEYWORDS, SAFETY_NOTICE
    from .service import run_recommendation

    def recommend(
        likes_text: str,
        avoids_text: str,
        preferred_types: list[str],
        spice_level: int,
        allergy_codes: list[str],
    ) -> tuple[str, pd.DataFrame]:
        try:
            return run_recommendation(
                frame,
                likes_text=likes_text,
                avoids_text=avoids_text,
                preferred_types=preferred_types,
                spice_level=int(spice_level),
                allergy_codes=_allergy_numbers(allergy_codes),
                top_n=3,
            )
        except (TypeError, ValueError) as exc:
            message = (
                "입력 내용을 다시 확인해 주세요. 이름·학번 같은 개인정보는 입력하지 않습니다. "
                f"\n\n확인할 내용: {exc}"
            )
            return message, pd.DataFrame(columns=RESULT_COLUMNS)

    with gr.Blocks(
        title="우리 학교 급식 추천 실험실",
    ) as demo:
        gr.Markdown(
            "# 🍽️ 우리 학교 급식 추천 실험실\n"
            f"**사용 데이터:** {data_source}\n\n"
            "좋아하는 음식의 특징을 입력하면 TF-IDF와 코사인 유사도를 이용해 "
            "급식 메뉴를 비교하고 추천 이유를 보여 줍니다."
        )
        gr.Markdown(
            "<div class='privacy-note'><strong>개인정보 약속</strong><br>"
            "이름·학번·반·연락처·질병명은 입력하지 않습니다. "
            "알레르기 항목은 실제 건강정보가 아닌 <strong>수업용 가상 번호</strong>로만 실험합니다.</div>"
        )

        with gr.Row():
            with gr.Column():
                likes = gr.Textbox(
                    label="좋아하는 재료·메뉴",
                    placeholder="예: 파스타, 치즈, 닭고기",
                    value="파스타, 피자",
                )
                avoids = gr.Textbox(
                    label="피하고 싶은 재료·메뉴",
                    placeholder="예: 오이, 버섯",
                    value="오이",
                )
                preferred_types = gr.CheckboxGroup(
                    choices=list(MENU_TYPE_KEYWORDS),
                    value=["면"],
                    label="선호 유형",
                )
                spice = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=2,
                    label="매운맛 선호도",
                    info="1은 순한 맛, 5는 매운 맛을 좋아한다는 뜻입니다.",
                )
                allergies = gr.CheckboxGroup(
                    choices=[str(number) for number in range(1, 20)],
                    label="알레르기 주의 번호(가상 시연 전용)",
                    info="실제 건강정보를 입력하지 말고, 수업에서 정한 가상 번호만 선택하세요.",
                )
                submit = gr.Button("추천 결과 보기", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Markdown(
                    "### 아직 추천하지 않았습니다\n"
                    "왼쪽에서 가상 취향을 정한 뒤 **추천 결과 보기**를 눌러 보세요."
                )
                result = gr.Dataframe(
                    value=pd.DataFrame(columns=RESULT_COLUMNS),
                    headers=RESULT_COLUMNS,
                    datatype=["number", "str", "number", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="추천 결과와 추천 이유",
                )

        submit.click(
            fn=recommend,
            inputs=[likes, avoids, preferred_types, spice, allergies],
            outputs=[summary, result],
        )
        gr.Markdown(
            "---\n"
            "이 결과는 수업용 추천 실험이며 건강·의료 판단이 아닙니다. "
            f"{SAFETY_NOTICE} 학교 급식표와 영양사 안내를 반드시 먼저 확인하세요."
        )

    return demo
