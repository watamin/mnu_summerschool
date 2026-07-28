"""Colab과 로컬에서 실행할 수 있는 Gradio 개인별 급식 추천기."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neis_meal_ai.service import load_meal_frame, run_recommendation  # noqa: E402
from neis_meal_ai.ui import is_google_colab, launch_options  # noqa: E402


SAMPLE_PATH = PROJECT_ROOT / "data" / "namak_meals_sample.json"


def recommend_for_ui(
    school_name: str,
    start_date: str,
    end_date: str,
    likes: str,
    avoids: str,
    preferred_types: list[str],
    spice_level: int,
    allergy_codes: list[str],
):
    try:
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        frame, source_message = load_meal_frame(school_name, start, end, SAMPLE_PATH)
        summary, table = run_recommendation(
            frame,
            likes_text=likes,
            avoids_text=avoids,
            preferred_types=preferred_types,
            spice_level=spice_level,
            allergy_codes=[int(code) for code in allergy_codes],
        )
        return f"### 데이터 출처\n{source_message}\n\n### 결과 안내\n{summary}", table
    except Exception as exc:
        return f"### 실행 안내\n{exc}", []


def build_demo():
    import gradio as gr

    today = date.today()
    start_default = (today - timedelta(days=180)).isoformat()
    with gr.Blocks(title="우리 학교 급식 AI 개인추천기") as demo:
        gr.Markdown(
            "# 🍱 우리 학교 급식 AI 개인추천기\n"
            "이름이나 학번은 입력하지 않습니다. UI 입력은 파일에 저장하지 않습니다.\n\n"
            "⚠️ Colab 링크는 임시 공개 링크입니다. 실제 알레르기·질병 정보는 입력하지 말고 "
            "수업에서는 가상 번호만 사용하세요."
        )
        with gr.Row():
            school = gr.Textbox(label="정확한 학교명", value="남악고등학교")
            start = gr.Textbox(label="시작일 YYYY-MM-DD", value=start_default)
            end = gr.Textbox(label="종료일 YYYY-MM-DD", value=today.isoformat())
        with gr.Row():
            likes = gr.Textbox(label="좋아하는 재료·메뉴, 쉼표로 최대 5개", value="치즈, 면")
            avoids = gr.Textbox(label="피하고 싶은 재료·메뉴, 쉼표로 최대 5개", value="")
        preferred_types = gr.CheckboxGroup(
            ["밥", "면", "국물", "튀김", "디저트"],
            label="선호 메뉴 유형",
            value=["면"],
        )
        spice = gr.Slider(1, 5, value=3, step=1, label="매운맛 선호도")
        allergies = gr.CheckboxGroup(
            [str(code) for code in range(1, 20)],
            label="알레르기 주의 번호(가상 시연 전용)",
        )
        run_button = gr.Button("나에게 맞는 급식 찾기", variant="primary")
        message = gr.Markdown()
        table = gr.Dataframe(interactive=False)
        run_button.click(
            recommend_for_ui,
            inputs=[school, start, end, likes, avoids, preferred_types, spice, allergies],
            outputs=[message, table],
        )
        gr.Markdown(
            "⚠️ 추천 결과는 취향 비교용입니다. 실제 식단과 알레르기 정보는 "
            "학교 급식표와 영양사 안내를 다시 확인하세요."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(**launch_options(is_colab=is_google_colab()))
