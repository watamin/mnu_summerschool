from __future__ import annotations

import inspect
from datetime import date

import pandas as pd

from .lunch_prediction import (
    compare_actual_ui_callback,
    predict_profile_callback,
    select_menu_row,
)
from .mokpo_analytics import recommend_high_schools, signature_terms
from .mokpo_data import MokpoDataset
from .student_profile_ui import load_profile_callback, save_profile_callback
from .student_profiles import StudentProfileStore


def _rating_table(gradio_module: object):
    dataframe = getattr(gradio_module, "Dataframe")
    kwargs: dict[str, object] = {
        "headers": ["순서", "음식", "구분", "평점"],
        "datatype": ["number", "str", "str", "number"],
        "value": pd.DataFrame(columns=["순서", "음식", "구분", "평점"]),
        "label": "내 음식 30개 평가표",
        "interactive": True,
    }
    parameters = inspect.signature(dataframe).parameters
    if "static_columns" in parameters:
        kwargs["static_columns"] = [0, 1, 2]
    if "pinned_columns" in parameters:
        kwargs["pinned_columns"] = 3
    return dataframe(**kwargs)


def _menu_summary(validation_menus: pd.DataFrame, menu_id: object) -> str:
    menu = select_menu_row(validation_menus, menu_id)
    dishes = " · ".join(str(dish) for dish in menu["dishes"])
    return f"**{menu['meal_date']} 목포대학교 학생식당 점심**  \n{dishes}"


def create_mokpo_app(
    dataset: MokpoDataset,
    validation_menus: pd.DataFrame,
    *,
    profile_store: StudentProfileStore | None = None,
):
    if profile_store is None:
        raise ValueError("학생 평가를 저장할 프로필 저장소가 필요합니다.")
    import gradio as gr

    today_text = date.today().isoformat()
    menu_ids = validation_menus["menu_id"].astype(str).tolist()
    today_rows = validation_menus.loc[validation_menus["meal_date"] == today_text]
    default_menu_id = str(today_rows.iloc[0]["menu_id"]) if len(today_rows) else menu_ids[0]
    choices = [
        (f"{row.meal_date} 점심", str(row.menu_id))
        for row in validation_menus.itertuples()
    ]
    school_choices = sorted(
        dataset.meals["school_name"].dropna().astype(str).unique().tolist()
    )
    default_school_name = school_choices[0]

    with gr.Blocks(title="오늘 점심 취향 예측") as demo:
        prediction_state = gr.State({})
        gr.Markdown(
            "# 오늘 점심 취향 예측\n음식 30개를 평가합니다.  \n"
            "오늘 점심을 식사 전에 예상합니다.  \n식사 후 실제 만족도와 비교합니다.\n\n"
            "예측은 정답이 아닙니다.  \n"
            "저장된 평점과 메뉴 이름을 비교합니다."
        )
        gr.Markdown("---\n## 1. 음식 30개 평가")
        with gr.Row():
            student_name = gr.Textbox(
                label="이름",
                placeholder="이름을 입력하세요",
                interactive=True,
            )
            load_button = gr.Button("내 평가표 불러오기")
            save_button = gr.Button("30개 평점 저장", variant="primary")
        profile_message = gr.Markdown("이름을 입력하고 내 평가표를 불러오세요.")
        rating_table = _rating_table(gr)

        gr.Markdown("---\n## 2. 오늘 점심 예상하기")
        meal_choice = gr.Radio(choices, value=default_menu_id, label="점심 날짜")
        meal_summary = gr.Markdown(_menu_summary(validation_menus, default_menu_id))
        predict_button = gr.Button("오늘 점심 예상하기", variant="primary")
        prediction_message = gr.Markdown("30개 평점을 저장한 뒤 예상하기를 누르세요.")
        dish_table = gr.Dataframe(label="메뉴별 예상과 근거", interactive=False)

        gr.Markdown("---\n## 3. 식사 후 실제 만족도 비교")
        actual_rating = gr.Slider(1, 5, value=3, step=1, label="실제 만족도(1~5)")
        compare_button = gr.Button("예측과 비교하기", variant="primary")
        comparison_message = gr.Markdown("식사 후 실제 만족도를 입력하세요.")
        comparison_table = gr.Dataframe(label="예측과 실제 비교", interactive=False)

        gr.Markdown("---\n## 4. 학교 급식 탐색")
        gr.Markdown(
            "학교별 시그니처 메뉴를 살펴보고, 좋아하는 음식으로 고등학교 급식 취향을 비교합니다.  \n"
            "이 순위는 급식 취향만 비교하며 학교의 교육·진학 적합도를 평가하지 않습니다."
        )
        with gr.Row():
            school_choice = gr.Dropdown(
                school_choices,
                value=default_school_name,
                label="시그니처 메뉴를 볼 학교",
            )
            preference_text = gr.Textbox(
                label="좋아하는 음식",
                value="돈까스, 파스타, 치즈",
                placeholder="예: 돈까스, 파스타, 치즈",
            )
        school_analysis_button = gr.Button(
            "시그니처 메뉴와 고등학교 추천 보기", variant="primary"
        )
        school_analysis_message = gr.Markdown(
            "학교를 고르고 좋아하는 음식을 입력한 뒤 분석을 시작하세요."
        )
        with gr.Row():
            signature_table = gr.Dataframe(
                label="학교별 시그니처 메뉴(TF-IDF)", interactive=False
            )
            high_school_table = gr.Dataframe(
                label="음식 취향 기준 고등학교 추천", interactive=False
            )

        def _load_profile(entered_name: object):
            return load_profile_callback(profile_store, entered_name)

        def _save_profile(entered_name: object, table: object):
            return save_profile_callback(profile_store, entered_name, table)

        def _predict_profile(entered_name: object, menu_id: object):
            return predict_profile_callback(
                profile_store,
                validation_menus,
                entered_name,
                menu_id,
            )

        def _compare_actual(
            prediction: object,
            actual: object,
            entered_name: object,
            menu_id: object,
        ):
            return compare_actual_ui_callback(
                prediction,
                actual,
                entered_name,
                menu_id,
            )

        def _explore_school_food(
            school_name: str, entered_preference: str
        ) -> tuple[str, pd.DataFrame, pd.DataFrame]:
            signatures = signature_terms(dataset.meals, school_name)
            preference = entered_preference.strip()
            if not preference:
                return (
                    "### 학교별 시그니처 메뉴\n"
                    "왼쪽 표에서 학교 급식에 특히 자주 나온 메뉴를 확인하세요.  \n\n"
                    "### 음식 취향 기준 고등학교 추천\n"
                    "좋아하는 음식을 입력하면 고등학교 급식 취향 점수를 계산합니다.",
                    signatures,
                    pd.DataFrame(
                        columns=["학교", "급식 취향 점수", "비교 급식 수"]
                    ),
                )
            rankings, notice = recommend_high_schools(
                dataset.meals, preference, method="tfidf"
            )
            return (
                f"### {school_name}의 시그니처 메뉴\n"
                "TF-IDF는 이 학교에서는 자주 나오고 다른 학교에서는 상대적으로 드문 메뉴를 찾습니다.  \n\n"
                "### 음식 취향 기준 고등학교 추천\n"
                f"{notice}",
                signatures,
                rankings,
            )

        load_button.click(
            _load_profile,
            inputs=[student_name],
            outputs=[profile_message, rating_table],
            api_name="load_student_survey",
        )
        save_button.click(
            _save_profile,
            inputs=[student_name, rating_table],
            outputs=[profile_message, rating_table],
            api_name="save_student_survey",
        )
        meal_choice.change(
            lambda menu_id: _menu_summary(validation_menus, menu_id),
            inputs=[meal_choice],
            outputs=[meal_summary],
        )
        predict_button.click(
            _predict_profile,
            inputs=[student_name, meal_choice],
            outputs=[prediction_message, dish_table, prediction_state],
            api_name="predict_today_lunch",
        )
        compare_button.click(
            _compare_actual,
            inputs=[prediction_state, actual_rating, student_name, meal_choice],
            outputs=[comparison_message, comparison_table],
            api_name="compare_actual_rating",
        )
        school_analysis_button.click(
            _explore_school_food,
            inputs=[school_choice, preference_text],
            outputs=[school_analysis_message, signature_table, high_school_table],
            api_name="school_signature_and_high_school_recommendation",
        )
    return demo
