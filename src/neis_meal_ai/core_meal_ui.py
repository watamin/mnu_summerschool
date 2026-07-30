from __future__ import annotations

import inspect
from datetime import date

import pandas as pd

from .lunch_prediction import (
    compare_actual_ui_callback,
    predict_profile_callback,
    select_menu_row,
)
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
    del dataset
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
    return demo
