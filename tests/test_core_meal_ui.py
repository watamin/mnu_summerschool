from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd
import pytest

from neis_meal_ai.core_meal_ui import create_mokpo_app
from neis_meal_ai.lunch_prediction import (
    compare_actual_callback,
    compare_actual_ui_callback,
    predict_profile_callback,
)
from neis_meal_ai.mokpo_data import load_validation_menus
from neis_meal_ai.student_profiles import StudentProfileStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _menus() -> pd.DataFrame:
    menus, _ = load_validation_menus(
        PROJECT_ROOT / "data" / "mnu_cafeteria_2026_07_30_31.json"
    )
    return menus


def _food_pool() -> list[str]:
    distinctive = [
        "돈까스",
        "잡곡밥",
        "김치",
        "부대찌개",
        "떡볶이",
        "스파게티",
        "계란국",
        "샐러드",
        "요구르트",
    ]
    return distinctive + [f"연습음식{number:02d}" for number in range(1, 37)]


def _complete_profile(store: StudentProfileStore, name: str) -> pd.DataFrame:
    survey = store.load_survey(name)
    survey["평점"] = 3
    survey.loc[survey["음식"] == "돈까스", "평점"] = 5
    survey.loc[survey["음식"] == "잡곡밥", "평점"] = 4
    survey.loc[survey["음식"] == "김치", "평점"] = 4
    store.save_ratings(name, survey)
    return survey


def test_prediction_uses_saved_thirty_food_ratings(tmp_path: Path) -> None:
    store = StudentProfileStore(tmp_path / "profiles.sqlite3", _food_pool())
    _complete_profile(store, "테스트학생")

    message, dishes, prediction = predict_profile_callback(
        store,
        _menus(),
        "테스트학생",
        "mnu-2026-07-30-lunch",
    )

    scores = dishes.set_index("오늘 메뉴")["예상 선호도(1~5)"]
    assert prediction["student_name"] == "테스트학생"
    assert prediction["meal_date"] == "2026-07-30"
    assert 1 <= prediction["predicted_score"] <= 5
    assert scores["등심돈까스"] > scores["고구마대나물"]
    assert "저장된 30개 평가" in message
    assert "식전에 만든 예측" in message


def test_prediction_requires_all_thirty_ratings(tmp_path: Path) -> None:
    store = StudentProfileStore(tmp_path / "profiles.sqlite3", _food_pool())
    survey = store.load_survey("연습학생")
    survey.loc[:9, "평점"] = 4
    store.save_ratings("연습학생", survey)

    with pytest.raises(ValueError, match="30개 평가를 모두"):
        predict_profile_callback(
            store,
            _menus(),
            "연습학생",
            "mnu-2026-07-30-lunch",
        )


def test_actual_review_compares_against_the_frozen_prediction() -> None:
    prediction = {
        "student_name": "테스트학생",
        "menu_id": "mnu-2026-07-30-lunch",
        "meal_date": "2026-07-30",
        "menu_text": "부대찌개 잡곡밥 등심돈까스",
        "predicted_score": 3.6,
    }

    message, comparison = compare_actual_callback(
        prediction, 4, "테스트학생", "mnu-2026-07-30-lunch"
    )

    assert "예상 3.60점" in message
    assert "실제 4.00점" in message
    assert "0.40점 높았습니다" in message
    assert comparison.iloc[0].to_dict() == {
        "날짜": "2026-07-30",
        "예상 만족도": 3.6,
        "실제 만족도": 4.0,
        "실제-예상": 0.4,
        "절대 오차": 0.4,
    }


def test_actual_review_rejects_a_prediction_from_another_student_or_menu() -> None:
    prediction = {
        "student_name": "첫학생",
        "menu_id": "mnu-2026-07-30-lunch",
        "meal_date": "2026-07-30",
        "menu_text": "부대찌개 잡곡밥",
        "predicted_score": 3.6,
    }

    with pytest.raises(ValueError, match="이름이나 점심 날짜"):
        compare_actual_callback(
            prediction, 4, "둘째학생", "mnu-2026-07-30-lunch"
        )
    with pytest.raises(ValueError, match="이름이나 점심 날짜"):
        compare_actual_callback(
            prediction, 4, "첫학생", "mnu-2026-07-31-lunch"
        )

    message, comparison = compare_actual_ui_callback(
        prediction, 4, "둘째학생", "mnu-2026-07-30-lunch"
    )
    assert "2단계에서 다시 예상" in message
    assert comparison.empty


def test_app_contains_the_experiment_and_high_school_side_dish_tab(tmp_path: Path) -> None:
    store = StudentProfileStore(tmp_path / "profiles.sqlite3", _food_pool())
    demo = create_mokpo_app(
        SimpleNamespace(
            metadata={},
            meals=pd.DataFrame(
                {
                    "school_name": ["목포고등학교"],
                    "school_kind": ["고등학교"],
                    "dishes": [["돈까스"]],
                }
            ),
        ),
        _menus(),
        profile_store=store,
    )
    config_file = demo.get_config_file()
    config = json.dumps(config_file, ensure_ascii=False, default=str)

    assert isinstance(demo, gr.Blocks)
    for text in (
        "오늘 점심 취향 예측",
        "1. 음식 30개 평가",
        "2. 오늘 점심 예상하기",
        "3. 식사 후 실제 만족도 비교",
        "내 평가표 불러오기",
        "30개 평점 저장",
        "오늘 점심 예상하기",
        "예측과 비교하기",
        "내 음식 30개 평가표",
        "고등학교 주요 반찬",
        "전체 고등학교 주요 반찬(TF-IDF)",
        "밥·국·김치·후식류를 제외한 주요 반찬",
    ):
        assert text in config
    for removed in (
        "학생 행렬분해 실험",
        "학생 설문·개인 결과",
        "30개 음식 역행렬 추천",
        "모둠 피드백 분석",
        "AI 식단 실험실",
        "NVIDIA NIM 데이터 해설",
        "학교별 시그니처 메뉴(TF-IDF)",
        "음식 취향 기준 고등학교 추천",
    ):
        assert removed not in config

    profile_table_props = next(
        component["props"]
        for component in config_file["components"]
        if component["props"].get("label") == "내 음식 30개 평가표"
    )
    assert profile_table_props["static_columns"] == [0, 1, 2]

    name_props = next(
        component["props"]
        for component in config_file["components"]
        if component["props"].get("label") == "이름"
    )
    assert name_props["interactive"] is True
    assert name_props["placeholder"] == "이름을 입력하세요"
