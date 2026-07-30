from __future__ import annotations

import json
from pathlib import Path

from jupyter_course.notebook_support import load_sample_frame
from neis_meal_ai.recommender import MENU_TYPE_KEYWORDS
from neis_meal_ai.ui import build_demo, launch_options


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_colab_launch_uses_share_link() -> None:
    options = launch_options(is_colab=True)

    assert options["share"] is True
    assert "server_name" not in options
    assert "inbrowser" not in options


def test_local_launch_stays_local() -> None:
    options = launch_options(is_colab=False)

    assert options["share"] is False
    assert options["server_name"] == "127.0.0.1"
    assert options["inbrowser"] is True


def test_student_demo_contains_korean_inputs_results_and_safety_notice() -> None:
    demo = build_demo(load_sample_frame(PROJECT_ROOT), "남악고 NEIS 예비 데이터")
    config_text = json.dumps(demo.get_config_file(), ensure_ascii=False)

    expected_text = (
        "우리 학교 급식 추천 실험실",
        "좋아하는 재료·메뉴",
        "피하고 싶은 재료·메뉴",
        "선호 유형",
        "매운맛 선호도",
        "알레르기 주의 번호(가상 시연 전용)",
        "추천 결과 보기",
        "추천 이유",
        "학교 급식표와 영양사 안내",
        "남악고 NEIS 예비 데이터",
    )
    assert all(text in config_text for text in expected_text)
    assert len(demo.get_config_file()["dependencies"]) == 1

    type_component = next(
        component
        for component in demo.get_config_file()["components"]
        if component.get("props", {}).get("label") == "선호 유형"
    )
    displayed_types = [value for _, value in type_component["props"]["choices"]]
    assert displayed_types == list(MENU_TYPE_KEYWORDS)
