from __future__ import annotations

import math

from neis_meal_ai.cleaning import (
    extract_allergy_codes,
    meals_to_frame,
    parse_calories,
    parse_nutrients,
    split_dishes,
)


RAW_ROW = {
    "SCHUL_NM": "남악고등학교",
    "MMEAL_SC_NM": "중식",
    "MLSV_YMD": "20260724",
    "DDISH_NM": (
        "미트소스치즈스파게티 (1.2.5.6.10.12.13.16)&lt;br/&gt;"
        "오이피클<br />통새우꼬치 (1.2.5.6.9)"
    ),
    "CAL_INFO": "927.7 Kcal",
    "NTR_INFO": "탄수화물(g) : 139.0<br/>단백질(g) : 40.6<br/>지방(g) : 23.0",
}


def test_split_dishes_removes_html_entities_and_allergy_suffixes() -> None:
    dishes = split_dishes(RAW_ROW["DDISH_NM"])
    assert dishes == ["미트소스치즈스파게티", "오이피클", "통새우꼬치"]


def test_extract_allergy_codes_deduplicates_and_sorts() -> None:
    assert extract_allergy_codes("가 (1.2.5)<br/>나 (2.6.19)") == (1, 2, 5, 6, 19)


def test_extract_allergy_codes_ignores_non_allergy_numbers() -> None:
    assert extract_allergy_codes("7월 메뉴<br/>비타민음료 100ml") == ()


def test_parse_nutrients_reads_core_numbers() -> None:
    parsed = parse_nutrients("탄수화물(g) : 139.0<br/>단백질(g) : 40.6<br/>지방(g) : 23.0")
    assert parsed == {"carbs_g": 139.0, "protein_g": 40.6, "fat_g": 23.0}


def test_parse_calories_returns_nan_when_missing() -> None:
    assert parse_calories("927.7 Kcal") == 927.7
    assert math.isnan(parse_calories(""))


def test_meals_to_frame_builds_stable_analysis_columns() -> None:
    frame = meals_to_frame([RAW_ROW])

    assert list(frame.columns) == [
        "date",
        "school_name",
        "meal_type",
        "dishes",
        "menu_text",
        "allergy_codes",
        "calories",
        "carbs_g",
        "protein_g",
        "fat_g",
        "dish_count",
    ]
    row = frame.iloc[0]
    assert row["date"] == "2026-07-24"
    assert row["menu_text"] == "미트소스치즈스파게티 오이피클 통새우꼬치"
    assert row["allergy_codes"] == (1, 2, 5, 6, 9, 10, 12, 13, 16)
    assert row["dish_count"] == 3
    assert row["protein_g"] == 40.6


def test_meals_to_frame_skips_rows_without_date_or_menu() -> None:
    no_date = {**RAW_ROW, "MLSV_YMD": ""}
    no_menu = {**RAW_ROW, "DDISH_NM": ""}
    assert meals_to_frame([no_date, no_menu]).empty
