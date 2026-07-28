from __future__ import annotations

import json

import pandas as pd
import pytest

from neis_meal_ai.neis import NeisApiError
from neis_meal_ai.service import load_meal_frame, run_recommendation


RAW_ROWS = [
    {
        "ATPT_OFCDC_SC_CODE": "Q10",
        "SD_SCHUL_CODE": "7140272",
        "SCHUL_NM": "남악고등학교",
        "MMEAL_SC_NM": "중식",
        "MLSV_YMD": "20260723",
        "DDISH_NM": "쌀밥 (5)<br/>매운짬뽕국 (6.9.17)<br/>통새우튀김 (1.5.6.9)",
        "CAL_INFO": "850.0 Kcal",
        "NTR_INFO": "탄수화물(g) : 125.0<br/>단백질(g) : 35.0<br/>지방(g) : 25.0",
    },
    {
        "ATPT_OFCDC_SC_CODE": "Q10",
        "SD_SCHUL_CODE": "7140272",
        "SCHUL_NM": "남악고등학교",
        "MMEAL_SC_NM": "중식",
        "MLSV_YMD": "20260724",
        "DDISH_NM": "치즈스파게티 (1.2.5.6)<br/>오이피클<br/>자몽푸딩 (2)",
        "CAL_INFO": "927.7 Kcal",
        "NTR_INFO": "탄수화물(g) : 139.0<br/>단백질(g) : 40.6<br/>지방(g) : 23.0",
    },
    {
        "ATPT_OFCDC_SC_CODE": "Q10",
        "SD_SCHUL_CODE": "7140272",
        "SCHUL_NM": "남악고등학교",
        "MMEAL_SC_NM": "중식",
        "MLSV_YMD": "20260725",
        "DDISH_NM": "현미밥<br/>된장국 (5.6)<br/>바나나",
        "CAL_INFO": "710.0 Kcal",
        "NTR_INFO": "탄수화물(g) : 108.0<br/>단백질(g) : 27.0<br/>지방(g) : 17.0",
    },
]


def write_sample(path) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "school_name": "남악고등학교",
                    "office_code": "Q10",
                    "school_code": "7140272",
                    "source": "NEIS 교육정보 개방 포털",
                },
                "rows": RAW_ROWS,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_meal_frame_uses_live_rows_when_fetch_succeeds(tmp_path) -> None:
    fallback = tmp_path / "sample.json"
    write_sample(fallback)

    frame, message = load_meal_frame(
        "남악고등학교",
        "20260723",
        "20260725",
        fallback,
        fetcher=lambda school, start, end: RAW_ROWS,
    )

    assert len(frame) == 3
    assert "실시간 NEIS 데이터" in message


def test_load_meal_frame_uses_fallback_on_api_error(tmp_path) -> None:
    fallback = tmp_path / "sample.json"
    write_sample(fallback)

    def raising_fetcher(school: str, start: str, end: str):
        raise NeisApiError("연결 실패")

    frame, message = load_meal_frame(
        "남악고등학교",
        "20260723",
        "20260725",
        fallback,
        fetcher=raising_fetcher,
    )

    assert len(frame) == 3
    assert "예비 데이터" in message
    assert "연결 실패" in message


def test_load_meal_frame_filters_fallback_to_requested_dates(tmp_path) -> None:
    fallback = tmp_path / "sample.json"
    write_sample(fallback)

    def raising_fetcher(school: str, start: str, end: str):
        raise NeisApiError("연결 실패")

    frame, _ = load_meal_frame(
        "남악고등학교",
        "20260724",
        "20260724",
        fallback,
        fetcher=raising_fetcher,
    )

    assert frame["date"].tolist() == ["2026-07-24"]


def test_load_meal_frame_explains_when_fallback_has_no_requested_dates(tmp_path) -> None:
    fallback = tmp_path / "sample.json"
    write_sample(fallback)

    def raising_fetcher(school: str, start: str, end: str):
        raise NeisApiError("연결 실패")

    with pytest.raises(NeisApiError, match="예비 데이터 기간"):
        load_meal_frame(
            "남악고등학교",
            "20260801",
            "20260801",
            fallback,
            fetcher=raising_fetcher,
        )


def test_load_meal_frame_does_not_hide_an_empty_live_period(tmp_path) -> None:
    fallback = tmp_path / "sample.json"
    write_sample(fallback)

    with pytest.raises(ValueError, match="급식 데이터가 없습니다"):
        load_meal_frame(
            "남악고등학교",
            "20261225",
            "20261225",
            fallback,
            fetcher=lambda school, start, end: [],
        )


def test_run_recommendation_returns_korean_service_table() -> None:
    from neis_meal_ai.cleaning import meals_to_frame

    frame = meals_to_frame(RAW_ROWS)
    summary, table = run_recommendation(
        frame,
        likes_text="스파게티, 치즈",
        avoids_text="오이",
        preferred_types=["면", "디저트"],
        spice_level=2,
        allergy_codes=[],
        top_n=3,
    )

    assert "학교 급식표와 영양사 안내" in summary
    assert list(table.columns) == [
        "순위",
        "날짜",
        "추천 점수",
        "메뉴",
        "추천 이유",
        "식단 군집",
        "알레르기 번호",
    ]
    assert "치즈스파게티" in table.iloc[0]["메뉴"]
    assert table.iloc[0]["순위"] == 1


def test_run_recommendation_explains_when_every_menu_is_excluded() -> None:
    from neis_meal_ai.cleaning import meals_to_frame

    frame = meals_to_frame(RAW_ROWS[:1])
    summary, table = run_recommendation(
        frame,
        likes_text="",
        avoids_text="",
        preferred_types=[],
        spice_level=3,
        allergy_codes=[5],
        top_n=3,
    )

    assert table.empty
    assert "모든 메뉴가 제외" in summary
