from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from neis_meal_ai.neis import (
    NeisApiError,
    SchoolInfo,
    fetch_meals,
    search_school,
    validate_date_range,
)


class FakeResponse:
    def __init__(self, payload: dict, *, raises: Exception | None = None) -> None:
        self.payload = payload
        self.raises = raises

    def raise_for_status(self) -> None:
        if self.raises:
            raise self.raises

    def json(self) -> dict:
        return self.payload


SCHOOL_PAYLOAD = {
    "schoolInfo": [
        {
            "head": [
                {"list_total_count": 1},
                {"RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다."}},
            ]
        },
        {
            "row": [
                {
                    "ATPT_OFCDC_SC_CODE": "Q10",
                    "SD_SCHUL_CODE": "7140272",
                    "SCHUL_NM": "남악고등학교",
                    "SCHUL_KND_SC_NM": "고등학교",
                    "ORG_RDNMA": "전라남도 무안군 삼향읍 남악4로60번길 24",
                }
            ]
        },
    ]
}

MEAL_ROW = {
    "ATPT_OFCDC_SC_CODE": "Q10",
    "SD_SCHUL_CODE": "7140272",
    "SCHUL_NM": "남악고등학교",
    "MMEAL_SC_NM": "중식",
    "MLSV_YMD": "20260724",
    "DDISH_NM": "스파게티 (1.2.5.6)<br/>오이피클",
    "CAL_INFO": "927.7 Kcal",
    "NTR_INFO": "탄수화물(g) : 139.0<br/>단백질(g) : 40.6<br/>지방(g) : 23.0",
}


def test_validate_date_range_accepts_inclusive_366_days() -> None:
    start, end = validate_date_range("20260101", "20270101")
    assert start == date(2026, 1, 1)
    assert end == date(2027, 1, 1)


def test_validate_date_range_rejects_reverse_order() -> None:
    with pytest.raises(ValueError, match="시작일"):
        validate_date_range("20260702", "20260701")


def test_validate_date_range_rejects_more_than_366_days() -> None:
    with pytest.raises(ValueError, match="366일"):
        validate_date_range("20260101", "20270102")


def test_search_school_reads_exact_school_and_query_contract() -> None:
    captured: dict = {}

    def fake_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse(SCHOOL_PAYLOAD)

    school = search_school("남악고등학교", http_get=fake_get)

    assert school == SchoolInfo(
        name="남악고등학교",
        office_code="Q10",
        school_code="7140272",
        school_kind="고등학교",
        address="전라남도 무안군 삼향읍 남악4로60번길 24",
    )
    assert captured["url"].endswith("/schoolInfo")
    assert captured["params"]["SCHUL_NM"] == "남악고등학교"
    assert captured["timeout"] == 15


def test_search_school_rejects_non_exact_match() -> None:
    def fake_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        payload = deepcopy(SCHOOL_PAYLOAD)
        payload["schoolInfo"][1]["row"][0]["SCHUL_NM"] = "남악초등학교"
        return FakeResponse(payload)

    with pytest.raises(NeisApiError, match="정확히 일치"):
        search_school("남악고등학교", http_get=fake_get)


def test_fetch_meals_returns_rows_and_uses_school_codes() -> None:
    captured: dict = {}
    school = SchoolInfo("남악고등학교", "Q10", "7140272", "고등학교", "전라남도 무안군")
    payload = {
        "mealServiceDietInfo": [
            {"head": [{"list_total_count": 1}, {"RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"}}]},
            {"row": [MEAL_ROW]},
        ]
    }

    def fake_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse(payload)

    rows = fetch_meals(school, "20260724", "20260724", http_get=fake_get)

    assert rows == [MEAL_ROW]
    assert captured["url"].endswith("/mealServiceDietInfo")
    assert captured["params"]["ATPT_OFCDC_SC_CODE"] == "Q10"
    assert captured["params"]["SD_SCHUL_CODE"] == "7140272"
    assert captured["params"]["MLSV_FROM_YMD"] == "20260724"


def test_fetch_meals_returns_empty_for_info_200() -> None:
    school = SchoolInfo("남악고등학교", "Q10", "7140272", "고등학교", "전라남도 무안군")

    def empty_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        return FakeResponse({"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}})

    assert fetch_meals(school, "20260701", "20260702", http_get=empty_get) == []


def test_fetch_meals_collects_every_page(monkeypatch) -> None:
    school = SchoolInfo("남악고등학교", "Q10", "7140272", "고등학교", "전라남도 무안군")
    rows = [
        {**MEAL_ROW, "MLSV_YMD": "20260723"},
        {**MEAL_ROW, "MLSV_YMD": "20260724"},
        {**MEAL_ROW, "MLSV_YMD": "20260725"},
    ]
    requested_pages: list[int] = []

    def paged_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        page = int(params["pIndex"])
        requested_pages.append(page)
        start = (page - 1) * 2
        payload = {
            "mealServiceDietInfo": [
                {"head": [{"list_total_count": 3}, {"RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"}}]},
                {"row": rows[start : start + 2]},
            ]
        }
        return FakeResponse(payload)

    monkeypatch.setattr("neis_meal_ai.neis.NEIS_PAGE_SIZE", 2)
    result = fetch_meals(school, "20260723", "20260725", http_get=paged_get)

    assert result == rows
    assert requested_pages == [1, 2]


def test_malformed_response_becomes_korean_neis_error() -> None:
    def malformed_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        return FakeResponse({"unexpected": []})

    with pytest.raises(NeisApiError, match="응답 형식"):
        search_school("남악고등학교", http_get=malformed_get)
