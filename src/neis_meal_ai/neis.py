"""NEIS 교육정보 개방 포털의 학교와 급식 API 경계."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

import requests


NEIS_BASE_URL = "https://open.neis.go.kr/hub"
NEIS_PAGE_SIZE = 1000
HttpGet = Callable[..., Any]


class NeisApiError(RuntimeError):
    """NEIS 조회 실패를 학생이 이해할 수 있는 한 종류의 오류로 표현한다."""


@dataclass(frozen=True)
class SchoolInfo:
    """급식 조회에 필요한 최소 학교 정보."""

    name: str
    office_code: str
    school_code: str
    school_kind: str = ""
    address: str = ""


def validate_date_range(start: str, end: str) -> tuple[date, date]:
    """YYYYMMDD 조회 범위를 검증하고 날짜 객체로 돌려준다."""

    try:
        start_date = datetime.strptime(start, "%Y%m%d").date()
        end_date = datetime.strptime(end, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError("날짜는 YYYYMMDD 형식의 실제 날짜여야 합니다.") from exc
    if start_date > end_date:
        raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
    if (end_date - start_date).days + 1 > 366:
        raise ValueError("조회 기간은 366일 이하여야 합니다.")
    return start_date, end_date


def _request_json(
    endpoint: str,
    params: dict[str, Any],
    *,
    http_get: HttpGet,
) -> dict[str, Any]:
    api_key = os.getenv("NEIS_API_KEY", "").strip()
    if api_key:
        params = {**params, "KEY": api_key}
    try:
        response = http_get(
            f"{NEIS_BASE_URL}/{endpoint}",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise NeisApiError("NEIS 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요.") from exc
    if not isinstance(payload, dict):
        raise NeisApiError("NEIS 응답 형식을 해석할 수 없습니다.")
    return payload


def _extract_rows(payload: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    root_result = payload.get("RESULT")
    if isinstance(root_result, dict):
        if root_result.get("CODE") == "INFO-200":
            return []
        raise NeisApiError(f"NEIS 오류: {root_result.get('MESSAGE', '알 수 없는 오류')}")

    dataset = payload.get(dataset_name)
    if not isinstance(dataset, list) or len(dataset) < 2:
        raise NeisApiError("NEIS 응답 형식을 해석할 수 없습니다.")

    head = dataset[0].get("head", []) if isinstance(dataset[0], dict) else []
    result = next(
        (item.get("RESULT") for item in head if isinstance(item, dict) and "RESULT" in item),
        None,
    )
    if isinstance(result, dict) and result.get("CODE") not in (None, "INFO-000"):
        if result.get("CODE") == "INFO-200":
            return []
        raise NeisApiError(f"NEIS 오류: {result.get('MESSAGE', '알 수 없는 오류')}")

    rows = dataset[1].get("row") if isinstance(dataset[1], dict) else None
    if rows is None:
        return []
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise NeisApiError("NEIS 응답 형식을 해석할 수 없습니다.")
    return rows


def _extract_total_count(payload: dict[str, Any], dataset_name: str) -> int | None:
    """NEIS head의 전체 행 수를 읽는다. 값이 없으면 단일 페이지로 처리한다."""

    dataset = payload.get(dataset_name)
    if not isinstance(dataset, list) or not dataset:
        return None
    head = dataset[0].get("head", []) if isinstance(dataset[0], dict) else []
    item = next(
        (part for part in head if isinstance(part, dict) and "list_total_count" in part),
        None,
    )
    if item is None:
        return None
    try:
        total = int(item["list_total_count"])
    except (TypeError, ValueError) as exc:
        raise NeisApiError("NEIS 전체 데이터 개수를 해석할 수 없습니다.") from exc
    return max(total, 0)


def search_school(name: str, *, http_get: HttpGet = requests.get) -> SchoolInfo:
    """정확한 학교명으로 교육청 코드와 학교 코드를 찾는다."""

    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("학교명을 입력하세요.")
    payload = _request_json(
        "schoolInfo",
        {
            "Type": "json",
            "pIndex": 1,
            "pSize": 100,
            "SCHUL_NM": cleaned_name,
        },
        http_get=http_get,
    )
    rows = _extract_rows(payload, "schoolInfo")
    exact_rows = [row for row in rows if str(row.get("SCHUL_NM", "")).strip() == cleaned_name]
    if not exact_rows:
        raise NeisApiError(f"'{cleaned_name}'와 정확히 일치하는 학교를 찾지 못했습니다.")
    row = exact_rows[0]
    return SchoolInfo(
        name=str(row.get("SCHUL_NM", "")),
        office_code=str(row.get("ATPT_OFCDC_SC_CODE", "")),
        school_code=str(row.get("SD_SCHUL_CODE", "")),
        school_kind=str(row.get("SCHUL_KND_SC_NM", "")),
        address=str(row.get("ORG_RDNMA", "")),
    )


def fetch_meals(
    school: SchoolInfo,
    start: str,
    end: str,
    *,
    http_get: HttpGet = requests.get,
) -> list[dict[str, Any]]:
    """학교와 기간에 해당하는 NEIS 급식 원본 행을 가져온다."""

    validate_date_range(start, end)
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _request_json(
            "mealServiceDietInfo",
            {
                "Type": "json",
                "pIndex": page,
                "pSize": NEIS_PAGE_SIZE,
                "ATPT_OFCDC_SC_CODE": school.office_code,
                "SD_SCHUL_CODE": school.school_code,
                "MLSV_FROM_YMD": start,
                "MLSV_TO_YMD": end,
            },
            http_get=http_get,
        )
        page_rows = _extract_rows(payload, "mealServiceDietInfo")
        rows.extend(page_rows)
        total_count = _extract_total_count(payload, "mealServiceDietInfo")
        if not page_rows or total_count is None or len(rows) >= total_count:
            break
        page += 1
    return rows
