"""NEIS 급식 원본 문자열을 학생이 분석할 수 있는 표로 바꾼다."""

from __future__ import annotations

import html
import math
import re
from datetime import datetime
from typing import Any, Iterable

import pandas as pd


ANALYSIS_COLUMNS = [
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

ALLERGY_LABELS = {
    1: "난류",
    2: "우유",
    3: "메밀",
    4: "땅콩",
    5: "대두",
    6: "밀",
    7: "고등어",
    8: "게",
    9: "새우",
    10: "돼지고기",
    11: "복숭아",
    12: "토마토",
    13: "아황산류",
    14: "호두",
    15: "닭고기",
    16: "쇠고기",
    17: "오징어",
    18: "조개류",
    19: "잣",
}

_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_ALLERGY_GROUP_RE = re.compile(r"\((\d{1,2}(?:\.\d{1,2})*)\)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _decoded(text: Any) -> str:
    return html.unescape(str(text or "")).strip()


def extract_allergy_codes(text: Any) -> tuple[int, ...]:
    """괄호 안 점으로 구분된 1~19 번호만 알레르기 코드로 읽는다."""

    codes: set[int] = set()
    for match in _ALLERGY_GROUP_RE.finditer(_decoded(text)):
        values = [int(value) for value in match.group(1).split(".")]
        if all(value in ALLERGY_LABELS for value in values):
            codes.update(values)
    return tuple(sorted(codes))


def split_dishes(text: Any) -> list[str]:
    """HTML 줄바꿈을 나누고 각 메뉴 끝의 알레르기 번호를 제거한다."""

    dishes: list[str] = []
    for part in _BR_RE.split(_decoded(text)):
        cleaned = _ALLERGY_GROUP_RE.sub("", part)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            dishes.append(cleaned)
    return dishes


def parse_calories(text: Any) -> float:
    """`927.7 Kcal` 같은 문자열에서 열량 숫자를 읽는다."""

    match = _NUMBER_RE.search(_decoded(text))
    return float(match.group()) if match else math.nan


def parse_nutrients(text: Any) -> dict[str, float]:
    """탄수화물·단백질·지방 수치를 추출하고 없으면 NaN으로 둔다."""

    decoded = _decoded(text)
    labels = {
        "carbs_g": "탄수화물",
        "protein_g": "단백질",
        "fat_g": "지방",
    }
    result: dict[str, float] = {}
    for key, korean_label in labels.items():
        match = re.search(
            rf"{korean_label}\s*\(g\)\s*:\s*(-?\d+(?:\.\d+)?)",
            decoded,
            flags=re.IGNORECASE,
        )
        result[key] = float(match.group(1)) if match else math.nan
    return result


def _date_text(value: Any) -> str | None:
    raw = str(value or "").strip()
    try:
        return datetime.strptime(raw, "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def meals_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """NEIS 급식 행을 고정 열을 가진 DataFrame으로 바꾼다."""

    records: list[dict[str, Any]] = []
    for row in rows:
        date_text = _date_text(row.get("MLSV_YMD"))
        dishes = split_dishes(row.get("DDISH_NM"))
        if not date_text or not dishes:
            continue
        nutrients = parse_nutrients(row.get("NTR_INFO"))
        records.append(
            {
                "date": date_text,
                "school_name": str(row.get("SCHUL_NM", "")).strip(),
                "meal_type": str(row.get("MMEAL_SC_NM", "")).strip(),
                "dishes": dishes,
                "menu_text": " ".join(dishes),
                "allergy_codes": extract_allergy_codes(row.get("DDISH_NM")),
                "calories": parse_calories(row.get("CAL_INFO")),
                **nutrients,
                "dish_count": len(dishes),
            }
        )
    return pd.DataFrame.from_records(records, columns=ANALYSIS_COLUMNS)
