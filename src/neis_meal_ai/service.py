"""실시간 NEIS 조회, 예비 데이터, 추천 결과를 한 흐름으로 연결한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .cleaning import meals_to_frame
from .neis import NeisApiError, fetch_meals, search_school, validate_date_range
from .recommender import PreferenceProfile, SAFETY_NOTICE, recommend_menus


Fetcher = Callable[[str, str, str], list[dict]]
DEFAULT_SCHOOL = "남악고등학교"


def _live_fetcher(school_name: str, start: str, end: str) -> list[dict]:
    school = search_school(school_name)
    return fetch_meals(school, start, end)


def _load_fallback(path: Path, school_name: str) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NeisApiError("예비 급식 데이터도 읽지 못했습니다.") from exc
    metadata = payload.get("metadata", {})
    rows = payload.get("rows")
    if metadata.get("school_name") != school_name or not isinstance(rows, list):
        raise NeisApiError("예비 급식 데이터의 학교명 또는 구조가 올바르지 않습니다.")
    return rows


def _filter_rows_by_date(rows: list[dict], start: str, end: str) -> list[dict]:
    return [row for row in rows if start <= str(row.get("MLSV_YMD", "")) <= end]


def load_meal_frame(
    school_name: str,
    start: str,
    end: str,
    fallback_path: str | Path,
    *,
    fetcher: Fetcher | None = None,
) -> tuple[pd.DataFrame, str]:
    """실시간 급식을 우선 사용하고 연결 실패 때만 남악고 예비 데이터로 전환한다."""

    validate_date_range(start, end)
    cleaned_school = school_name.strip()
    if not cleaned_school:
        raise ValueError("학교명을 입력하세요.")
    selected_fetcher = fetcher or _live_fetcher
    try:
        rows = selected_fetcher(cleaned_school, start, end)
    except NeisApiError as exc:
        if cleaned_school != DEFAULT_SCHOOL:
            raise
        fallback_rows = _load_fallback(Path(fallback_path), cleaned_school)
        rows = _filter_rows_by_date(fallback_rows, start, end)
        if not rows:
            available_dates = sorted(
                str(row.get("MLSV_YMD", ""))
                for row in fallback_rows
                if str(row.get("MLSV_YMD", ""))
            )
            period = (
                f"{available_dates[0]}~{available_dates[-1]}"
                if available_dates
                else "확인할 수 없음"
            )
            raise NeisApiError(
                f"실시간 조회에 실패했고 요청 기간({start}~{end})과 겹치는 예비 데이터가 없습니다. "
                f"예비 데이터 기간: {period}. 수업용 시연은 이 기간으로 조회하세요."
            ) from exc
        frame = meals_to_frame(rows)
        if frame.empty:
            raise NeisApiError("예비 급식 데이터가 비어 있습니다.") from exc
        return frame, f"남악고 예비 데이터 사용 · 실시간 조회 사유: {exc}"

    frame = meals_to_frame(rows)
    if frame.empty:
        raise ValueError("선택한 기간에 급식 데이터가 없습니다. 다른 기간을 선택하세요.")
    return frame, "실시간 NEIS 데이터 사용"


def _csv_terms(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text or "").split(",") if part.strip())


def run_recommendation(
    frame: pd.DataFrame,
    *,
    likes_text: str,
    avoids_text: str,
    preferred_types: Iterable[str],
    spice_level: int,
    allergy_codes: Iterable[int],
    top_n: int = 3,
) -> tuple[str, pd.DataFrame]:
    """화면 입력을 익명 프로필로 바꾸고 한국어 결과 표를 만든다."""

    profile = PreferenceProfile(
        likes=_csv_terms(likes_text),
        avoids=_csv_terms(avoids_text),
        preferred_types=tuple(preferred_types or ()),
        spice_level=int(spice_level),
        allergy_codes=tuple(int(code) for code in (allergy_codes or ())),
    )
    result = recommend_menus(frame, profile, top_n=top_n)
    excluded_count = int(result.attrs.get("excluded_count", 0))
    if result.empty:
        summary = (
            f"선택한 알레르기 주의 번호 때문에 모든 메뉴가 제외되었습니다. "
            f"제외된 메뉴: {excluded_count}개\n\n{SAFETY_NOTICE}"
        )
        return summary, pd.DataFrame(
            columns=["순위", "날짜", "추천 점수", "메뉴", "추천 이유", "식단 군집", "알레르기 번호"]
        )

    table = pd.DataFrame(
        {
            "순위": range(1, len(result) + 1),
            "날짜": result["date"],
            "추천 점수": result["score"],
            "메뉴": result["menu_text"],
            "추천 이유": result["reason"],
            "식단 군집": result["cluster_name"],
            "알레르기 번호": result["allergy_codes"].apply(
                lambda codes: ", ".join(str(code) for code in codes) if codes else "없음"
            ),
        }
    )
    summary = (
        f"취향을 비교해 {len(table)}개 메뉴를 추천했습니다. "
        f"알레르기 주의 번호로 제외된 메뉴는 {excluded_count}개입니다.\n\n{SAFETY_NOTICE}"
    )
    return summary, table
