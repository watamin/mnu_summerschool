"""각 Jupyter 장이 같은 프로젝트 경로와 예비 데이터를 쓰게 돕는다."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


SAMPLE_RELATIVE_PATH = Path("data") / "namak_meals_sample.json"
LIVE_RELATIVE_PATH = Path("data") / "namak_meals_live.json"
EXPECTED_SCHOOL = "남악고등학교"
EXPECTED_OFFICE_CODE = "Q10"
EXPECTED_SCHOOL_CODE = "7140272"
REQUIRED_LIVE_ROW_FIELDS = {
    "ATPT_OFCDC_SC_CODE",
    "SD_SCHUL_CODE",
    "SCHUL_NM",
    "MMEAL_SC_NM",
    "MLSV_YMD",
    "DDISH_NM",
    "CAL_INFO",
    "NTR_INFO",
}
DEFAULT_START = "20260624"
DEFAULT_END = "20260630"
ClassroomFetcher = Callable[[str, str, str], list[dict]]


def find_project_root(start: Path | None = None) -> Path:
    """현재 위치의 부모에서 NEIS 급식 AI 프로젝트 루트를 찾는다."""

    candidate = Path(start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for location in (candidate, *candidate.parents):
        if (
            (location / "src" / "neis_meal_ai").is_dir()
            and (location / SAMPLE_RELATIVE_PATH).is_file()
        ):
            return location
    raise RuntimeError(
        "NEIS 급식 AI 프로젝트 폴더를 찾지 못했습니다. "
        "프로젝트 최상위 폴더에서 .venv\\Scripts\\python.exe -m notebook 명령으로 "
        "Jupyter를 다시 시작하세요."
    )


def evaluate_jupyter_environment(
    root: Path,
    *,
    python_version: tuple[int, int],
    notebook_version: str,
    executable: str | Path,
) -> dict[str, object]:
    """0장에서 지원 버전과 프로젝트 전용 가상환경 사용 여부를 판정한다."""

    project_root = find_project_root(root)
    issues: list[str] = []

    if python_version < (3, 11):
        issues.append("Python 3.11 이상이 필요합니다.")
    if notebook_version.split(".", 1)[0] != "7":
        issues.append("Jupyter Notebook 7 버전이 필요합니다.")

    expected_parent = (project_root / ".venv" / "Scripts").resolve()
    actual_parent = Path(executable).resolve().parent
    if os.path.normcase(str(actual_parent)) != os.path.normcase(str(expected_parent)):
        issues.append("프로젝트의 .venv Python으로 Jupyter를 실행해야 합니다.")

    return {"ready": not issues, "issues": issues}


def _enable_project_imports(root: Path) -> None:
    source_path = str(root / "src")
    root_path = str(root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


def load_sample_rows(root: Path | None = None) -> list[dict[str, Any]]:
    """내장된 실제 남악고 NEIS 공개 급식 행 5개를 읽고 구조를 검증한다."""

    project_root = find_project_root(root)
    try:
        payload = json.loads(
            (project_root / SAMPLE_RELATIVE_PATH).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("수업용 예비 급식 파일을 읽을 수 없습니다.") from exc

    metadata = payload.get("metadata", {})
    rows = payload.get("rows")
    if metadata.get("school_name") != EXPECTED_SCHOOL:
        raise RuntimeError("예비 급식 파일의 학교명이 남악고등학교가 아닙니다.")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("예비 급식 파일의 행 구조가 올바르지 않습니다.")
    return rows


def load_sample_frame(root: Path | None = None):
    """예비 급식 행을 분석 가능한 Pandas 표로 바꾼다."""

    project_root = find_project_root(root)
    _enable_project_imports(project_root)
    from neis_meal_ai.cleaning import meals_to_frame

    frame = meals_to_frame(load_sample_rows(project_root))
    if frame.empty:
        raise RuntimeError("수업용 예비 급식 표가 비어 있습니다.")
    return frame


def _read_live_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """실제 수집본의 공개 메타데이터와 급식 행을 검증한다."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("실제 NEIS 수집본을 읽을 수 없습니다.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("실제 NEIS 수집본의 최상위 구조가 올바르지 않습니다.")
    metadata = payload.get("metadata")
    rows = payload.get("rows")
    if not isinstance(metadata, dict) or metadata.get("snapshot_kind") != "live":
        raise RuntimeError("실제 NEIS 수집본의 메타데이터가 올바르지 않습니다.")
    if metadata.get("school_name") != EXPECTED_SCHOOL:
        raise RuntimeError("실제 NEIS 수집본의 학교명이 남악고등학교가 아닙니다.")
    if metadata.get("office_code") != EXPECTED_OFFICE_CODE:
        raise RuntimeError("실제 NEIS 수집본의 교육청 코드가 전라남도교육청이 아닙니다.")
    if metadata.get("school_code") != EXPECTED_SCHOOL_CODE:
        raise RuntimeError("실제 NEIS 수집본의 학교 코드가 남악고등학교가 아닙니다.")
    fetched_at = metadata.get("fetched_at_utc")
    if not isinstance(fetched_at, str):
        raise RuntimeError("실제 NEIS 수집본의 수집 시각이 올바르지 않습니다.")
    try:
        fetched_datetime = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("실제 NEIS 수집본의 수집 시각이 올바르지 않습니다.") from exc
    if fetched_datetime.utcoffset() != timedelta(0):
        raise RuntimeError("실제 NEIS 수집본의 수집 시각은 UTC여야 합니다.")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("실제 NEIS 수집본의 급식 행 구조가 올바르지 않습니다.")
    if metadata.get("row_count") != len(rows):
        raise RuntimeError("실제 NEIS 수집본의 행 수 기록이 맞지 않습니다.")
    for row in rows:
        if not REQUIRED_LIVE_ROW_FIELDS.issubset(row):
            raise RuntimeError("실제 NEIS 수집본에 필요한 급식 항목이 없습니다.")
        if (
            row["ATPT_OFCDC_SC_CODE"] != EXPECTED_OFFICE_CODE
            or row["SD_SCHUL_CODE"] != EXPECTED_SCHOOL_CODE
            or row["SCHUL_NM"] != EXPECTED_SCHOOL
            or row["MMEAL_SC_NM"] != "중식"
        ):
            raise RuntimeError("실제 NEIS 수집본에 다른 학교나 식사 종류가 섞여 있습니다.")
        if not all(isinstance(row[field], str) for field in REQUIRED_LIVE_ROW_FIELDS):
            raise RuntimeError("실제 NEIS 수집본의 급식 항목 형식이 올바르지 않습니다.")
        if not row["DDISH_NM"].strip():
            raise RuntimeError("실제 NEIS 수집본에 메뉴가 비어 있습니다.")
        if re.fullmatch(r"[0-9]{8}", row["MLSV_YMD"]) is None:
            raise RuntimeError("실제 NEIS 수집본의 날짜 형식이 올바르지 않습니다.")
        try:
            datetime.strptime(row["MLSV_YMD"], "%Y%m%d")
        except ValueError as exc:
            raise RuntimeError("실제 NEIS 수집본의 날짜 형식이 올바르지 않습니다.") from exc
    return rows, metadata


def _date_span(frame) -> str:
    dates = sorted(str(value) for value in frame["date"] if str(value))
    return f"{dates[0]}~{dates[-1]}" if dates else "날짜 없음"


def load_web_frame(
    root: Path | None = None,
    *,
    live_path: str | Path | None = None,
):
    """웹 앱용 실제 수집본을 우선 읽고 문제가 있으면 예비 자료를 사용한다."""

    project_root = find_project_root(root)
    _enable_project_imports(project_root)
    from neis_meal_ai.cleaning import meals_to_frame

    selected_live_path = Path(live_path) if live_path else project_root / LIVE_RELATIVE_PATH
    live_error = False
    if selected_live_path.is_file():
        try:
            rows, metadata = _read_live_snapshot(selected_live_path)
            frame = meals_to_frame(rows)
            if frame.empty:
                raise RuntimeError("실제 NEIS 수집본을 표로 바꿀 수 없습니다.")
            fetched_date = str(metadata.get("fetched_at_utc", ""))[:10] or "수집일 미상"
            source = (
                f"{metadata['school_name']} NEIS 실제 수집 데이터 {len(frame)}일 "
                f"({_date_span(frame)}, {fetched_date} 수집)"
            )
            return frame, source
        except (RuntimeError, ValueError, TypeError, KeyError):
            live_error = True

    frame = load_sample_frame(project_root)
    source = f"남악고 NEIS 예비 데이터 {len(frame)}일 ({_date_span(frame)})"
    if live_error:
        source += " · 실제 수집본 오류로 예비 자료 사용"
    return frame, source


def load_classroom_frame(
    root: Path | None = None,
    *,
    school_name: str = EXPECTED_SCHOOL,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    fetcher: ClassroomFetcher | None = None,
):
    """NEIS를 먼저 조회하고 실패하면 같은 기간의 남악고 예비 자료를 사용한다."""

    project_root = find_project_root(root)
    _enable_project_imports(project_root)
    from neis_meal_ai.service import load_meal_frame

    return load_meal_frame(
        school_name,
        start,
        end,
        project_root / SAMPLE_RELATIVE_PATH,
        fetcher=fetcher,
    )


def course_setup(start: Path | None = None) -> dict[str, object]:
    """한 장을 독립 실행하는 데 필요한 루트·원본·정제 표를 준비한다."""

    root = find_project_root(start)
    _enable_project_imports(root)
    rows = load_sample_rows(root)
    frame = load_sample_frame(root)
    return {
        "root": root,
        "rows": rows,
        "frame": frame,
        "source": "남악고 NEIS 예비 데이터",
    }
