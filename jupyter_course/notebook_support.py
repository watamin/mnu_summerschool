"""각 Jupyter 장이 같은 프로젝트 경로와 예비 데이터를 쓰게 돕는다."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


SAMPLE_RELATIVE_PATH = Path("data") / "namak_meals_sample.json"
EXPECTED_SCHOOL = "남악고등학교"
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
