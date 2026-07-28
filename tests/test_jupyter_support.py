from __future__ import annotations

from pathlib import Path

import pytest

from jupyter_course.notebook_support import (
    course_setup,
    evaluate_jupyter_environment,
    find_project_root,
    load_classroom_frame,
    load_sample_frame,
    load_sample_rows,
)
from neis_meal_ai.neis import NeisApiError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_course_setup_finds_root_and_loads_five_sample_rows() -> None:
    setup = course_setup(PROJECT_ROOT / "jupyter_course" / "chapters")

    assert setup["root"] == PROJECT_ROOT
    assert len(setup["rows"]) == 5
    assert len(setup["frame"]) == 5
    assert setup["source"] == "남악고 NEIS 예비 데이터"


def test_find_project_root_explains_invalid_location(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="프로젝트 폴더"):
        find_project_root(tmp_path)


def test_load_sample_rows_rejects_a_different_root(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="프로젝트 폴더"):
        load_sample_rows(tmp_path)


def test_load_sample_frame_has_expected_learning_columns() -> None:
    frame = load_sample_frame(PROJECT_ROOT)

    assert frame["date"].tolist()[0] == "2026-06-24"
    assert {
        "menu_text",
        "allergy_codes",
        "calories",
        "carbs_g",
        "protein_g",
        "fat_g",
    }.issubset(frame.columns)


def test_load_classroom_frame_uses_dated_fallback_when_neis_is_unavailable() -> None:
    def unavailable_fetcher(_school: str, _start: str, _end: str) -> list[dict]:
        raise NeisApiError("수업용 연결 실패")

    frame, source = load_classroom_frame(PROJECT_ROOT, fetcher=unavailable_fetcher)

    assert len(frame) == 5
    assert "예비 데이터" in source
    assert "수업용 연결 실패" in source


def test_load_classroom_frame_explains_non_overlapping_fallback_dates() -> None:
    def unavailable_fetcher(_school: str, _start: str, _end: str) -> list[dict]:
        raise NeisApiError("수업용 연결 실패")

    with pytest.raises(NeisApiError, match="예비 데이터 기간"):
        load_classroom_frame(
            PROJECT_ROOT,
            start="20260101",
            end="20260102",
            fetcher=unavailable_fetcher,
        )


def test_environment_check_accepts_the_project_venv_and_supported_versions() -> None:
    result = evaluate_jupyter_environment(
        PROJECT_ROOT,
        python_version=(3, 12),
        notebook_version="7.6.1",
        executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    )

    assert result == {"ready": True, "issues": []}


@pytest.mark.parametrize(
    ("python_version", "notebook_version", "executable", "expected_issue"),
    [
        ((3, 10), "7.6.1", ".venv/Scripts/python.exe", "Python 3.11 이상"),
        ((3, 12), "8.0.0", ".venv/Scripts/python.exe", "Notebook 7"),
        ((3, 12), "7.6.1", "C:/Python312/python.exe", ".venv Python"),
    ],
)
def test_environment_check_rejects_an_unsupported_or_wrong_kernel(
    python_version: tuple[int, int],
    notebook_version: str,
    executable: str,
    expected_issue: str,
) -> None:
    executable_path = (
        PROJECT_ROOT / executable
        if executable.startswith(".venv")
        else Path(executable)
    )

    result = evaluate_jupyter_environment(
        PROJECT_ROOT,
        python_version=python_version,
        notebook_version=notebook_version,
        executable=executable_path,
    )

    assert result["ready"] is False
    assert any(expected_issue in issue for issue in result["issues"])
