from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.build_jupyter_textbook import CHAPTER_FILES, build_textbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHAPTERS = (
    "00_시작하기.ipynb",
    "01_NEIS_API와_JSON.ipynb",
    "02_급식데이터_정리와_그래프.ipynb",
    "03_TFIDF_글자를_숫자로.ipynb",
    "04_유사도와_식단군집.ipynb",
    "05_개인추천_점수설계.ipynb",
    "06_Jupyter_추천화면.ipynb",
    "07_테스트와_모델카드.ipynb",
    "08_발표와_체험.ipynb",
)
REQUIRED_SECTIONS = (
    "이 장에서 배울 내용",
    "생각 열기",
    "핵심 용어",
    "개념 익히기",
    "활동 전 생각",
    "예상하기",
    "코드 살펴보기",
    "결과 해석하기",
    "탐구 활동",
    "확인 문제",
    "핵심 정리",
)
TEMPLATE_PHRASES = (
    "이번 장에서 할 수 있게 되는 것",
    "개념을 이야기로 이해하기",
    "멈춰서 생각하기",
    "✏️",
    "이미 만든 도구를 꺼냅니다",
    "오른쪽 결과를 왼쪽 이름표에 저장합니다",
)
BANNED_STUDENT_PHRASES = (
    "100점 평가표",
    "역할 순환표",
    "역할 배정표",
    "배점",
)
EXPECTED_RESULT_KEYS = {
    "00": {
        "environment_ready",
        "sample_rows",
        "python_version",
        "notebook_version",
        "packages_checked",
    },
    "01": {"source", "raw_rows", "first_keys"},
    "02": {"clean_rows", "columns", "chart_ready"},
    "03": {"similarities", "query"},
    "04": {"top_similar_menu", "cluster_names"},
    "05": {"recommendations", "top_score", "top_reason"},
    "06": {
        "widget_ready",
        "callback_rows",
        "callback_status",
        "callback_source",
    },
    "07": {"tests_passed", "model_card_complete"},
    "08": {"presentation_sections", "demo_checklist_ready"},
}


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])


def _execute_code_cells(path: Path) -> dict:
    notebook = _read_notebook(path)
    module = types.ModuleType(f"chapter_{path.stem}")
    namespace = module.__dict__
    old_cwd = Path.cwd()
    old_verify = os.environ.get("NEIS_JUPYTER_VERIFY")
    os.environ["NEIS_JUPYTER_VERIFY"] = "1"
    os.chdir(PROJECT_ROOT)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = str(cell.get("source", ""))
                exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(old_cwd)
        if old_verify is None:
            os.environ.pop("NEIS_JUPYTER_VERIFY", None)
        else:
            os.environ["NEIS_JUPYTER_VERIFY"] = old_verify
    return namespace


def test_build_textbook_creates_nine_jupyter_notebooks(tmp_path: Path) -> None:
    paths = build_textbook(tmp_path)

    assert CHAPTER_FILES == EXPECTED_CHAPTERS
    assert [path.name for path in paths] == list(EXPECTED_CHAPTERS)
    for path in paths:
        notebook = _read_notebook(path)
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert notebook["metadata"]["language_info"]["name"] == "python"
        assert "colab" not in notebook["metadata"]
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(cell_ids)
        assert len(cell_ids) == len(set(cell_ids))


def test_first_cell_bootstraps_from_the_real_chapter_directory() -> None:
    chapter_dir = PROJECT_ROOT / "jupyter_course" / "chapters"
    notebook = _read_notebook(chapter_dir / EXPECTED_CHAPTERS[0])
    first_code = next(
        str(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", first_code],
        cwd=chapter_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "급식 행 수: 5" in completed.stdout


def test_chapter_zero_teaches_manual_venv_and_jupyter_installation(
    tmp_path: Path,
) -> None:
    chapter_path = next(
        path for path in build_textbook(tmp_path) if path.name.startswith("00")
    )
    source = _source(_read_notebook(chapter_path))

    commands = (
        "py -3 --version",
        "py -3 -m venv .venv",
        r".\.venv\Scripts\python.exe -m pip install --upgrade pip",
        r".\.venv\Scripts\python.exe -m pip install -r requirements-jupyter.txt",
        r".\.venv\Scripts\python.exe -m notebook",
    )
    assert all(command in source for command in commands)
    assert "가상환경" in source
    assert "00_설치_준비.md" in source
    assert "각 패키지가 하는 일" in source
    assert all(f"### 설치 {number}단계" in source for number in range(1, 6))


def test_every_chapter_uses_textbook_editorial_structure(tmp_path: Path) -> None:
    for path in build_textbook(tmp_path):
        notebook = _read_notebook(path)
        source = _source(notebook)
        assert all(section in source for section in REQUIRED_SECTIONS), path.name
        assert not any(phrase in source for phrase in TEMPLATE_PHRASES), path.name
        assert not any(phrase in source for phrase in BANNED_STUDENT_PHRASES), path.name
        assert "demo.launch" not in source
        assert sum(cell["cell_type"] == "markdown" for cell in notebook["cells"]) >= 10
        assert sum(cell["cell_type"] == "code" for cell in notebook["cells"]) >= 3

        roles = [
            cell["metadata"].get("textbook_role")
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        assert roles[:7] == [
            "chapter-opener",
            "objectives",
            "opener",
            "terms",
            "concept",
            "pre-activity",
            "prediction",
        ]
        assert roles[-5:] == [
            "inquiry",
            "observation",
            "check",
            "answer",
            "summary",
        ]


def test_activity_numbers_are_sequential(tmp_path: Path) -> None:
    for path in build_textbook(tmp_path):
        notebook = _read_notebook(path)
        activity_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and cell["metadata"].get("textbook_role") == "activity"
        ]

        assert activity_cells, path.name
        assert [
            cell["metadata"].get("activity_number") for cell in activity_cells
        ] == list(range(1, len(activity_cells) + 1)), path.name
        assert all(
            str(cell["source"]).startswith(
                f"## 활동 {cell['metadata']['activity_number']}."
            )
            for cell in activity_cells
        ), path.name


def test_every_activity_has_a_chapter_specific_code_guide(tmp_path: Path) -> None:
    for path in build_textbook(tmp_path):
        notebook = _read_notebook(path)
        guide_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and cell["metadata"].get("textbook_role") == "code-guide"
        ]
        activity_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and cell["metadata"].get("textbook_role") == "activity"
        ]

        assert len(guide_cells) == len(activity_cells), path.name
        for guide_cell in guide_cells:
            source = str(guide_cell["source"])
            assert "`" in source, path.name
            assert "코드에 나온 변수 이름과 함수 이름" not in source, path.name


def test_personalization_inquiries_change_only_one_condition(
    tmp_path: Path,
) -> None:
    notebooks = {
        path.name[:2]: _read_notebook(path) for path in build_textbook(tmp_path)
    }

    chapter_05_exercise = next(
        str(cell["source"])
        for cell in notebooks["05"]["cells"]
        if "student-exercise" in cell["metadata"].get("tags", [])
    )
    assert 'practice_like = "치킨"' in chapter_05_exercise
    assert "practice_avoid" not in chapter_05_exercise
    assert 'avoids=("오이",)' in chapter_05_exercise
    assert 'preferred_types=("면", "디저트")' in chapter_05_exercise

    chapter_06_exercise = next(
        str(cell["source"])
        for cell in notebooks["06"]["cells"]
        if "student-exercise" in cell["metadata"].get("tags", [])
    )
    assert "practice_spice = 3" in chapter_06_exercise
    assert '"파스타, 피자", "오이", ["면"], practice_spice, []' in (
        " ".join(chapter_06_exercise.split())
    )


@pytest.mark.parametrize("chapter", [f"{number:02d}" for number in range(9)])
def test_chapter_executes_independently_with_expected_result(
    tmp_path: Path,
    chapter: str,
) -> None:
    path = next(path for path in build_textbook(tmp_path) if path.name.startswith(chapter))

    namespace = _execute_code_cells(path)
    result = namespace["chapter_result"]

    assert result["chapter"] == chapter
    assert EXPECTED_RESULT_KEYS[chapter].issubset(result)


def test_key_chapter_results_are_meaningful(tmp_path: Path) -> None:
    paths = build_textbook(tmp_path)
    results = {
        path.name[:2]: _execute_code_cells(path)["chapter_result"]
        for path in paths
    }

    assert results["00"]["sample_rows"] == 5
    assert results["01"]["raw_rows"] == 5
    assert results["02"]["clean_rows"] == 5
    assert len(results["03"]["similarities"]) == 5
    assert len(results["04"]["cluster_names"]) >= 2
    assert results["05"]["recommendations"] == 3
    assert results["06"]["widget_ready"] is True
    assert results["06"]["callback_rows"] == 3
    assert results["06"]["callback_status"] == "success"
    assert results["06"]["callback_source"] == "남악고 NEIS 예비 데이터"
    assert results["07"]["tests_passed"] >= 4
    assert results["07"]["model_card_complete"] is True
    assert results["08"]["presentation_sections"] == 8
    assert results["08"]["demo_checklist_ready"] is True


def test_api_chapter_teaches_live_meals_fallback_and_date_mismatch() -> None:
    chapter_path = PROJECT_ROOT / "jupyter_course" / "chapters" / EXPECTED_CHAPTERS[1]
    source = _source(_read_notebook(chapter_path))

    assert "load_classroom_frame" in source
    assert "실시간 우선·예비 자료 전환" in source
    assert "날짜가 겹치지 않을 때의 안내" in source


def test_widget_callback_shows_source_and_handles_bad_input(tmp_path: Path) -> None:
    chapter_path = next(
        path for path in build_textbook(tmp_path) if path.name.startswith("06")
    )
    namespace = _execute_code_cells(chapter_path)

    assert namespace["callback_state"]["status"] == "success"
    assert namespace["callback_state"]["rows"] == 3
    assert namespace["data_source"] in namespace["callback_state"]["message"]

    namespace["likes_widget"].value = "가, 나, 다, 라, 마, 바"
    namespace["on_recommend_clicked"](None)

    assert namespace["callback_state"]["status"] == "error"
    assert "최대 5개" in namespace["callback_state"]["message"]


def test_difficult_chapters_split_code_and_explain_line_groups() -> None:
    chapter_dir = PROJECT_ROOT / "jupyter_course" / "chapters"
    tfidf_source = _source(_read_notebook(chapter_dir / EXPECTED_CHAPTERS[3]))
    cluster_source = _source(_read_notebook(chapter_dir / EXPECTED_CHAPTERS[4]))
    widget_source = _source(_read_notebook(chapter_dir / EXPECTED_CHAPTERS[6]))

    assert "작은 TF-IDF를 직접 계산" in tfidf_source
    assert "math.log" in tfidf_source
    assert "tiny_counts[0][term] / sum(tiny_counts[0].values())" in tfidf_source
    assert "중심을 옮깁니다" in cluster_source
    assert "입력 위젯 만들기" in widget_source
    assert "버튼 콜백 연결과 화면 조립" in widget_source


def test_jupyter_guides_exist_without_role_or_score_tables() -> None:
    guide_paths = [
        PROJECT_ROOT / "jupyter_course" / "README.md",
        PROJECT_ROOT / "jupyter_course" / "교사용_운영안.md",
    ]

    for path in guide_paths:
        source = path.read_text(encoding="utf-8")
        assert not any(phrase in source for phrase in BANNED_STUDENT_PHRASES)
    teacher_guide = guide_paths[1].read_text(encoding="utf-8")
    assert sum(
        f"## {number}회차" in teacher_guide for number in range(1, 7)
    ) == 6
