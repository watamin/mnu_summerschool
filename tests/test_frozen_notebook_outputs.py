from __future__ import annotations

import json
from pathlib import Path

from scripts.build_jupyter_textbook import CHAPTER_FILES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = PROJECT_ROOT / "jupyter_course" / "chapters"


def _notebook(filename: str) -> dict:
    return json.loads((CHAPTER_DIR / filename).read_text(encoding="utf-8"))


def _all_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _all_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _all_strings(item)]
    return []


def test_tracked_notebooks_store_outputs_for_reading_without_execution() -> None:
    for filename in CHAPTER_FILES:
        notebook = _notebook(filename)
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]

        assert code_cells, filename
        assert all(cell.get("execution_count") is not None for cell in code_cells), filename
        assert all(cell.get("outputs") for cell in code_cells), filename
        assert notebook["metadata"]["jupyter_course"]["stored_outputs"] is True

        serialized_outputs = json.dumps(
            [cell.get("outputs", []) for cell in code_cells],
            ensure_ascii=False,
        )
        assert "__CHAPTER_RESULT__=" in serialized_outputs, filename


def test_tfidf_notebook_stores_document_scores_and_menu_ranking() -> None:
    notebook = _notebook("03_TFIDF_글자를_숫자로.ipynb")
    outputs = json.dumps(
        [cell.get("outputs", []) for cell in notebook["cells"]],
        ensure_ascii=False,
    )

    assert "뉴런" in outputs
    assert "0.0342" in outputs
    assert "이미지" in outputs
    assert "0.0407" in outputs
    assert "책임" in outputs
    assert "0.026" in outputs
    assert "2026-06-24" in outputs


def test_graph_notebook_stores_a_rendered_chart() -> None:
    notebook = _notebook("02_급식데이터_정리와_그래프.ipynb")
    mime_types = {
        mime_type
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        for mime_type in output.get("data", {})
    }

    assert "image/png" in mime_types


def test_stored_outputs_do_not_expose_the_build_computer_path() -> None:
    build_paths = {str(PROJECT_ROOT), PROJECT_ROOT.as_posix()}

    for filename in CHAPTER_FILES:
        notebook = _notebook(filename)
        output_text = "\n".join(
            _all_strings(
                [cell.get("outputs", []) for cell in notebook["cells"]]
            )
        )
        assert all(path not in output_text for path in build_paths), filename
