from __future__ import annotations

import json
from pathlib import Path

from scripts.build_colab import build_notebook
from scripts.verify_colab import verify_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "namak_meals_sample.json"


def test_build_notebook_creates_valid_standalone_colab(tmp_path) -> None:
    output = tmp_path / "student.ipynb"

    built = build_notebook(output, SAMPLE_PATH)
    notebook = json.loads(built.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == "우리학교_급식_AI_개인추천기_학생용.ipynb"
    assert len(notebook["cells"]) >= 20
    assert all(cell["cell_type"] in {"markdown", "code"} for cell in notebook["cells"])
    assert any("EMBEDDED_SAMPLE_ROWS" in cell.get("source", "") for cell in notebook["cells"])
    assert any("build_colab_demo" in cell.get("source", "") for cell in notebook["cells"])
    all_source = "\n".join(cell.get("source", "") for cell in notebook["cells"])
    assert "demo.launch(**launch_options(is_colab=is_google_colab()))" in all_source
    assert "실제 알레르기·질병 정보는 입력하지" in all_source
    assert 'print("익명 취향 프로필 준비 완료:", validated_profile)' not in all_source


def test_verify_notebook_executes_recommendation_flow(tmp_path) -> None:
    notebook_path = build_notebook(tmp_path / "student.ipynb", SAMPLE_PATH)

    report = verify_notebook(notebook_path)

    assert report["code_cells_executed"] >= 9
    assert report["recommendation_count"] >= 1
    assert 0 <= report["top_score"] <= 100
    assert report["data_source"] == "내장 NEIS 예비 데이터"
