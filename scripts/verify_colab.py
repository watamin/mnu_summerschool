"""Jupyter 서버 없이 학생 노트북의 Python 셀을 순서대로 실행해 검증한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "우리학교_급식_AI_개인추천기_학생용.ipynb"


def verify_notebook(path: str | Path) -> dict[str, Any]:
    notebook_path = Path(path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    module_name = "__notebook__"
    notebook_module = types.ModuleType(module_name)
    namespace = notebook_module.__dict__
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = notebook_module
    old_mode = os.environ.get("NEIS_MEAL_AI_VERIFY")
    os.environ["NEIS_MEAL_AI_VERIFY"] = "1"
    executed = 0
    try:
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            try:
                exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)
            except Exception as exc:
                raise RuntimeError(f"노트북 코드 셀 {index} 실행 실패: {exc}") from exc
            executed += 1
    finally:
        if old_mode is None:
            os.environ.pop("NEIS_MEAL_AI_VERIFY", None)
        else:
            os.environ["NEIS_MEAL_AI_VERIFY"] = old_mode
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    result = namespace.get("recommendation_result")
    if result is None or result.empty:
        raise RuntimeError("추천 결과가 생성되지 않았습니다.")
    scores = result["추천 점수"]
    if not scores.between(0, 100).all():
        raise RuntimeError("추천 점수가 0~100 범위를 벗어났습니다.")
    return {
        "notebook": str(notebook_path),
        "code_cells_executed": executed,
        "recommendation_count": int(len(result)),
        "top_score": float(scores.iloc[0]),
        "data_source": namespace.get("data_source"),
        "model_card_complete": bool(namespace.get("model_card")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", nargs="?", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "verification" / "notebook_execution.json")
    args = parser.parse_args()
    report = verify_notebook(args.notebook)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
