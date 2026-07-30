"""Jupyter 교과서 본문과 부록을 각각 새 커널에서 실행해 검증한다."""

# noqa: SIZE_OK - 장별 결과 계약표와 새 커널 실행 검증은 하나의 검증 경계다.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_jupyter_textbook import CHAPTER_FILES


DEFAULT_CHAPTER_DIR = PROJECT_ROOT / "jupyter_course" / "chapters"
DEFAULT_EXECUTED_DIR = PROJECT_ROOT / "verification" / "jupyter_executed"
DEFAULT_REPORT = PROJECT_ROOT / "verification" / "jupyter_textbook_execution.json"
RESULT_PREFIX = "__CHAPTER_RESULT__="
REQUIRED_RESULT_KEYS = {
    "00": {
        "environment_ready",
        "sample_rows",
        "python_version",
        "notebook_version",
        "packages_checked",
    },
    "A": {
        "value_types",
        "second_dish",
        "school_name",
        "nested_date",
        "json_is_dict",
        "sample_rows",
        "sample_menu",
        "tutorial_steps",
    },
    "01": {"source", "raw_rows", "first_keys"},
    "02": {"clean_rows", "columns", "chart_ready"},
    "B": {
        "training_rows",
        "feature_name",
        "target_name",
        "labels",
        "selected_food",
        "one_hot",
        "decoded_food",
        "ones_count",
        "learned_weights",
        "predicted_rating",
        "actual_rating",
        "absolute_error",
        "multi_hot",
        "multi_hot_ones",
        "tutorial_steps",
    },
    "03": {
        "cross_document_comparison",
        "document_summaries",
        "document_term_details",
        "similarities",
        "query",
        "document_count",
        "document_top_terms",
        "document_sources_verified",
        "menu_similarity_ranking",
        "toy_tfidf_table",
    },
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


def _configure_notebook_event_loop() -> None:
    """Windows에서 Jupyter ZMQ가 요구하는 selector 이벤트 루프를 사용한다."""

    import asyncio
    import sys

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        if not isinstance(
            asyncio.get_event_loop_policy(),
            asyncio.WindowsSelectorEventLoopPolicy,
        ):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _normalize_curve_keys(connection_info: dict[str, Any]) -> dict[str, Any]:
    """jupyter_client 8.9의 문자열 Curve 키를 비동기 클라이언트용 bytes로 바꾼다."""

    normalized = dict(connection_info)
    for key in ("curve_publickey", "curve_secretkey"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.encode("ascii")
    return normalized


def _result_from_outputs(notebook: Any, path: Path) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            for line in str(output.get("text", "")).splitlines():
                if line.startswith(RESULT_PREFIX):
                    result = json.loads(line.removeprefix(RESULT_PREFIX))
    if result is None:
        raise RuntimeError(f"{path.name}에서 장 결과 표식을 찾지 못했습니다.")
    return result


def _validate_chapter_result(
    chapter: str,
    result: dict[str, Any],
    path: Path,
) -> None:
    """장별 결과가 단순 표식이 아니라 핵심 기능의 성공을 증명하는지 확인한다."""

    if result.get("chapter") != chapter:
        raise RuntimeError(
            f"{path.name}의 장 번호 결과가 {result.get('chapter')}로 잘못되었습니다."
        )
    required = REQUIRED_RESULT_KEYS[chapter]
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(
            f"{path.name}의 필수 결과가 없습니다: {', '.join(missing)}"
        )

    validators = {
        "00": lambda item: item["environment_ready"] is True
        and item["sample_rows"] >= 1
        and tuple(int(part) for part in item["python_version"].split(".")[:2])
        >= (3, 11)
        and item["notebook_version"].split(".", 1)[0] == "7"
        and item["packages_checked"] >= 8,
        "A": lambda item: item["value_types"] == ["str", "float", "bool"]
        and item["second_dish"] == "미트볼로제파스타"
        and item["school_name"] == "남악고등학교"
        and item["nested_date"] == "20260624"
        and item["json_is_dict"] is True
        and item["sample_rows"] == 5
        and "미트볼로제파스타" in item["sample_menu"]
        and item["tutorial_steps"] == 6,
        "01": lambda item: bool(item["source"])
        and item["raw_rows"] >= 1
        and bool(item["first_keys"]),
        "02": lambda item: item["clean_rows"] >= 1
        and bool(item["columns"])
        and item["chart_ready"] is True,
        "B": lambda item: item["training_rows"] == 8
        and item["feature_name"] == "menu"
        and item["target_name"] == "rating"
        and item["labels"] == ["밥", "국", "김치", "돈까스"]
        and item["selected_food"] == "김치"
        and item["one_hot"] == [0, 0, 1, 0]
        and item["decoded_food"] == item["selected_food"]
        and item["ones_count"] == item["one_hot"].count(1) == 1
        and item["learned_weights"] == [4.5, 3.5, 5.0, 4.0]
        and item["predicted_rating"] == 5.0
        and item["actual_rating"] == 4.0
        and item["absolute_error"]
        == abs(item["actual_rating"] - item["predicted_rating"])
        == 1.0
        and item["multi_hot"] == [1, 0, 1, 1]
        and item["multi_hot_ones"] == item["multi_hot"].count(1) == 3
        and item["tutorial_steps"] == 6,
        "03": lambda item: bool(item["similarities"])
        and bool(item["query"])
        and len(item["document_summaries"]) == 3
        and set(item["document_term_details"])
        == {"신경망 소개", "컴퓨터 비전 소개", "윤리적이고 책임 있는 AI"}
        and all(len(rows) >= 5 for rows in item["document_term_details"].values())
        and len(item["cross_document_comparison"]) == 12
        and len(item["menu_similarity_ranking"]) >= 1
        and len(item["toy_tfidf_table"]) == 6,
        "04": lambda item: bool(item["top_similar_menu"])
        and len(item["cluster_names"]) >= 2,
        "05": lambda item: item["recommendations"] >= 1
        and isinstance(item["top_score"], (int, float))
        and bool(item["top_reason"]),
        "06": lambda item: item["widget_ready"] is True
        and item["callback_rows"] >= 1
        and item["callback_status"] == "success"
        and bool(item["callback_source"]),
        "07": lambda item: item["tests_passed"] >= 4
        and item["model_card_complete"] is True,
        "08": lambda item: item["presentation_sections"] == 8
        and item["demo_checklist_ready"] is True,
    }
    if not validators[chapter](result):
        detail = (
            "실제 콜백 실행 결과가 아닙니다."
            if chapter == "06"
            else "결과 값이 학습 목표를 증명하지 못합니다."
        )
        raise RuntimeError(f"{path.name}: {detail}")


def verify_textbook(
    chapter_dir: str | Path = DEFAULT_CHAPTER_DIR,
    *,
    timeout: int = 180,
    executed_dir: str | Path = DEFAULT_EXECUTED_DIR,
) -> list[dict[str, Any]]:
    """교과서의 모든 장을 독립 커널에서 실행하고 장별 보고서를 반환한다."""

    _configure_notebook_event_loop()
    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client.manager import AsyncKernelManager
    except ImportError as exc:
        raise RuntimeError(
            "Jupyter 검증 패키지가 없습니다. "
            ".venv\\Scripts\\python.exe -m pip install -r "
            "requirements-jupyter.txt 명령을 실행하세요."
        ) from exc

    class CurveCompatibleAsyncKernelManager(AsyncKernelManager):
        def client(self, **kwargs: Any):
            client_kwargs = _normalize_curve_keys(
                self.get_connection_info(session=True)
            )
            client_kwargs.update(
                {
                    "connection_file": self.connection_file,
                    "parent": self,
                }
            )
            client_kwargs.update(kwargs)
            return self.client_factory(**client_kwargs)

    source_dir = Path(chapter_dir)
    paths = [source_dir / filename for filename in CHAPTER_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Jupyter 교과서 파일 {len(CHAPTER_FILES)}개가 모두 필요합니다. "
            f"없는 파일: {', '.join(missing)}"
        )

    output_dir = Path(executed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    previous_verify = os.environ.get("NEIS_JUPYTER_VERIFY")
    os.environ["NEIS_JUPYTER_VERIFY"] = "1"
    try:
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            chapter = str(notebook.metadata["jupyter_course"]["chapter"])
            started = time.perf_counter()
            try:
                executed = NotebookClient(
                    notebook,
                    timeout=timeout,
                    kernel_name="python3",
                    kernel_manager_class=CurveCompatibleAsyncKernelManager,
                    resources={"metadata": {"path": str(path.parent)}},
                ).execute(
                    cwd=str(path.parent),
                    transport_encryption="auto",
                )
            except Exception as exc:
                raise RuntimeError(f"{path.name} 새 커널 실행 실패: {exc}") from exc
            elapsed = round(time.perf_counter() - started, 2)
            result = _result_from_outputs(executed, path)
            _validate_chapter_result(chapter, result, path)
            executed_path = output_dir / path.name
            nbformat.write(executed, executed_path)
            reports.append(
                {
                    "chapter": chapter,
                    "filename": path.name,
                    "status": "PASS",
                    "seconds": elapsed,
                    "code_cells": sum(
                        cell.cell_type == "code" for cell in executed.cells
                    ),
                    "markdown_cells": sum(
                        cell.cell_type == "markdown" for cell in executed.cells
                    ),
                    "result_keys": sorted(result.keys()),
                }
            )
    finally:
        if previous_verify is None:
            os.environ.pop("NEIS_JUPYTER_VERIFY", None)
        else:
            os.environ["NEIS_JUPYTER_VERIFY"] = previous_verify
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=Path, default=DEFAULT_CHAPTER_DIR)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    reports = verify_textbook(args.chapters, timeout=args.timeout)
    summary = {
        "chapters": reports,
        "passed": sum(report["status"] == "PASS" for report in reports),
        "failed": sum(report["status"] != "PASS" for report in reports),
        "total_seconds": round(sum(report["seconds"] for report in reports), 2),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
