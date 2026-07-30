"""본문 노트북을 새 커널로 실행하고 결과가 보이는 상태로 저장한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_jupyter_textbook import CHAPTER_FILES
from scripts.verify_jupyter_textbook import DEFAULT_CHAPTER_DIR, verify_textbook


def _hide_build_path(value, build_paths: tuple[str, ...]):
    """저장된 화면 결과에서 제작 PC의 절대 경로를 교재용 표시로 바꾼다."""

    if isinstance(value, str):
        for build_path in build_paths:
            value = value.replace(build_path, "<프로젝트 폴더>")
        return value
    if isinstance(value, list):
        return [_hide_build_path(item, build_paths) for item in value]
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _hide_build_path(item, build_paths)
    return value


def freeze_notebook_outputs(
    chapter_dir: str | Path = DEFAULT_CHAPTER_DIR,
    *,
    timeout: int = 180,
) -> list[dict[str, object]]:
    """모든 교재 노트북을 검증한 뒤 실행 번호와 출력을 원본에 기록한다."""

    target_dir = Path(chapter_dir).resolve()
    project_root = target_dir.parents[1]
    build_paths = (str(project_root), project_root.as_posix())
    with TemporaryDirectory(prefix="mnu_executed_notebooks_") as temp_dir:
        reports = verify_textbook(
            target_dir,
            timeout=timeout,
            executed_dir=Path(temp_dir),
        )
        for filename in CHAPTER_FILES:
            executed_path = Path(temp_dir) / filename
            notebook = nbformat.read(executed_path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = _hide_build_path(cell.get("outputs", []), build_paths)
            course_metadata = notebook.metadata.setdefault("jupyter_course", {})
            course_metadata["stored_outputs"] = True
            destination = target_dir / filename
            destination.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="교과서 노트북을 실행 결과가 보이는 상태로 저장합니다."
    )
    parser.add_argument("--chapters", type=Path, default=DEFAULT_CHAPTER_DIR)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    reports = freeze_notebook_outputs(args.chapters, timeout=args.timeout)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
