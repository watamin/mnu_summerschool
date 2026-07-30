"""완성된 우리 학교 급식 추천 웹 프로그램."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jupyter_course.notebook_support import load_web_frame  # noqa: E402
from neis_meal_ai.ui import build_demo, launch_options  # noqa: E402


def create_web_app():
    """실제 수집본을 우선 사용하고 예비 자료도 지원하는 추천기를 반환한다."""

    frame, data_source = load_web_frame(PROJECT_ROOT)
    return build_demo(frame, data_source)


def local_launch_options() -> dict[str, object]:
    """학교 PC 한 대 안에서만 접속하도록 실행 옵션을 정한다."""

    return launch_options(is_colab=False)


def main() -> int:
    create_web_app().launch(**local_launch_options())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
