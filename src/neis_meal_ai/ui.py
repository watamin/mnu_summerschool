"""Gradio 실행 환경에 따른 공개 범위를 명시적으로 결정한다."""

from __future__ import annotations

import sys


def is_google_colab() -> bool:
    return "google.colab" in sys.modules


def launch_options(*, is_colab: bool) -> dict[str, bool]:
    """Colab에서는 접속 가능한 임시 링크를, 로컬에서는 localhost를 사용한다."""

    return {
        "share": bool(is_colab),
        "debug": False,
        "show_error": True,
    }
