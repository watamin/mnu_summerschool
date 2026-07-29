from __future__ import annotations

import json

import gradio as gr

import web_app
from jupyter_course.notebook_support import load_sample_frame


PROJECT_ROOT = web_app.PROJECT_ROOT


def test_web_app_builds_with_the_selected_data_source(monkeypatch) -> None:
    monkeypatch.setattr(
        web_app,
        "load_web_frame",
        lambda _root: (
            load_sample_frame(PROJECT_ROOT),
            "남악고등학교 NEIS 실제 수집 데이터 42일 (2026-06-01~2026-07-29, 2026-07-29 수집)",
        ),
    )
    demo = web_app.create_web_app()
    config_text = json.dumps(demo.get_config_file(), ensure_ascii=False)

    assert isinstance(demo, gr.Blocks)
    assert "남악고등학교 NEIS 실제 수집 데이터 42일" in config_text
    assert "미트볼로제파스타" not in config_text


def test_web_app_main_is_local_and_does_not_launch_during_import() -> None:
    options = web_app.local_launch_options()

    assert options["share"] is False
    assert options["server_name"] == "127.0.0.1"
    assert options["inbrowser"] is True
