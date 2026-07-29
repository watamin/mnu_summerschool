from __future__ import annotations

import json

import gradio as gr

import web_app


def test_web_app_builds_with_the_offline_classroom_sample() -> None:
    demo = web_app.create_web_app()
    config_text = json.dumps(demo.get_config_file(), ensure_ascii=False)

    assert isinstance(demo, gr.Blocks)
    assert "남악고 NEIS 예비 데이터 5일" in config_text
    assert "미트볼로제파스타" not in config_text


def test_web_app_main_is_local_and_does_not_launch_during_import() -> None:
    options = web_app.local_launch_options()

    assert options["share"] is False
    assert options["server_name"] == "127.0.0.1"
    assert options["inbrowser"] is True
