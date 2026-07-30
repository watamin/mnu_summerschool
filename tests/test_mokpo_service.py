from __future__ import annotations

from pathlib import Path
import socket

from gradio_client import Client
import pytest
import requests

import mokpo_service
from neis_meal_ai.student_profiles import StudentProfileStore


def test_local_and_lan_launch_options_have_no_login() -> None:
    local = mokpo_service.local_launch_options()
    lan = mokpo_service.launch_options(lan=True)

    assert local["server_name"] == "127.0.0.1"
    assert "auth" not in local
    assert local["share"] is False
    assert lan["server_name"] == "0.0.0.0"
    assert lan["share"] is False
    assert "auth" not in lan
    assert "auth_message" not in lan


def test_cli_parser_enables_lan_and_custom_paths(tmp_path: Path) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    args = mokpo_service.parse_args(
        [
            "--lan",
            "--db",
            str(database_path),
            "--port",
            "8899",
        ]
    )

    assert args.lan is True
    assert args.db == database_path
    assert args.port == 8899


def test_create_service_app_injects_real_profile_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_create_app(_dataset, _validation, *, profile_store):
        captured["profile_store"] = profile_store
        return "app"

    monkeypatch.setattr(mokpo_service, "create_mokpo_app", fake_create_app)

    result = mokpo_service.create_service_app(profile_db_path=tmp_path / "class.sqlite3")

    assert result == "app"
    assert isinstance(captured["profile_store"], StudentProfileStore)
    assert len(captured["profile_store"].food_pool) == 45
    assert captured["profile_store"].db_path == tmp_path / "class.sqlite3"


@pytest.mark.filterwarnings(
    "ignore:.*future.no_silent_downcasting.*:pandas.errors.Pandas4Warning"
)
@pytest.mark.filterwarnings(
    "ignore:The copy keyword is deprecated.*:pandas.errors.Pandas4Warning"
)
def test_real_gradio_without_login_loads_named_student_profile(
    tmp_path: Path,
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    app = mokpo_service.create_service_app(profile_db_path=tmp_path / "http.sqlite3")
    options = mokpo_service.launch_options(lan=True, port=port)
    options.update(
        server_name="127.0.0.1",
        inbrowser=False,
        prevent_thread_lock=True,
        quiet=True,
    )
    app.launch(**options)
    try:
        base_url = f"http://127.0.0.1:{port}"
        config_response = requests.get(f"{base_url}/config", timeout=10)
        client = Client(base_url, verbose=False)
        profile_message, profile_table = client.predict(
            "학생통합", api_name="/load_student_survey"
        )
        rated_table = dict(profile_table)
        rated_table["data"] = [
            [*row[:3], 4] for row in profile_table["data"]
        ]
        save_message, _ = client.predict(
            "학생통합",
            rated_table,
            api_name="/save_student_survey",
        )
        prediction_message, _ = client.predict(
            "학생통합",
            "mnu-2026-07-30-lunch",
            api_name="/predict_today_lunch",
        )
        other_message, _ = client.predict(
            "다른학생", api_name="/load_student_survey"
        )

        assert config_response.status_code == 200
        assert "학생통합" in profile_message
        assert "0/30" in profile_message
        assert len(profile_table["data"]) == 30
        assert "30/30" in save_message
        assert "학생통합 학생 예상" in prediction_message
        assert "다른학생" in other_message
        assert "0/30" in other_message
    finally:
        app.close()
