from __future__ import annotations

from pathlib import Path
import socket

from gradio_client import Client
import pytest
import requests

import mokpo_service
from neis_meal_ai.student_profiles import StudentProfileStore


def test_password_file_must_exist_and_contain_a_value(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(RuntimeError, match="비밀번호 파일"):
        mokpo_service.load_shared_password(missing)

    blank = tmp_path / "blank.txt"
    blank.write_text("   \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="비어"):
        mokpo_service.load_shared_password(blank)


def test_password_file_accepts_plain_text_or_named_assignment(tmp_path: Path) -> None:
    plain = tmp_path / "plain.txt"
    plain.write_text("교실테스트암호\n", encoding="utf-8")
    assigned = tmp_path / "assigned.txt"
    assigned.write_text("password=두번째테스트암호\n", encoding="utf-8")

    assert mokpo_service.load_shared_password(plain) == "교실테스트암호"
    assert mokpo_service.load_shared_password(assigned) == "두번째테스트암호"


def test_authenticator_accepts_safe_student_name_and_exact_password() -> None:
    authenticate = mokpo_service.build_authenticator("교실테스트암호")

    assert authenticate("학생A", "교실테스트암호") is True
    assert authenticate("학생A", "틀린암호") is False
    assert authenticate("학생/../../1", "교실테스트암호") is False
    assert authenticate("", "교실테스트암호") is False


def test_local_and_lan_launch_options_have_different_network_boundaries() -> None:
    local = mokpo_service.local_launch_options()
    lan = mokpo_service.launch_options(lan=True, password="교실테스트암호")

    assert local["server_name"] == "127.0.0.1"
    assert "auth" not in local
    assert local["share"] is False
    assert lan["server_name"] == "0.0.0.0"
    assert lan["share"] is False
    assert callable(lan["auth"])
    assert lan["auth"]("학생A", "교실테스트암호") is True
    assert "이름" in lan["auth_message"]
    with pytest.raises(ValueError, match="비밀번호"):
        mokpo_service.launch_options(lan=True, password=None)


def test_cli_parser_enables_lan_and_custom_paths(tmp_path: Path) -> None:
    password_path = tmp_path / "password.txt"
    database_path = tmp_path / "profiles.sqlite3"
    args = mokpo_service.parse_args(
        [
            "--lan",
            "--password-file",
            str(password_path),
            "--db",
            str(database_path),
            "--port",
            "8899",
        ]
    )

    assert args.lan is True
    assert args.password_file == password_path
    assert args.db == database_path
    assert args.port == 8899


def test_create_service_app_injects_real_profile_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_create_app(_dataset, _validation, *, profile_store, classroom_mode):
        captured["profile_store"] = profile_store
        captured["classroom_mode"] = classroom_mode
        return "app"

    monkeypatch.setattr(mokpo_service, "create_mokpo_app", fake_create_app)

    result = mokpo_service.create_service_app(
        profile_db_path=tmp_path / "class.sqlite3", classroom_mode=True
    )

    assert result == "app"
    assert isinstance(captured["profile_store"], StudentProfileStore)
    assert len(captured["profile_store"].food_pool) == 45
    assert captured["profile_store"].db_path == tmp_path / "class.sqlite3"
    assert captured["classroom_mode"] is True


@pytest.mark.filterwarnings(
    "ignore:.*future.no_silent_downcasting.*:pandas.errors.Pandas4Warning"
)
@pytest.mark.filterwarnings(
    "ignore:The copy keyword is deprecated.*:pandas.errors.Pandas4Warning"
)
def test_real_gradio_login_protects_api_and_loads_named_student_profile(
    tmp_path: Path,
) -> None:
    password = "통합테스트암호"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    app = mokpo_service.create_service_app(
        profile_db_path=tmp_path / "http.sqlite3", classroom_mode=True
    )
    options = mokpo_service.launch_options(lan=True, password=password, port=port)
    options.update(
        server_name="127.0.0.1",
        inbrowser=False,
        prevent_thread_lock=True,
        quiet=True,
    )
    app.launch(**options)
    try:
        base_url = f"http://127.0.0.1:{port}"
        unauthenticated_config = requests.get(f"{base_url}/config", timeout=10)
        wrong_login = requests.post(
            f"{base_url}/login",
            data={"username": "학생통합", "password": "틀린암호"},
            timeout=10,
        )
        client = Client(
            base_url,
            auth=("학생통합", password),
            verbose=False,
        )
        profile_message, profile_table = client.predict(
            "무시되어야할이름", api_name="/load_student_survey"
        )
        rated_table = dict(profile_table)
        rated_table["data"] = [
            [*row[:3], 4] for row in profile_table["data"]
        ]
        save_message, _ = client.predict(
            "무시되어야할이름",
            rated_table,
            api_name="/save_student_survey",
        )
        prediction_message, _ = client.predict(
            "무시되어야할이름",
            "mnu-2026-07-30-lunch",
            api_name="/predict_today_lunch",
        )
        other_client = Client(
            base_url,
            auth=("다른학생", password),
            verbose=False,
        )
        other_message, _ = other_client.predict(
            "학생통합", api_name="/load_student_survey"
        )

        assert unauthenticated_config.status_code == 401
        assert wrong_login.status_code == 400
        assert wrong_login.json()["detail"] == "Incorrect credentials."
        assert "학생통합" in profile_message
        assert "무시되어야할이름" not in profile_message
        assert "0/30" in profile_message
        assert len(profile_table["data"]) == 30
        assert "30/30" in save_message
        assert "학생통합 학생 예상" in prediction_message
        assert "다른학생" in other_message
        assert "0/30" in other_message
    finally:
        app.close()
