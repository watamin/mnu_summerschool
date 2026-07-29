from __future__ import annotations

from pathlib import Path

import pytest

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

    def fake_create_app(
        _dataset, _validation, *, embedder, nim_client, profile_store
    ):
        captured["embedder"] = embedder
        captured["nim_client"] = nim_client
        captured["profile_store"] = profile_store
        return "app"

    monkeypatch.setattr(mokpo_service, "create_mokpo_app", fake_create_app)

    result = mokpo_service.create_service_app(profile_db_path=tmp_path / "class.sqlite3")

    assert result == "app"
    assert isinstance(captured["profile_store"], StudentProfileStore)
    assert len(captured["profile_store"].food_pool) == 45
    assert captured["profile_store"].db_path == tmp_path / "class.sqlite3"
