from __future__ import annotations

import json
import os
import re
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jupyter_course.notebook_support import load_web_frame
from neis_meal_ai.neis import SchoolInfo
from scripts.collect_live_meals import (
    build_snapshot_payload,
    neis_api_key_from_file,
    read_api_key_file,
    run_collection_cli,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "namak_meals_sample.json"
LIVE_PATH = PROJECT_ROOT / "data" / "namak_meals_live.json"


def _sample_rows() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["rows"]


def test_key_file_is_read_without_changing_its_value(tmp_path: Path) -> None:
    key_file = tmp_path / "neis.txt"
    key_file.write_text("a1b2c3d4e5f60718293a4b5c6d7e8f90\n", encoding="utf-8")

    assert read_api_key_file(key_file) == "a1b2c3d4e5f60718293a4b5c6d7e8f90"


def test_key_file_context_restores_the_previous_environment(tmp_path: Path) -> None:
    key_file = tmp_path / "neis.txt"
    key_file.write_text("a1b2c3d4e5f60718293a4b5c6d7e8f90", encoding="utf-8")
    previous = os.environ.get("NEIS_API_KEY")
    os.environ["NEIS_API_KEY"] = "previous-test-value"

    try:
        with neis_api_key_from_file(key_file):
            assert os.environ["NEIS_API_KEY"] == "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        assert os.environ["NEIS_API_KEY"] == "previous-test-value"
    finally:
        if previous is None:
            os.environ.pop("NEIS_API_KEY", None)
        else:
            os.environ["NEIS_API_KEY"] = previous


def test_key_file_context_restores_environment_when_collection_fails(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "neis.txt"
    key_file.write_text("temporary-test-key-1234567890", encoding="utf-8")
    previous = os.environ.get("NEIS_API_KEY")
    os.environ["NEIS_API_KEY"] = "previous-test-value"

    try:
        with pytest.raises(RuntimeError, match="의도한 실패"):
            with neis_api_key_from_file(key_file):
                raise RuntimeError("의도한 실패")
        assert os.environ["NEIS_API_KEY"] == "previous-test-value"
    finally:
        if previous is None:
            os.environ.pop("NEIS_API_KEY", None)
        else:
            os.environ["NEIS_API_KEY"] = previous


def test_cli_failure_message_never_exposes_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "fake-secret-key-1234567890123456"
    args = Namespace(
        key_file=tmp_path / "neis.txt",
        school="남악고등학교",
        start="20260624",
        end="20260630",
        output=tmp_path / "live.json",
    )

    def failing_collector(**_kwargs):
        raise RuntimeError(f"https://open.neis.go.kr/?KEY={secret}")

    exit_code = run_collection_cli(args, collector=failing_collector)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert secret not in captured.out
    assert secret not in captured.err
    assert "수집 실패" in captured.err


def test_snapshot_payload_contains_public_meals_but_never_the_api_key() -> None:
    secret = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    os.environ["NEIS_API_KEY"] = secret
    try:
        payload = build_snapshot_payload(
            SchoolInfo("남악고등학교", "Q10", "7140272"),
            _sample_rows(),
            start="20260624",
            end="20260630",
            fetched_at=datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
        )
    finally:
        os.environ.pop("NEIS_API_KEY", None)

    serialized = json.dumps(payload, ensure_ascii=False)
    assert secret not in serialized
    assert payload["metadata"]["snapshot_kind"] == "live"
    assert payload["metadata"]["row_count"] == 5
    assert payload["metadata"]["school_code"] == "7140272"


def test_snapshot_payload_keeps_only_approved_public_meal_fields() -> None:
    rows = _sample_rows()
    rows[0]["UNEXPECTED_UPSTREAM_FIELD"] = "저장하면 안 되는 값"

    payload = build_snapshot_payload(
        SchoolInfo("남악고등학교", "Q10", "7140272"),
        rows,
        start="20260624",
        end="20260630",
        fetched_at=datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )

    assert "UNEXPECTED_UPSTREAM_FIELD" not in payload["rows"][0]
    assert payload["rows"][0]["DDISH_NM"]


def test_web_loader_prefers_a_valid_live_snapshot(tmp_path: Path) -> None:
    live_path = tmp_path / "live.json"
    payload = build_snapshot_payload(
        SchoolInfo("남악고등학교", "Q10", "7140272"),
        _sample_rows(),
        start="20260624",
        end="20260630",
        fetched_at=datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )
    live_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    frame, source = load_web_frame(PROJECT_ROOT, live_path=live_path)

    assert len(frame) == 5
    assert "NEIS 실제 수집 데이터 5일" in source
    assert "2026-06-24~2026-06-30" in source
    assert "2026-07-29 수집" in source


def test_web_loader_uses_sample_when_the_live_snapshot_is_broken(
    tmp_path: Path,
) -> None:
    broken_path = tmp_path / "live.json"
    broken_path.write_text('{"metadata": {}, "rows": []}', encoding="utf-8")

    frame, source = load_web_frame(PROJECT_ROOT, live_path=broken_path)

    assert len(frame) == 5
    assert "예비 데이터 5일" in source
    assert "실제 수집본 오류" in source


@pytest.mark.parametrize(
    ("metadata_change", "row_change"),
    [
        ({"office_code": "B10"}, {}),
        ({"school_code": "9999999"}, {}),
        ({"fetched_at_utc": "not-a-date"}, {}),
        ({}, {"MMEAL_SC_NM": "석식"}),
        ({}, {"MLSV_YMD": "2026061"}),
        ({}, {"MLSV_YMD": None}),
        ({}, {"DDISH_NM": None}),
    ],
)
def test_web_loader_falls_back_for_wrong_or_malformed_live_rows(
    tmp_path: Path,
    metadata_change: dict,
    row_change: dict,
) -> None:
    payload = build_snapshot_payload(
        SchoolInfo("남악고등학교", "Q10", "7140272"),
        _sample_rows(),
        start="20260624",
        end="20260630",
        fetched_at=datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )
    payload["metadata"].update(metadata_change)
    payload["rows"][0].update(row_change)
    live_path = tmp_path / "invalid-live.json"
    live_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    frame, source = load_web_frame(PROJECT_ROOT, live_path=live_path)

    assert len(frame) == 5
    assert "실제 수집본 오류" in source


def test_repository_live_snapshot_is_public_and_contains_more_than_the_demo() -> None:
    payload = json.loads(LIVE_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["metadata"]["snapshot_kind"] == "live"
    assert payload["metadata"]["school_name"] == "남악고등학교"
    assert payload["metadata"]["school_code"] == "7140272"
    assert payload["metadata"]["row_count"] == 24
    assert "api_key" not in serialized.casefold()
    assert '"KEY"' not in serialized
    assert re.search(r"\b[0-9a-fA-F]{32}\b", serialized) is None
