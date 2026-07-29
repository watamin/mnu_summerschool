from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_jupyter_textbook import (
    _configure_notebook_event_loop,
    _normalize_curve_keys,
    _validate_chapter_result,
    verify_textbook,
)


def test_verifier_script_can_run_directly_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/verify_jupyter_textbook.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--chapters" in completed.stdout


def test_verify_textbook_rejects_missing_chapters(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="9개"):
        verify_textbook(tmp_path)


def test_verifier_uses_selector_event_loop_policy_on_windows() -> None:
    import asyncio
    import sys

    _configure_notebook_event_loop()

    if sys.platform == "win32":
        assert isinstance(
            asyncio.get_event_loop_policy(),
            asyncio.WindowsSelectorEventLoopPolicy,
        )


def test_verifier_normalizes_curve_keys_for_async_client() -> None:
    connection_info = {
        "curve_publickey": "public-key",
        "curve_secretkey": "secret-key",
        "ip": "127.0.0.1",
    }

    normalized = _normalize_curve_keys(connection_info)

    assert normalized["curve_publickey"] == b"public-key"
    assert normalized["curve_secretkey"] == b"secret-key"
    assert normalized["ip"] == "127.0.0.1"
    assert connection_info["curve_publickey"] == "public-key"


def test_verifier_rejects_incomplete_or_meaningless_chapter_contracts() -> None:
    with pytest.raises(RuntimeError, match="필수 결과"):
        _validate_chapter_result(
            "00",
            {
                "chapter": "00",
                "environment_ready": True,
                "sample_rows": 5,
            },
            Path("00.ipynb"),
        )

    with pytest.raises(RuntimeError, match="필수 결과"):
        _validate_chapter_result("06", {"chapter": "06"}, Path("06.ipynb"))

    with pytest.raises(RuntimeError, match="실제 콜백"):
        _validate_chapter_result(
            "06",
            {
                "chapter": "06",
                "widget_ready": True,
                "callback_rows": 3,
                "callback_status": "not_run",
                "callback_source": "남악고 NEIS 예비 데이터",
            },
            Path("06.ipynb"),
        )


def test_verify_textbook_executes_all_chapters_in_fresh_kernels(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    chapter_dir = Path(__file__).resolve().parents[1] / "jupyter_course" / "chapters"
    reports = verify_textbook(
        chapter_dir,
        timeout=120,
        executed_dir=tmp_path / "executed",
    )

    assert len(reports) == 10
    assert [report["chapter"] for report in reports] == [
        "00",
        "A",
        *[f"{number:02d}" for number in range(1, 9)],
    ]
    assert all(report["status"] == "PASS" for report in reports)
    assert all(report["code_cells"] >= 3 for report in reports)
    captured = capfd.readouterr()
    assert "without encryption" not in captured.err
