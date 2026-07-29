from __future__ import annotations

import json
import re
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neis_meal_ai.mokpo_data import load_mokpo_dataset, load_validation_menus
from scripts.collect_mokpo_meals import (
    build_catalog_payload,
    build_meal_payload,
    collect_mokpo_snapshot,
    run_collection_cli,
)
from neis_meal_ai.neis import NeisApiError


SCHOOLS = [
    {
        "school_name": "목포가람중학교",
        "school_kind": "중학교",
        "office_code": "Q10",
        "school_code": "8500001",
        "address": "전라남도 목포시 가람로 1",
    },
    {
        "school_name": "목포가람고등학교",
        "school_kind": "고등학교",
        "office_code": "Q10",
        "school_code": "7140001",
        "address": "전라남도 목포시 가람로 2",
    },
]


def _meal_row(school: dict, date: str, menu: str) -> dict:
    return {
        "ATPT_OFCDC_SC_CODE": "Q10",
        "ATPT_OFCDC_SC_NM": "전라남도교육청",
        "SD_SCHUL_CODE": school["school_code"],
        "SCHUL_NM": school["school_name"],
        "MMEAL_SC_CODE": "2",
        "MMEAL_SC_NM": "중식",
        "MLSV_YMD": date,
        "MLSV_FGR": "100",
        "DDISH_NM": f"{menu} (1.2.5.6)<br/>배추김치 (9)",
        "ORPLC_INFO": "",
        "CAL_INFO": "700.0 Kcal",
        "NTR_INFO": "탄수화물(g) : 100.0<br/>단백질(g) : 30.0<br/>지방(g) : 20.0",
        "MLSV_FROM_YMD": date,
        "MLSV_TO_YMD": date,
        "LOAD_DTM": "20260729000000",
    }


def _write_dataset(
    tmp_path: Path,
    *,
    school_change: dict | None = None,
    meal_change: dict | None = None,
    meal_metadata_change: dict | None = None,
) -> tuple[Path, Path]:
    schools = [dict(item) for item in SCHOOLS]
    schools[0].update(school_change or {})
    fetched_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    catalog = build_catalog_payload(schools, fetched_at=fetched_at)
    rows = [
        _meal_row(schools[0], "20260701", "치즈파스타"),
        _meal_row(schools[1], "20260702", "닭갈비덮밥"),
    ]
    rows[0].update(meal_change or {})
    meals = build_meal_payload(
        schools,
        rows,
        start="20260701",
        end="20260729",
        fetched_at=fetched_at,
        skipped_schools=[],
    )
    meals["metadata"].update(meal_metadata_change or {})
    school_path = tmp_path / "schools.json"
    meal_path = tmp_path / "meals.json"
    school_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    meal_path.write_text(json.dumps(meals, ensure_ascii=False), encoding="utf-8")
    return school_path, meal_path


def test_loader_returns_clean_middle_and_high_school_dataset(tmp_path: Path) -> None:
    school_path, meal_path = _write_dataset(tmp_path)

    dataset = load_mokpo_dataset(school_path, meal_path)

    assert set(dataset.schools["school_name"]) == {
        "목포가람중학교",
        "목포가람고등학교",
    }
    assert set(dataset.schools["school_kind"]) == {"중학교", "고등학교"}
    assert len(dataset.meals) == 2
    assert set(dataset.meals["school_kind"]) == {"중학교", "고등학교"}
    assert dataset.metadata["meal_row_count"] == 2


@pytest.mark.parametrize(
    ("school_change", "meal_change", "metadata_change"),
    [
        ({"office_code": "B10"}, {}, {}),
        ({"address": "전라남도 무안군 삼향읍"}, {}, {}),
        ({"school_kind": "초등학교"}, {}, {}),
        ({}, {"ATPT_OFCDC_SC_CODE": "B10"}, {}),
        ({}, {"SD_SCHUL_CODE": "9999999"}, {}),
        ({}, {"MMEAL_SC_NM": "석식"}, {}),
        ({}, {"MLSV_YMD": "2026071"}, {}),
        ({}, {}, {"fetched_at_utc": "not-a-date"}),
    ],
)
def test_loader_rejects_wrong_school_or_meal_identity(
    tmp_path: Path,
    school_change: dict,
    meal_change: dict,
    metadata_change: dict,
) -> None:
    school_path, meal_path = _write_dataset(
        tmp_path,
        school_change=school_change,
        meal_change=meal_change,
        meal_metadata_change=metadata_change,
    )

    with pytest.raises(RuntimeError):
        load_mokpo_dataset(school_path, meal_path)


def test_snapshot_builders_drop_unapproved_upstream_fields() -> None:
    fetched_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    raw_school = {
        "SCHUL_NM": "목포가람중학교",
        "SCHUL_KND_SC_NM": "중학교",
        "ATPT_OFCDC_SC_CODE": "Q10",
        "SD_SCHUL_CODE": "8500001",
        "ORG_RDNMA": "전라남도 목포시 가람로 1",
        "UNEXPECTED_FIELD": "저장 금지",
    }
    raw_meal = _meal_row(SCHOOLS[0], "20260701", "치즈파스타")
    raw_meal["UNEXPECTED_FIELD"] = "저장 금지"

    catalog = build_catalog_payload([raw_school], fetched_at=fetched_at)
    meals = build_meal_payload(
        catalog["schools"],
        [raw_meal],
        start="20260701",
        end="20260729",
        fetched_at=fetched_at,
        skipped_schools=[],
    )

    assert "UNEXPECTED_FIELD" not in catalog["schools"][0]
    assert "UNEXPECTED_FIELD" not in meals["rows"][0]
    assert meals["metadata"]["row_count"] == 1


def test_collection_cli_hides_a_key_embedded_in_an_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "fake-neis-secret-123456789012345"
    args = Namespace(
        key_file=tmp_path / "neis.txt",
        start="20260701",
        end="20260729",
        school_output=tmp_path / "schools.json",
        meal_output=tmp_path / "meals.json",
    )

    def failing_collector(**_kwargs):
        raise RuntimeError(f"https://open.neis.go.kr/?KEY={secret}")

    exit_code = run_collection_cli(args, collector=failing_collector)
    output = capsys.readouterr()

    assert exit_code == 1
    assert secret not in output.out
    assert secret not in output.err
    assert "수집 실패" in output.err


def test_mnu_validation_menus_preserve_user_provided_dates_and_text() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "mnu_cafeteria_2026_07_30_31.json"
    )

    frame, metadata = load_validation_menus(path)

    assert list(frame["meal_date"]) == ["2026-07-30", "2026-07-31"]
    assert "고구마대나물" in frame.iloc[0]["menu_text"]
    assert "양상추템더샐러드" in frame.iloc[1]["menu_text"]
    assert metadata["source"] == "수업 담당자 제공 목포대 학생식당 식단"
    assert metadata["is_neis_data"] is False


def test_collection_script_can_start_from_its_file_path() -> None:
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/collect_mokpo_meals.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--school-output" in completed.stdout


def test_committed_mokpo_snapshots_are_complete_and_secret_free() -> None:
    project_root = Path(__file__).resolve().parents[1]
    school_path = project_root / "data" / "mokpo_schools.json"
    meal_path = project_root / "data" / "mokpo_meals_live.json"

    dataset = load_mokpo_dataset(school_path, meal_path)
    public_text = school_path.read_text(encoding="utf-8") + meal_path.read_text(
        encoding="utf-8"
    )

    assert dataset.metadata["school_count"] == 31
    assert dataset.metadata["meal_school_count"] == 31
    assert dataset.metadata["meal_row_count"] == 674
    assert dataset.metadata["query_start"] == "20260601"
    assert dataset.metadata["actual_start"] == "20260624"
    assert dataset.metadata["actual_end"] == "20260729"
    assert set(dataset.schools["school_kind"]) == {"중학교", "고등학교"}
    assert not re.search(r'(?i)"(?:api_)?key"\s*:', public_text)
    assert not re.search(r"(?i)\b[0-9a-f]{32}\b", public_text)


def test_collection_aborts_on_api_error_instead_of_writing_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "neis.txt"
    key_file.write_text("x" * 32, encoding="utf-8")
    raw_schools = [
        {
            "SCHUL_NM": school["school_name"],
            "SCHUL_KND_SC_NM": school["school_kind"],
            "ATPT_OFCDC_SC_CODE": school["office_code"],
            "SD_SCHUL_CODE": school["school_code"],
            "ORG_RDNMA": school["address"],
        }
        for school in SCHOOLS
    ]
    calls = 0

    def partly_failing_fetch(school, _start, _end):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_meal_row(SCHOOLS[0], "20260701", "치즈파스타")]
        raise NeisApiError("통신 실패")

    monkeypatch.setattr(
        "scripts.collect_mokpo_meals._fetch_mokpo_school_rows", lambda: raw_schools
    )
    monkeypatch.setattr("scripts.collect_mokpo_meals.fetch_meals", partly_failing_fetch)
    school_output = tmp_path / "schools.json"
    meal_output = tmp_path / "meals.json"

    with pytest.raises(NeisApiError, match="통신 실패"):
        collect_mokpo_snapshot(
            key_file=key_file,
            start="20260701",
            end="20260729",
            school_output=school_output,
            meal_output=meal_output,
        )

    assert not school_output.exists()
    assert not meal_output.exists()
