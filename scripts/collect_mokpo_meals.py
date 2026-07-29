"""교사용 인증키로 목포시 중·고교 목록과 중식 공개 스냅샷을 만든다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from neis_meal_ai.mokpo_data import (  # noqa: E402
    ALLOWED_SCHOOL_KINDS,
    EXPECTED_LOCATION,
    EXPECTED_OFFICE_CODE,
    PUBLIC_MEAL_FIELDS,
    validate_mokpo_payloads,
)
from neis_meal_ai.neis import (  # noqa: E402
    SchoolInfo,
    _extract_rows,
    _request_json,
    fetch_meals,
    validate_date_range,
)
from scripts.collect_live_meals import neis_api_key_from_file  # noqa: E402


Collector = Callable[..., tuple[dict, dict]]


def _school_value(row: dict, public_name: str, neis_name: str) -> str:
    return str(row.get(public_name, row.get(neis_name, ""))).strip()


def _public_school(row: dict) -> dict[str, str]:
    return {
        "school_name": _school_value(row, "school_name", "SCHUL_NM"),
        "school_kind": _school_value(row, "school_kind", "SCHUL_KND_SC_NM"),
        "office_code": _school_value(row, "office_code", "ATPT_OFCDC_SC_CODE"),
        "school_code": _school_value(row, "school_code", "SD_SCHUL_CODE"),
        "address": _school_value(row, "address", "ORG_RDNMA"),
    }


def build_catalog_payload(rows: list[dict], *, fetched_at: datetime) -> dict:
    """NEIS 학교 행에서 공개 가능한 다섯 필드만 남긴다."""

    schools = [_public_school(row) for row in rows]
    schools.sort(key=lambda item: (item["school_kind"], item["school_name"]))
    return {
        "metadata": {
            "snapshot_kind": "mokpo_school_catalog",
            "office_code": EXPECTED_OFFICE_CODE,
            "location": EXPECTED_LOCATION,
            "source": "NEIS 교육정보 개방 포털",
            "fetched_at_utc": fetched_at.astimezone(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "school_count": len(schools),
        },
        "schools": schools,
    }


def build_meal_payload(
    schools: list[dict],
    rows: list[dict],
    *,
    start: str,
    end: str,
    fetched_at: datetime,
    skipped_schools: list[str],
) -> dict:
    """인증정보와 예상 밖 필드를 제외한 공개 급식 스냅샷을 만든다."""

    public_rows = [
        {field: row[field] for field in PUBLIC_MEAL_FIELDS if field in row}
        for row in rows
    ]
    used_codes = {str(row.get("SD_SCHUL_CODE", "")) for row in public_rows}
    return {
        "metadata": {
            "snapshot_kind": "mokpo_live_meals",
            "office_code": EXPECTED_OFFICE_CODE,
            "location": EXPECTED_LOCATION,
            "source": "NEIS 교육정보 개방 포털",
            "query_start": start,
            "query_end": end,
            "fetched_at_utc": fetched_at.astimezone(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "catalog_school_count": len(schools),
            "school_count": len(used_codes),
            "row_count": len(public_rows),
        },
        "skipped_schools": sorted(set(skipped_schools)),
        "rows": public_rows,
    }


def _fetch_mokpo_school_rows() -> list[dict]:
    payload = _request_json(
        "schoolInfo",
        {
            "Type": "json",
            "pIndex": 1,
            "pSize": 1000,
            "ATPT_OFCDC_SC_CODE": EXPECTED_OFFICE_CODE,
        },
        http_get=requests.get,
    )
    rows = _extract_rows(payload, "schoolInfo")
    return [
        row
        for row in rows
        if EXPECTED_LOCATION in str(row.get("ORG_RDNMA", ""))
        and row.get("SCHUL_KND_SC_NM") in ALLOWED_SCHOOL_KINDS
    ]


def collect_mokpo_snapshot(
    *,
    key_file: str | Path,
    start: str,
    end: str,
    school_output: str | Path,
    meal_output: str | Path,
) -> tuple[dict, dict]:
    """목포 중·고교와 중식을 모아 검증 후 두 JSON 파일로 저장한다."""

    validate_date_range(start, end)
    fetched_at = datetime.now(timezone.utc)
    meal_rows: list[dict] = []
    skipped_schools: list[str] = []
    with neis_api_key_from_file(key_file):
        raw_schools = _fetch_mokpo_school_rows()
        if not raw_schools:
            raise RuntimeError("목포시 중·고교를 찾지 못했습니다.")
        catalog = build_catalog_payload(raw_schools, fetched_at=fetched_at)
        for school_row in catalog["schools"]:
            school = SchoolInfo(
                name=school_row["school_name"],
                office_code=school_row["office_code"],
                school_code=school_row["school_code"],
                school_kind=school_row["school_kind"],
                address=school_row["address"],
            )
            rows = fetch_meals(school, start, end)
            lunches = [row for row in rows if row.get("MMEAL_SC_NM") == "중식"]
            if not lunches:
                skipped_schools.append(school.name)
                continue
            meal_rows.extend(lunches)

    if not meal_rows:
        raise RuntimeError("수집된 목포시 중식 데이터가 없습니다.")
    meals = build_meal_payload(
        catalog["schools"],
        meal_rows,
        start=start,
        end=end,
        fetched_at=fetched_at,
        skipped_schools=skipped_schools,
    )
    validate_mokpo_payloads(catalog, meals)
    school_path = Path(school_output)
    meal_path = Path(meal_output)
    school_path.parent.mkdir(parents=True, exist_ok=True)
    meal_path.parent.mkdir(parents=True, exist_ok=True)
    school_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    meal_path.write_text(
        json.dumps(meals, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return catalog, meals


def run_collection_cli(
    args: argparse.Namespace,
    *,
    collector: Collector = collect_mokpo_snapshot,
) -> int:
    try:
        catalog, meals = collector(
            key_file=args.key_file,
            start=args.start,
            end=args.end,
            school_output=args.school_output,
            meal_output=args.meal_output,
        )
    except Exception:
        print(
            "수집 실패: 인증키, 날짜, 인터넷 연결을 확인해 주세요.",
            file=sys.stderr,
        )
        return 1
    print(
        f"수집 완료: 목포시 중·고교 {catalog['metadata']['school_count']}개교 · "
        f"중식 {meals['metadata']['row_count']}일"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="목포시 중·고교 NEIS 중식을 수집합니다.")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--school-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "mokpo_schools.json",
    )
    parser.add_argument(
        "--meal-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "mokpo_meals_live.json",
    )
    return run_collection_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
