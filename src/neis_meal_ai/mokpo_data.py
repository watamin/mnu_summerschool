"""목포시 중·고교 공개 급식 스냅샷의 검증과 로딩."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .cleaning import meals_to_frame


EXPECTED_OFFICE_CODE = "Q10"
EXPECTED_LOCATION = "목포시"
ALLOWED_SCHOOL_KINDS = {"중학교", "고등학교"}
PUBLIC_SCHOOL_FIELDS = (
    "school_name",
    "school_kind",
    "office_code",
    "school_code",
    "address",
)
PUBLIC_MEAL_FIELDS = (
    "ATPT_OFCDC_SC_CODE",
    "ATPT_OFCDC_SC_NM",
    "SD_SCHUL_CODE",
    "SCHUL_NM",
    "MMEAL_SC_CODE",
    "MMEAL_SC_NM",
    "MLSV_YMD",
    "MLSV_FGR",
    "DDISH_NM",
    "ORPLC_INFO",
    "CAL_INFO",
    "NTR_INFO",
    "MLSV_FROM_YMD",
    "MLSV_TO_YMD",
    "LOAD_DTM",
)
REQUIRED_MEAL_FIELDS = {
    "ATPT_OFCDC_SC_CODE",
    "SD_SCHUL_CODE",
    "SCHUL_NM",
    "MMEAL_SC_NM",
    "MLSV_YMD",
    "DDISH_NM",
    "CAL_INFO",
    "NTR_INFO",
}


@dataclass(frozen=True)
class MokpoDataset:
    """웹 서비스가 사용하는 검증 완료 학교·급식 표."""

    schools: pd.DataFrame
    meals: pd.DataFrame
    metadata: dict[str, Any]


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} 파일을 읽을 수 없습니다.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} 파일의 최상위 구조가 올바르지 않습니다.")
    return payload


def _validate_utc(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} 수집 시각이 올바르지 않습니다.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} 수집 시각이 올바르지 않습니다.") from exc
    if parsed.utcoffset() != timedelta(0):
        raise RuntimeError(f"{label} 수집 시각은 UTC여야 합니다.")


def _validate_date(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{8}", value) is None:
        raise RuntimeError("목포 급식 데이터의 날짜 형식이 올바르지 않습니다.")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise RuntimeError("목포 급식 데이터의 날짜가 올바르지 않습니다.") from exc
    return value


def validate_mokpo_payloads(
    catalog: dict[str, Any], meals: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """두 공개 스냅샷의 출처·학교·행 구조를 함께 검증한다."""

    catalog_metadata = catalog.get("metadata")
    school_rows = catalog.get("schools")
    if not isinstance(catalog_metadata, dict) or catalog_metadata.get(
        "snapshot_kind"
    ) != "mokpo_school_catalog":
        raise RuntimeError("목포 학교 목록의 메타데이터가 올바르지 않습니다.")
    if catalog_metadata.get("office_code") != EXPECTED_OFFICE_CODE:
        raise RuntimeError("목포 학교 목록의 교육청 코드가 올바르지 않습니다.")
    if catalog_metadata.get("location") != EXPECTED_LOCATION:
        raise RuntimeError("목포 학교 목록의 지역 정보가 올바르지 않습니다.")
    _validate_utc(catalog_metadata.get("fetched_at_utc"), "목포 학교 목록")
    if not isinstance(school_rows, list) or not school_rows:
        raise RuntimeError("목포 학교 목록이 비어 있습니다.")
    if catalog_metadata.get("school_count") != len(school_rows):
        raise RuntimeError("목포 학교 목록의 학교 수 기록이 맞지 않습니다.")

    schools_by_code: dict[str, dict[str, Any]] = {}
    school_names: set[str] = set()
    for school in school_rows:
        if not isinstance(school, dict) or set(school) != set(PUBLIC_SCHOOL_FIELDS):
            raise RuntimeError("목포 학교 목록의 공개 필드가 올바르지 않습니다.")
        if school["office_code"] != EXPECTED_OFFICE_CODE:
            raise RuntimeError("목포 학교 목록에 다른 교육청 학교가 있습니다.")
        if school["school_kind"] not in ALLOWED_SCHOOL_KINDS:
            raise RuntimeError("목포 학교 목록에 중·고교가 아닌 학교가 있습니다.")
        if EXPECTED_LOCATION not in str(school["address"]):
            raise RuntimeError("목포 학교 목록에 목포시 밖의 학교가 있습니다.")
        school_code = str(school["school_code"]).strip()
        school_name = str(school["school_name"]).strip()
        if not school_code or not school_name:
            raise RuntimeError("목포 학교 목록의 학교 식별 정보가 비어 있습니다.")
        if school_code in schools_by_code or school_name in school_names:
            raise RuntimeError("목포 학교 목록에 중복 학교가 있습니다.")
        schools_by_code[school_code] = school
        school_names.add(school_name)

    meal_metadata = meals.get("metadata")
    meal_rows = meals.get("rows")
    if not isinstance(meal_metadata, dict) or meal_metadata.get(
        "snapshot_kind"
    ) != "mokpo_live_meals":
        raise RuntimeError("목포 급식 데이터의 메타데이터가 올바르지 않습니다.")
    if meal_metadata.get("office_code") != EXPECTED_OFFICE_CODE:
        raise RuntimeError("목포 급식 데이터의 교육청 코드가 올바르지 않습니다.")
    _validate_utc(meal_metadata.get("fetched_at_utc"), "목포 급식 데이터")
    start = _validate_date(meal_metadata.get("query_start"))
    end = _validate_date(meal_metadata.get("query_end"))
    if start > end:
        raise RuntimeError("목포 급식 데이터의 조회 기간이 올바르지 않습니다.")
    if not isinstance(meal_rows, list) or not meal_rows:
        raise RuntimeError("목포 급식 데이터가 비어 있습니다.")
    if meal_metadata.get("row_count") != len(meal_rows):
        raise RuntimeError("목포 급식 데이터의 행 수 기록이 맞지 않습니다.")

    used_school_codes: set[str] = set()
    for row in meal_rows:
        if not isinstance(row, dict):
            raise RuntimeError("목포 급식 행의 구조가 올바르지 않습니다.")
        if not REQUIRED_MEAL_FIELDS.issubset(row) or not set(row).issubset(
            PUBLIC_MEAL_FIELDS
        ):
            raise RuntimeError("목포 급식 행의 공개 필드가 올바르지 않습니다.")
        school_code = str(row["SD_SCHUL_CODE"]).strip()
        school = schools_by_code.get(school_code)
        if school is None or row["SCHUL_NM"] != school["school_name"]:
            raise RuntimeError("목포 급식 행의 학교가 학교 목록과 맞지 않습니다.")
        if row["ATPT_OFCDC_SC_CODE"] != EXPECTED_OFFICE_CODE:
            raise RuntimeError("목포 급식 행에 다른 교육청 데이터가 있습니다.")
        if row["MMEAL_SC_NM"] != "중식":
            raise RuntimeError("목포 급식 행에 중식이 아닌 데이터가 있습니다.")
        meal_date = _validate_date(row["MLSV_YMD"])
        if not start <= meal_date <= end:
            raise RuntimeError("목포 급식 행의 날짜가 조회 기간 밖입니다.")
        if not isinstance(row["DDISH_NM"], str) or not row["DDISH_NM"].strip():
            raise RuntimeError("목포 급식 행의 메뉴가 비어 있습니다.")
        used_school_codes.add(school_code)

    if meal_metadata.get("school_count") != len(used_school_codes):
        raise RuntimeError("목포 급식 데이터의 학교 수 기록이 맞지 않습니다.")
    if meal_metadata.get("catalog_school_count") != len(school_rows):
        raise RuntimeError("목포 급식 데이터의 전체 학교 수 기록이 맞지 않습니다.")
    return school_rows, meal_rows


def load_mokpo_dataset(
    school_path: str | Path, meal_path: str | Path
) -> MokpoDataset:
    """검증한 JSON을 학교·급식 분석용 DataFrame으로 바꾼다."""

    catalog = _read_json(school_path, "목포 학교 목록")
    meal_payload = _read_json(meal_path, "목포 급식 데이터")
    school_rows, meal_rows = validate_mokpo_payloads(catalog, meal_payload)
    school_frame = pd.DataFrame.from_records(school_rows, columns=PUBLIC_SCHOOL_FIELDS)
    meal_frame = meals_to_frame(meal_rows)
    if meal_frame.empty:
        raise RuntimeError("목포 급식 데이터를 분석 표로 바꿀 수 없습니다.")
    meal_frame = meal_frame.merge(
        school_frame[["school_name", "school_kind", "school_code", "address"]],
        on="school_name",
        how="left",
        validate="many_to_one",
    )
    metadata = {
        "school_count": len(school_frame),
        "meal_school_count": int(meal_frame["school_name"].nunique()),
        "meal_row_count": len(meal_frame),
        "query_start": meal_payload["metadata"]["query_start"],
        "query_end": meal_payload["metadata"]["query_end"],
        "actual_start": str(meal_frame["date"].min()).replace("-", ""),
        "actual_end": str(meal_frame["date"].max()).replace("-", ""),
        "fetched_at_utc": meal_payload["metadata"]["fetched_at_utc"],
        "skipped_schools": meal_payload.get("skipped_schools", []),
    }
    return MokpoDataset(school_frame, meal_frame, metadata)


def load_validation_menus(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """NEIS와 분리된 목포대 학생식당 검증 식단을 읽는다."""

    payload = _read_json(path, "목포대 검증 식단")
    metadata = payload.get("metadata")
    rows = payload.get("meals")
    if not isinstance(metadata, dict) or metadata.get(
        "snapshot_kind"
    ) != "mnu_cafeteria_validation":
        raise RuntimeError("목포대 검증 식단의 메타데이터가 올바르지 않습니다.")
    if metadata.get("is_neis_data") is not False:
        raise RuntimeError("목포대 검증 식단을 NEIS 데이터로 표시할 수 없습니다.")
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError("목포대 검증 식단은 이틀치여야 합니다.")
    expected_dates = ["2026-07-30", "2026-07-31"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("meal_date") != expected_dates[index]:
            raise RuntimeError("목포대 검증 식단의 날짜가 올바르지 않습니다.")
        dishes = row.get("dishes")
        if not isinstance(dishes, list) or not dishes or not all(
            isinstance(dish, str) and dish.strip() for dish in dishes
        ):
            raise RuntimeError("목포대 검증 식단의 메뉴가 올바르지 않습니다.")
        if row.get("menu_text") != " ".join(dishes):
            raise RuntimeError("목포대 검증 식단의 메뉴 문장이 원문과 맞지 않습니다.")
    return pd.DataFrame.from_records(rows), metadata
