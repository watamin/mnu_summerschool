"""교사용 인증키로 NEIS 급식을 수집해 공개 가능한 JSON 스냅샷으로 저장한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neis_meal_ai.neis import SchoolInfo, fetch_meals, search_school  # noqa: E402


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


def read_api_key_file(path: str | Path) -> str:
    """한 줄짜리 키 파일을 읽되 값은 출력하거나 저장하지 않는다."""

    key_path = Path(path)
    try:
        raw_value = key_path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise RuntimeError("NEIS 인증키 파일을 읽을 수 없습니다.") from exc
    if raw_value.startswith("NEIS_API_KEY="):
        raw_value = raw_value.split("=", 1)[1].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", raw_value):
        raise RuntimeError("NEIS 인증키 파일은 인증키 한 줄만 포함해야 합니다.")
    return raw_value


@contextmanager
def neis_api_key_from_file(path: str | Path) -> Iterator[None]:
    """수집하는 동안만 키를 환경 변수에 넣고 이전 상태로 되돌린다."""

    previous = os.environ.get("NEIS_API_KEY")
    os.environ["NEIS_API_KEY"] = read_api_key_file(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NEIS_API_KEY", None)
        else:
            os.environ["NEIS_API_KEY"] = previous


def build_snapshot_payload(
    school: SchoolInfo,
    rows: list[dict],
    *,
    start: str,
    end: str,
    fetched_at: datetime,
) -> dict[str, object]:
    """인증정보를 제외한 공개 급식 행과 출처 메타데이터만 만든다."""

    public_rows = [
        {field: row[field] for field in PUBLIC_MEAL_FIELDS if field in row}
        for row in rows
    ]
    return {
        "metadata": {
            "snapshot_kind": "live",
            "school_name": school.name,
            "office_code": school.office_code,
            "school_code": school.school_code,
            "source": "NEIS 교육정보 개방 포털",
            "query_start": start,
            "query_end": end,
            "fetched_at_utc": fetched_at.astimezone(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "row_count": len(public_rows),
        },
        "rows": public_rows,
    }


def collect_live_meals(
    *,
    key_file: str | Path,
    school_name: str,
    start: str,
    end: str,
    output: str | Path,
) -> dict[str, object]:
    """NEIS에서 중식 행을 받아 JSON으로 저장하고 저장된 내용을 반환한다."""

    with neis_api_key_from_file(key_file):
        school = search_school(school_name)
        rows = fetch_meals(school, start, end)

    lunch_rows = [row for row in rows if row.get("MMEAL_SC_NM") == "중식"]
    if len(lunch_rows) < 5:
        raise RuntimeError(f"수집된 중식이 너무 적습니다: {len(lunch_rows)}일")

    payload = build_snapshot_payload(
        school,
        lunch_rows,
        start=start,
        end=end,
        fetched_at=datetime.now(timezone.utc),
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def run_collection_cli(
    args: argparse.Namespace,
    *,
    collector: Callable[..., dict[str, object]] = collect_live_meals,
) -> int:
    """명령줄 실패를 인증키가 없는 짧은 안내문으로 바꾼다."""

    try:
        payload = collector(
            key_file=args.key_file,
            school_name=args.school,
            start=args.start,
            end=args.end,
            output=args.output,
        )
    except Exception:
        print(
            "수집 실패: 인증키, 학교명, 날짜, 인터넷 연결을 확인해 주세요.",
            file=sys.stderr,
        )
        return 1

    metadata = payload["metadata"]
    print(
        f"수집 완료: {args.output} · {metadata['school_name']} · "
        f"{metadata['query_start']}~{metadata['query_end']} · {metadata['row_count']}일"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="교사용 NEIS 인증키로 급식 데이터를 안전하게 수집합니다."
    )
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--school", default="남악고등학교")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "namak_meals_live.json",
    )
    args = parser.parse_args(argv)
    return run_collection_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
