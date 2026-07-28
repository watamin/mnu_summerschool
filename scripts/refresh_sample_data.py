"""남악고 NEIS 공개 급식 데이터를 수업용 예비 JSON으로 저장한다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neis_meal_ai.neis import fetch_meals, search_school  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", default="남악고등학교")
    parser.add_argument("--start", default="20260101")
    parser.add_argument("--end", default="20261231")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "namak_meals_sample.json")
    args = parser.parse_args()

    school = search_school(args.school)
    rows = fetch_meals(school, args.start, args.end)
    lunch_rows = [row for row in rows if row.get("MMEAL_SC_NM") == "중식"]
    if len(lunch_rows) < 5:
        raise RuntimeError(f"예비 데이터가 너무 적습니다: {len(lunch_rows)}행")

    payload = {
        "metadata": {
            "school_name": school.name,
            "office_code": school.office_code,
            "school_code": school.school_code,
            "source": "NEIS 교육정보 개방 포털",
            "query_start": args.start,
            "query_end": args.end,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_count": len(lunch_rows),
        },
        "rows": lunch_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output} school={school.name} code={school.office_code}/{school.school_code} rows={len(lunch_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
