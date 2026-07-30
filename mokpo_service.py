"""목포시 학교 급식 AI 서비스를 로컬 또는 교실 LAN으로 실행한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from neis_meal_ai.mokpo_data import load_mokpo_dataset, load_validation_menus  # noqa: E402
from neis_meal_ai.core_meal_ui import create_mokpo_app  # noqa: E402
from neis_meal_ai.mokpo_analytics import global_food_values  # noqa: E402
from neis_meal_ai.student_profiles import StudentProfileStore  # noqa: E402


SCHOOL_PATH = PROJECT_ROOT / "data" / "mokpo_schools.json"
MEAL_PATH = PROJECT_ROOT / "data" / "mokpo_meals_live.json"
MNU_PATH = PROJECT_ROOT / "data" / "mnu_cafeteria_2026_07_30_31.json"
PROFILE_DB_PATH = PROJECT_ROOT / "runtime_data" / "student_profiles.sqlite3"


def create_service_app(
    *,
    profile_db_path: str | Path = PROFILE_DB_PATH,
):
    dataset = load_mokpo_dataset(SCHOOL_PATH, MEAL_PATH)
    validation_menus, _ = load_validation_menus(MNU_PATH)
    food_pool = global_food_values(dataset.meals, top_n=45)["음식"].tolist()
    profile_store = StudentProfileStore(profile_db_path, food_pool)
    return create_mokpo_app(
        dataset,
        validation_menus,
        profile_store=profile_store,
    )


def launch_options(
    *,
    lan: bool,
    port: int | None = None,
) -> dict[str, object]:
    """로그인 없이 로컬 또는 교실 LAN 실행 경계를 만든다."""

    options: dict[str, object] = {
        "server_name": "0.0.0.0" if lan else "127.0.0.1",
        "share": False,
        "inbrowser": True,
        "show_error": True,
    }
    if port is not None:
        options["server_port"] = int(port)
    return options


def local_launch_options() -> dict[str, object]:
    """기존 로컬 실행 API를 유지한다."""

    return launch_options(lan=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="목포 급식 AI 교실 웹 서비스")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="같은 네트워크의 학생 PC에서 접속할 수 있도록 연다.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROFILE_DB_PATH,
        help="학생 프로필과 평점을 저장할 SQLite 파일",
    )
    parser.add_argument("--port", type=int, default=None, help="웹 서비스 포트")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    app = create_service_app(profile_db_path=args.db)
    app.launch(**launch_options(lan=args.lan, port=args.port))


if __name__ == "__main__":
    main()
