"""목포시 학교 급식 AI 서비스를 로컬 또는 교실 LAN으로 실행한다."""

from __future__ import annotations

import argparse
import hmac
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from neis_meal_ai.mokpo_data import load_mokpo_dataset, load_validation_menus  # noqa: E402
from neis_meal_ai.mokpo_analytics import global_food_values  # noqa: E402
from neis_meal_ai.mokpo_ui import create_mokpo_app  # noqa: E402
from neis_meal_ai.nim_chat import NvidiaNimClient  # noqa: E402
from neis_meal_ai.student_profiles import (  # noqa: E402
    StudentProfileStore,
    validate_student_name,
)
from neis_meal_ai.text_vectors import SentenceTransformerEmbedder  # noqa: E402


SCHOOL_PATH = PROJECT_ROOT / "data" / "mokpo_schools.json"
MEAL_PATH = PROJECT_ROOT / "data" / "mokpo_meals_live.json"
MNU_PATH = PROJECT_ROOT / "data" / "mnu_cafeteria_2026_07_30_31.json"
NIM_KEY_PATH = PROJECT_ROOT.parent / "nvidia_nim.txt"
PASSWORD_PATH = PROJECT_ROOT.parent / "mokpo_password.txt"
PROFILE_DB_PATH = PROJECT_ROOT / "runtime_data" / "student_profiles.sqlite3"


def create_service_app(
    *,
    profile_db_path: str | Path = PROFILE_DB_PATH,
    classroom_mode: bool = False,
):
    dataset = load_mokpo_dataset(SCHOOL_PATH, MEAL_PATH)
    validation_menus, _ = load_validation_menus(MNU_PATH)
    food_pool = global_food_values(dataset.meals, top_n=45)["음식"].tolist()
    profile_store = StudentProfileStore(profile_db_path, food_pool)
    return create_mokpo_app(
        dataset,
        validation_menus,
        embedder=SentenceTransformerEmbedder(),
        nim_client=NvidiaNimClient(key_path=NIM_KEY_PATH),
        profile_store=profile_store,
        classroom_mode=classroom_mode,
    )


def load_shared_password(path: str | Path = PASSWORD_PATH) -> str:
    """저장소 밖의 한 줄짜리 교실 공통 비밀번호를 읽는다."""

    password_path = Path(path)
    try:
        text = password_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RuntimeError(
            f"교실 비밀번호 파일을 찾을 수 없습니다: {password_path}"
        ) from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("교실 비밀번호 파일은 비어 있지 않은 한 줄이어야 합니다.")
    value = lines[0]
    if "=" in value:
        key, candidate = value.split("=", 1)
        if key.strip().lower() not in {"password", "shared_password"}:
            raise RuntimeError("교실 비밀번호 파일의 이름=값 형식이 올바르지 않습니다.")
        value = candidate.strip()
    if not value:
        raise RuntimeError("교실 비밀번호 파일의 비밀번호가 비어 있습니다.")
    return value


def build_authenticator(password: str):
    """안전한 학생 이름과 정확한 공통 비밀번호만 허용한다."""

    if not isinstance(password, str) or not password:
        raise ValueError("교실 공통 비밀번호가 필요합니다.")
    expected = password.encode("utf-8")

    def authenticate(username: str, candidate: str) -> bool:
        try:
            validate_student_name(username)
        except ValueError:
            return False
        supplied = str(candidate or "").encode("utf-8")
        return hmac.compare_digest(supplied, expected)

    return authenticate


def launch_options(
    *, lan: bool, password: str | None = None, port: int | None = None
) -> dict[str, object]:
    """기본 로컬 모드와 명시적인 교실 LAN 모드의 실행 경계를 만든다."""

    options: dict[str, object] = {
        "server_name": "0.0.0.0" if lan else "127.0.0.1",
        "share": False,
        "inbrowser": True,
        "show_error": True,
    }
    if port is not None:
        options["server_port"] = int(port)
    if lan:
        if not password:
            raise ValueError("교실 LAN 실행에는 공통 비밀번호가 필요합니다.")
        options["auth"] = build_authenticator(password)
        options["auth_message"] = (
            "사용자 이름에는 자기 이름을, 비밀번호에는 교사가 안내한 공통 비밀번호를 입력하세요."
        )
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
        "--password-file",
        type=Path,
        default=PASSWORD_PATH,
        help="교실 공통 비밀번호가 든 저장소 밖 텍스트 파일",
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
    password = load_shared_password(args.password_file) if args.lan else None
    app = create_service_app(profile_db_path=args.db, classroom_mode=args.lan)
    app.launch(**launch_options(lan=args.lan, password=password, port=args.port))


if __name__ == "__main__":
    main()
