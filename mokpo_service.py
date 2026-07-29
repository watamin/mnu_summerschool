"""목포시 학교 급식과 목포대 리뷰 검증을 묶은 로컬 웹 서비스."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from neis_meal_ai.mokpo_data import load_mokpo_dataset, load_validation_menus  # noqa: E402
from neis_meal_ai.mokpo_ui import create_mokpo_app  # noqa: E402
from neis_meal_ai.nim_chat import NvidiaNimClient  # noqa: E402
from neis_meal_ai.text_vectors import SentenceTransformerEmbedder  # noqa: E402


SCHOOL_PATH = PROJECT_ROOT / "data" / "mokpo_schools.json"
MEAL_PATH = PROJECT_ROOT / "data" / "mokpo_meals_live.json"
MNU_PATH = PROJECT_ROOT / "data" / "mnu_cafeteria_2026_07_30_31.json"
NIM_KEY_PATH = PROJECT_ROOT.parent / "nvidia_nim.txt"


def create_service_app():
    dataset = load_mokpo_dataset(SCHOOL_PATH, MEAL_PATH)
    validation_menus, _ = load_validation_menus(MNU_PATH)
    return create_mokpo_app(
        dataset,
        validation_menus,
        embedder=SentenceTransformerEmbedder(),
        nim_client=NvidiaNimClient(key_path=NIM_KEY_PATH),
    )


def local_launch_options() -> dict[str, object]:
    return {
        "server_name": "127.0.0.1",
        "share": False,
        "inbrowser": True,
        "show_error": True,
    }


if __name__ == "__main__":
    create_service_app().launch(**local_launch_options())
