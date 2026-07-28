from __future__ import annotations

import json
from pathlib import Path

from neis_meal_ai.cleaning import meals_to_frame


SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "namak_meals_sample.json"


def test_namak_sample_is_realistic_public_meal_data() -> None:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    assert payload["metadata"]["school_name"] == "남악고등학교"
    assert payload["metadata"]["office_code"] == "Q10"
    assert payload["metadata"]["school_code"] == "7140272"
    assert payload["metadata"]["source"] == "NEIS 교육정보 개방 포털"
    assert len(payload["rows"]) >= 5

    frame = meals_to_frame(payload["rows"])
    assert len(frame) >= 5
    assert set(frame["school_name"]) == {"남악고등학교"}
    assert frame["menu_text"].str.len().gt(0).all()
