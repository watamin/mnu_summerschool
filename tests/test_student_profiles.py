from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from neis_meal_ai.student_profiles import (
    StudentProfileStore,
    validate_student_name,
)


def food_pool() -> list[str]:
    return [f"음식{i:02d}" for i in range(1, 46)]


def test_student_name_normalizes_spaces_and_rejects_unsafe_characters() -> None:
    assert validate_student_name("  김 하늘  ") == "김 하늘"
    assert validate_student_name("학생_A-2") == "학생_A-2"

    for invalid in ("", "   ", "학생/1", "학생\n1", "가" * 21):
        with pytest.raises(ValueError, match="이름"):
            validate_student_name(invalid)


def test_new_profiles_receive_fifteen_common_and_fifteen_rotating_foods(
    tmp_path: Path,
) -> None:
    store = StudentProfileStore(tmp_path / "profiles.sqlite3", food_pool())

    first = store.load_survey("학생A")
    second = store.load_survey("학생B")

    assert list(first.columns) == ["순서", "음식", "구분", "평점"]
    assert len(first) == 30
    assert first["음식"].is_unique
    assert first["구분"].value_counts().to_dict() == {"공통": 15, "순환": 15}
    assert first.loc[first["구분"] == "공통", "음식"].tolist() == food_pool()[:15]
    assert first.loc[first["구분"] == "순환", "음식"].tolist() == food_pool()[15:30]
    assert second.loc[second["구분"] == "순환", "음식"].tolist() == food_pool()[20:35]


def test_survey_assignment_and_partial_ratings_survive_reopening_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profiles.sqlite3"
    original = StudentProfileStore(database, food_pool())
    survey = original.load_survey("학생A")
    result = original.save_ratings(
        "학생A",
        [
            {"음식": survey.iloc[0]["음식"], "평점": 5},
            {"음식": survey.iloc[1]["음식"], "평점": 2},
            {"음식": survey.iloc[2]["음식"], "평점": ""},
        ],
    )

    reopened = StudentProfileStore(database, food_pool())
    restored = reopened.load_survey("학생A")

    assert result.saved_count == 2
    assert result.total_questions == 30
    assert result.complete is False
    assert restored["음식"].tolist() == survey["음식"].tolist()
    assert restored["평점"].tolist()[:3] == [5, 2, 0]


def test_save_rejects_invalid_or_unassigned_ratings(tmp_path: Path) -> None:
    store = StudentProfileStore(tmp_path / "profiles.sqlite3", food_pool())
    store.load_survey("학생A")

    with pytest.raises(ValueError, match="1부터 5"):
        store.save_ratings("학생A", [{"음식": "음식01", "평점": 6}])
    with pytest.raises(ValueError, match="배정되지 않은"):
        store.save_ratings("학생A", [{"음식": "음식45", "평점": 4}])


def test_status_rating_matrix_and_csv_export_reflect_saved_observations(
    tmp_path: Path,
) -> None:
    store = StudentProfileStore(tmp_path / "profiles.sqlite3", food_pool())
    for name, rating in (("학생A", 5), ("학생B", 2)):
        survey = store.load_survey(name)
        store.save_ratings(
            name,
            [
                {"음식": row["음식"], "평점": rating}
                for row in survey.to_dict("records")
            ],
        )

    status = store.status()
    matrix = store.rating_matrix()
    export_path = store.export_ratings(tmp_path / "exports" / "ratings.csv")
    exported = pd.read_csv(export_path)

    assert status[["이름", "저장 문항", "전체 문항", "완료"]].to_dict("records") == [
        {"이름": "학생A", "저장 문항": 30, "전체 문항": 30, "완료": "완료"},
        {"이름": "학생B", "저장 문항": 30, "전체 문항": 30, "완료": "완료"},
    ]
    assert matrix.shape == (2, 45)
    assert int(matrix.notna().sum().sum()) == 60
    assert matrix.loc["학생A", "음식01"] == 5
    assert matrix.loc["학생B", "음식01"] == 2
    assert export_path.exists()
    assert len(exported) == 60
    assert set(exported.columns) == {"이름", "음식", "평점", "저장 시각"}


def test_concurrent_students_can_save_without_database_lock(tmp_path: Path) -> None:
    database = tmp_path / "profiles.sqlite3"
    StudentProfileStore(database, food_pool())

    def save_student(index: int) -> int:
        store = StudentProfileStore(database, food_pool())
        name = f"학생{index}"
        survey = store.load_survey(name)
        return store.save_ratings(
            name,
            [
                {"음식": row["음식"], "평점": index % 5 + 1}
                for row in survey.head(5).to_dict("records")
            ],
        ).saved_count

    with ThreadPoolExecutor(max_workers=6) as executor:
        counts = list(executor.map(save_student, range(6)))

    final_store = StudentProfileStore(database, food_pool())
    assert counts == [5, 5, 5, 5, 5, 5]
    assert len(final_store.status()) == 6
    assert int(final_store.rating_matrix().notna().sum().sum()) == 30


def test_food_pool_must_contain_exactly_forty_five_unique_names(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="45개"):
        StudentProfileStore(tmp_path / "short.sqlite3", food_pool()[:44])
    with pytest.raises(ValueError, match="중복"):
        StudentProfileStore(
            tmp_path / "duplicate.sqlite3", food_pool()[:44] + ["음식01"]
        )
