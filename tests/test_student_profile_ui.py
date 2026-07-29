from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from neis_meal_ai.student_profile_ui import (
    export_ratings_callback,
    load_profile_callback,
    matrix_dashboard_callback,
    save_profile_callback,
)
from neis_meal_ai.student_profiles import StudentProfileStore


def make_store(tmp_path: Path) -> StudentProfileStore:
    foods = [f"음식{i:02d}" for i in range(1, 46)]
    return StudentProfileStore(tmp_path / "profiles.sqlite3", foods)


def fill_six_profiles(store: StudentProfileStore) -> None:
    for student_index, name in enumerate(["학생A", "학생B", "학생C", "학생D", "학생E", "학생F"]):
        survey = store.load_survey(name)
        survey["평점"] = [
            ((food_index + student_index * 2) % 5) + 1
            for food_index in range(len(survey))
        ]
        store.save_ratings(name, survey)


def test_profile_callbacks_load_save_and_resume_authenticated_student(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)

    load_message, survey = load_profile_callback(store, "학생A")
    survey.loc[0, "평점"] = 5
    survey.loc[1, "평점"] = 2
    save_message, saved_survey = save_profile_callback(store, "학생A", survey)
    _, resumed = load_profile_callback(store, "학생A")

    assert "학생A" in load_message and "0/30" in load_message
    assert "2/30" in save_message and "중간 저장" in save_message
    assert saved_survey.loc[:1, "평점"].tolist() == [5, 2]
    assert resumed.loc[:1, "평점"].tolist() == [5, 2]


def test_matrix_dashboard_explains_when_class_data_is_not_ready(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    survey = store.load_survey("학생A")
    survey.loc[:4, "평점"] = [5, 4, 3, 2, 1]
    store.save_ratings("학생A", survey)

    message, status, observed, completed, metrics, recommendations, heatmap, coordinates, student_map = matrix_dashboard_callback(store)

    assert "2명" in message
    assert len(status) == 1
    assert observed.shape == (1, 45)
    assert completed.empty
    assert metrics.empty
    assert recommendations.empty
    assert heatmap is None and student_map is None
    assert coordinates.empty


def test_matrix_dashboard_marks_actual_and_predicted_cells_for_six_students(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    fill_six_profiles(store)

    message, status, observed, completed, metrics, recommendations, heatmap, coordinates, student_map = matrix_dashboard_callback(store)

    assert "180개 실제 평점" in message
    assert "90개 빈칸" in message
    assert status["완료"].tolist() == ["완료"] * 6
    assert observed.shape == (6, 45)
    assert completed.shape == (6, 45)
    flat_cells = completed.astype(str).to_numpy().ravel().tolist()
    assert sum("실제" in cell for cell in flat_cells) == 180
    assert sum("예측" in cell for cell in flat_cells) == 90
    assert metrics["지표"].tolist() == ["행렬분해 MAE", "행렬분해 RMSE", "전체평균 기준선 MAE"]
    assert len(recommendations) == 60
    assert isinstance(heatmap, Figure)
    assert isinstance(student_map, Figure)
    assert coordinates["학생"].tolist() == ["학생A", "학생B", "학생C", "학생D", "학생E", "학생F"]


def test_export_callback_creates_long_form_csv(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    survey = store.load_survey("학생A")
    survey.loc[0, "평점"] = 5
    store.save_ratings("학생A", survey)

    exported_path = Path(export_ratings_callback(store))
    exported = pd.read_csv(exported_path)

    assert exported_path.exists()
    assert len(exported) == 1
    assert exported.iloc[0]["이름"] == "학생A"
