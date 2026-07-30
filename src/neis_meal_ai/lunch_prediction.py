from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .student_profiles import StudentProfileStore, validate_student_name
from .text_vectors import encode_texts


@dataclass(frozen=True, slots=True)
class DishPrediction:
    dish: str
    score: float
    evidence: str


def select_menu_row(validation_menus: pd.DataFrame, menu_id: object) -> pd.Series:
    required = {"menu_id", "meal_date", "dishes", "menu_text"}
    if not required.issubset(validation_menus.columns):
        raise ValueError("점심 메뉴 데이터의 열이 올바르지 않습니다.")
    selected = validation_menus.loc[
        validation_menus["menu_id"].astype(str) == str(menu_id or "")
    ]
    if len(selected) != 1:
        raise ValueError("예측할 점심 날짜를 선택해 주세요.")
    return selected.iloc[0]


def _complete_ratings(survey: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    if len(survey) != 30 or not {"음식", "평점"}.issubset(survey.columns):
        raise ValueError("예측하려면 음식 30개 평가를 모두 저장해 주세요.")
    ratings = pd.to_numeric(survey["평점"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    if not bool(valid.all()):
        raise ValueError("예측하려면 음식 30개 평가를 모두 저장해 주세요.")
    return survey["음식"].astype(str).tolist(), ratings


def _predict_dishes(
    foods: list[str], ratings: np.ndarray, dishes: list[str]
) -> tuple[DishPrediction, ...]:
    vectors = encode_texts([*foods, *dishes], method="tfidf").matrix
    food_vectors = vectors[: len(foods)]
    personal_mean = float(ratings.mean())
    results: list[DishPrediction] = []
    for offset, dish in enumerate(dishes):
        similarities = np.clip(vectors[len(foods) + offset] @ food_vectors.T, 0, 1)
        ranked = np.argsort(-similarities)
        matched = [index for index in ranked[:3] if similarities[index] > 1e-9]
        if matched:
            weights = similarities[matched]
            score = float(np.average(ratings[matched], weights=weights))
            evidence = ", ".join(
                f"{foods[index]}({ratings[index]:.0f}점, 유사도 {similarities[index]:.2f})"
                for index in matched
            )
        else:
            score = personal_mean
            evidence = f"비슷한 음식 이름이 없어 개인 평균 {personal_mean:.2f}점 사용"
        results.append(DishPrediction(str(dish), round(score, 2), evidence))
    return tuple(results)


def predict_profile_callback(
    store: StudentProfileStore,
    validation_menus: pd.DataFrame,
    student_name: object,
    menu_id: object,
) -> tuple[str, pd.DataFrame, dict[str, object]]:
    survey = store.load_survey(student_name)
    foods, ratings = _complete_ratings(survey)
    menu = select_menu_row(validation_menus, menu_id)
    dishes = [str(dish) for dish in menu["dishes"]]
    predictions = _predict_dishes(foods, ratings, dishes)
    predicted_score = round(float(np.mean([item.score for item in predictions])), 2)
    result = pd.DataFrame.from_records(
        [
            {
                "오늘 메뉴": item.dish,
                "예상 선호도(1~5)": item.score,
                "평가에서 찾은 근거": item.evidence,
            }
            for item in predictions
        ]
    )
    normalized_name = str(student_name).strip()
    state: dict[str, object] = {
        "student_name": normalized_name,
        "menu_id": str(menu["menu_id"]),
        "meal_date": str(menu["meal_date"]),
        "menu_text": str(menu["menu_text"]),
        "predicted_score": predicted_score,
    }
    message = (
        f"### {normalized_name} 학생 예상: {predicted_score:.2f}점\n"
        "저장된 30개 평가와 오늘 메뉴를 비교했습니다.\n\n"
        "계산 방법: **n‑gram TF‑IDF 유사도 가중평균**\n\n"
        "**식전에 만든 예측입니다.** 식후 평점과 관계없이 그대로 유지됩니다."
    )
    return message, result, state


def compare_actual_callback(
    prediction: object,
    actual_rating: object,
    student_name: object,
    menu_id: object,
) -> tuple[str, pd.DataFrame]:
    if not isinstance(prediction, Mapping) or "predicted_score" not in prediction:
        raise ValueError("먼저 2단계에서 오늘 점심을 예상해 주세요.")
    normalized_name = validate_student_name(student_name)
    if (
        prediction.get("student_name") != normalized_name
        or prediction.get("menu_id") != str(menu_id or "")
    ):
        raise ValueError(
            "이름이나 점심 날짜가 바뀌었습니다. 2단계에서 다시 예상해 주세요."
        )
    try:
        actual = float(actual_rating)
        predicted = float(prediction["predicted_score"])
    except (TypeError, ValueError) as exc:
        raise ValueError("실제 만족도는 1점부터 5점까지 입력해 주세요.") from exc
    if not 1 <= actual <= 5:
        raise ValueError("실제 만족도는 1점부터 5점까지 입력해 주세요.")
    difference = round(actual - predicted, 2)
    absolute_error = round(abs(difference), 2)
    if difference > 0:
        reading = f"실제가 예상보다 {difference:.2f}점 높았습니다."
    elif difference < 0:
        reading = f"실제가 예상보다 {abs(difference):.2f}점 낮았습니다."
    else:
        reading = "예상과 실제가 같았습니다."
    message = (
        f"### 비교 결과: 예상 {predicted:.2f}점 · 실제 {actual:.2f}점\n"
        f"{reading} 절대 오차는 **{absolute_error:.2f}점**입니다."
    )
    result = pd.DataFrame.from_records(
        [
            {
                "날짜": str(prediction.get("meal_date", "")),
                "예상 만족도": round(predicted, 2),
                "실제 만족도": round(actual, 2),
                "실제-예상": difference,
                "절대 오차": absolute_error,
            }
        ]
    )
    return message, result


def compare_actual_ui_callback(
    prediction: object,
    actual_rating: object,
    student_name: object,
    menu_id: object,
) -> tuple[str, pd.DataFrame]:
    try:
        return compare_actual_callback(
            prediction, actual_rating, student_name, menu_id
        )
    except ValueError as exc:
        columns = ["날짜", "예상 만족도", "실제 만족도", "실제-예상", "절대 오차"]
        return f"### 비교할 수 없습니다\n{exc}", pd.DataFrame(columns=columns)
