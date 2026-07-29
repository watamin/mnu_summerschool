"""목포 급식 서비스의 콘텐츠·유저 기반 추천과 모둠 분석."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .recommender import MENU_TYPE_KEYWORDS, _menu_types, _spice_level
from .text_vectors import TextEmbedder, cosine_scores, encode_texts


GRADE_BANDS = {"중1", "중2", "중3", "고1", "고2", "고3", "대학생"}
FEEDBACK_TAGS = {"맛", "양", "매운맛", "메뉴 조합", "기타"}
FEEDBACK_COLUMNS = [
    "participant_code",
    "grade_band",
    "menu_id",
    "meal_date",
    "source",
    "menu_text",
    "likes_text",
    "avoids_text",
    "preferred_types",
    "spice_level",
    "content_method",
    "content_prediction",
    "actual_rating",
    "feedback_tag",
]
CSV_TEXT_COLUMNS = {
    "participant_code",
    "grade_band",
    "menu_id",
    "meal_date",
    "source",
    "menu_text",
    "likes_text",
    "avoids_text",
    "preferred_types",
    "content_method",
    "feedback_tag",
}


def _csv_terms(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text or "").split(",") if part.strip())


def predict_satisfaction(
    frame: pd.DataFrame,
    *,
    likes_text: str,
    avoids_text: str,
    preferred_types: Sequence[str],
    spice_level: int,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> tuple[pd.DataFrame, str]:
    """익명 취향과 메뉴의 콘텐츠 유사도로 0~100 예측 점수를 만든다."""

    if "menu_text" not in frame or frame.empty:
        raise ValueError("분석할 메뉴 데이터가 없습니다.")
    likes = _csv_terms(likes_text)
    avoids = _csv_terms(avoids_text)
    types = tuple(str(value).strip() for value in preferred_types if str(value).strip())
    if not likes and not types:
        raise ValueError("좋아하는 메뉴나 선호 유형을 하나 이상 입력하세요.")
    if set(types) - set(MENU_TYPE_KEYWORDS):
        raise ValueError("지원하지 않는 선호 유형이 있습니다.")
    if not 1 <= int(spice_level) <= 5:
        raise ValueError("매운맛 선호도는 1에서 5 사이여야 합니다.")
    query = " ".join([*likes, *likes, *types])
    similarities, vector_result = cosine_scores(
        query,
        frame["menu_text"].astype(str).tolist(),
        method=method,
        embedder=embedder,
    )
    result = frame.copy()
    scores: list[float] = []
    reasons: list[str] = []
    for position, row in result.reset_index(drop=True).iterrows():
        menu_text = str(row["menu_text"])
        lowered = menu_text.casefold()
        type_hits = [value for value in types if value in _menu_types(menu_text)]
        avoid_hits = [value for value in avoids if value.casefold() in lowered]
        spice_difference = abs(int(spice_level) - _spice_level(menu_text))
        type_bonus = 15.0 if type_hits else 0.0
        score = float(similarities[position]) * 70.0
        score += type_bonus - 2.5 * spice_difference - 20.0 * len(avoid_hits)
        scores.append(round(float(np.clip(score, 0.0, 100.0)), 1))
        reason = [f"콘텐츠 유사도 {similarities[position]:.2f}"]
        if type_hits:
            reason.append(f"선호 유형 {', '.join(type_hits)}")
        if avoid_hits:
            reason.append(f"기피 키워드 {', '.join(avoid_hits)}")
        reason.append(f"매운맛 차이 {spice_difference}")
        reasons.append(" · ".join(reason))
    result["content_score"] = scores
    result["content_similarity"] = np.round(similarities, 4)
    result["prediction_reason"] = reasons
    result.attrs["vector_backend"] = vector_result.backend
    result.attrs["vector_device"] = vector_result.device
    return result, vector_result.notice


def best_worst_menus(
    scored: pd.DataFrame, *, top_n: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """겹치지 않는 Best와 Worst 메뉴를 반환한다."""

    if top_n < 1 or "content_score" not in scored:
        raise ValueError("예측 점수와 1개 이상의 결과 수가 필요합니다.")
    best = scored.sort_values(
        ["content_score", "date"], ascending=[False, True]
    ).head(top_n)
    remaining = scored.drop(index=best.index)
    worst = remaining.sort_values(
        ["content_score", "date"], ascending=[True, True]
    ).head(top_n)
    return best, worst


def school_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """학교별 급식 일수·메뉴 빈도·평균 열량을 요약한다."""

    records: list[dict] = []
    for school_name, group in frame.groupby("school_name", sort=True):
        dishes = [dish for values in group["dishes"] for dish in values]
        counts = Counter(dishes)
        most = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        least = sorted(counts.items(), key=lambda item: (item[1], item[0]))[0]
        records.append(
            {
                "학교": school_name,
                "학교급": group["school_kind"].iloc[0],
                "급식 일수": int(group["date"].nunique()),
                "메뉴 항목 수": len(dishes),
                "평균 열량": round(float(group["calories"].mean()), 1),
                "최다 메뉴": most[0],
                "최다 횟수": most[1],
                "최소 메뉴": least[0],
                "최소 횟수": least[1],
            }
        )
    return pd.DataFrame.from_records(records)


def signature_terms(
    frame: pd.DataFrame, school_name: str, *, top_n: int = 10
) -> pd.DataFrame:
    """학교를 문서로 보고 메뉴 이름의 학교별 TF-IDF를 계산한다."""

    school_counters: dict[str, Counter[str]] = {}
    for name, group in frame.groupby("school_name"):
        school_counters[str(name)] = Counter(
            dish for values in group["dishes"] for dish in values
        )
    if school_name not in school_counters:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    document_frequency = Counter()
    for counter in school_counters.values():
        document_frequency.update(counter.keys())
    counter = school_counters[school_name]
    total = sum(counter.values()) or 1
    records = []
    for term, count in counter.items():
        idf = math.log(
            (1 + len(school_counters)) / (1 + document_frequency[term])
        ) + 1.0
        records.append(
            {
                "시그니처 메뉴": term,
                "등장 횟수": count,
                "TF-IDF": round((count / total) * idf, 4),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["TF-IDF", "시그니처 메뉴"], ascending=[False, True]
    ).head(top_n).reset_index(drop=True)


def _school_food_counters(
    frame: pd.DataFrame,
) -> tuple[dict[str, Counter[str]], dict[str, str]]:
    """학교별 음식 횟수와 학교급을 검증해 한 번에 만든다."""

    required = {"school_name", "school_kind", "dishes"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("학교별로 분석할 급식 데이터가 없습니다.")
    counters: dict[str, Counter[str]] = {}
    school_kinds: dict[str, str] = {}
    for school_name, group in frame.groupby("school_name", sort=True):
        name = str(school_name)
        counter = Counter(
            str(dish).strip()
            for dishes in group["dishes"]
            for dish in dishes
            if str(dish).strip()
        )
        if counter:
            counters[name] = counter
            school_kinds[name] = str(group["school_kind"].iloc[0])
    if not counters:
        raise ValueError("학교별로 분석할 음식 항목이 없습니다.")
    return counters, school_kinds


def school_food_values(
    frame: pd.DataFrame,
    school_name: str | None = None,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """학교를 문서로 보고 음식별 TF·IDF·데이터 가치 점수를 모두 보인다."""

    if top_n is not None and int(top_n) < 1:
        raise ValueError("표시할 음식 수는 1개 이상이어야 합니다.")
    counters, school_kinds = _school_food_counters(frame)
    if school_name is not None and str(school_name) not in counters:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    selected = [str(school_name)] if school_name is not None else sorted(counters)
    school_count = len(counters)
    document_frequency: Counter[str] = Counter()
    for counter in counters.values():
        document_frequency.update(counter.keys())

    records: list[dict[str, object]] = []
    for name in selected:
        counter = counters[name]
        total_items = sum(counter.values())
        for food, count in counter.items():
            tf = count / total_items
            food_school_count = document_frequency[food]
            idf = math.log(
                (1 + school_count) / (1 + food_school_count)
            ) + 1.0
            records.append(
                {
                    "학교": name,
                    "학교급": school_kinds[name],
                    "음식": food,
                    "등장 횟수": int(count),
                    "학교 전체 음식 수": int(total_items),
                    "TF": tf,
                    "등장 학교 수": int(food_school_count),
                    "전체 학교 수": int(school_count),
                    "IDF": idf,
                    "데이터 가치 점수": tf * idf,
                }
            )
    result = pd.DataFrame.from_records(records).sort_values(
        ["학교", "데이터 가치 점수", "등장 횟수", "음식"],
        ascending=[True, False, False, True],
    )
    result.insert(2, "순위", result.groupby("학교").cumcount() + 1)
    if top_n is not None:
        result = result.loc[result["순위"] <= int(top_n)]
    for column in ("TF", "IDF", "데이터 가치 점수"):
        result[column] = result[column].round(4)
    return result.reset_index(drop=True)


def global_food_values(
    frame: pd.DataFrame, *, top_n: int | None = None
) -> pd.DataFrame:
    """모든 학교의 같은 음식을 합쳐 전체 TF-IDF 가치 순위를 만든다."""

    if top_n is not None and int(top_n) < 1:
        raise ValueError("표시할 음식 수는 1개 이상이어야 합니다.")
    counters, _ = _school_food_counters(frame)
    school_count = len(counters)
    total_items = sum(sum(counter.values()) for counter in counters.values())
    foods = sorted({food for counter in counters.values() for food in counter})
    records: list[dict[str, object]] = []
    for food in foods:
        school_counts = {
            school: int(counter[food])
            for school, counter in counters.items()
            if counter[food] > 0
        }
        count = sum(school_counts.values())
        food_school_count = len(school_counts)
        tf = count / total_items
        idf = math.log((1 + school_count) / (1 + food_school_count)) + 1.0
        leading_school, leading_count = sorted(
            school_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        records.append(
            {
                "음식": food,
                "전체 등장 횟수": int(count),
                "전체 음식 수": int(total_items),
                "전체 TF": tf,
                "등장 학교 수": int(food_school_count),
                "전체 학교 수": int(school_count),
                "IDF": idf,
                "전체 데이터 가치 점수": tf * idf,
                "가장 많이 나온 학교": leading_school,
                "해당 학교 횟수": int(leading_count),
            }
        )
    result = pd.DataFrame.from_records(records).sort_values(
        ["전체 데이터 가치 점수", "전체 등장 횟수", "음식"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result.insert(0, "전체 순위", range(1, len(result) + 1))
    if top_n is not None:
        result = result.head(int(top_n)).copy()
    for column in ("전체 TF", "IDF", "전체 데이터 가치 점수"):
        result[column] = result[column].round(4)
    return result.reset_index(drop=True)


def global_food_value_explanation(values: pd.DataFrame) -> str:
    """전역 1위 음식에 실제 수를 대입한 Markdown 계산식을 만든다."""

    required = {
        "음식",
        "전체 등장 횟수",
        "전체 음식 수",
        "등장 학교 수",
        "전체 학교 수",
        "가장 많이 나온 학교",
        "해당 학교 횟수",
    }
    if values.empty or not required.issubset(values.columns):
        raise ValueError("설명할 전체 음식 가치 결과가 없습니다.")
    row = values.sort_values("전체 순위").iloc[0]
    count = int(row["전체 등장 횟수"])
    total = int(row["전체 음식 수"])
    food_school_count = int(row["등장 학교 수"])
    school_count = int(row["전체 학교 수"])
    tf = count / total
    idf = math.log((1 + school_count) / (1 + food_school_count)) + 1.0
    value = tf * idf
    return (
        f"### 전체 음식 중요도 1위: {row['음식']}\n"
        f"- 전체 TF = {count} ÷ {total} = **{tf:.4f}**\n"
        f"- IDF = ln((1 + {school_count}) ÷ (1 + {food_school_count})) + 1 "
        f"= **{idf:.4f}**\n"
        f"- 전체 데이터 가치 점수 = {tf:.4f} × {idf:.4f} "
        f"= **{value:.4f}**\n"
        f"- 가장 많이 나온 학교: **{row['가장 많이 나온 학교']}에서 "
        f"{int(row['해당 학교 횟수'])}회**"
    )


def school_food_value_explanation(values: pd.DataFrame) -> str:
    """TF-IDF 1위 음식에 실제 수를 대입한 Markdown 계산식을 만든다."""

    required = {
        "학교",
        "음식",
        "등장 횟수",
        "학교 전체 음식 수",
        "등장 학교 수",
        "전체 학교 수",
    }
    if values.empty or not required.issubset(values.columns):
        raise ValueError("설명할 TF-IDF 음식 가치 결과가 없습니다.")
    row = values.sort_values(["학교", "순위"]).iloc[0]
    count = int(row["등장 횟수"])
    total = int(row["학교 전체 음식 수"])
    food_school_count = int(row["등장 학교 수"])
    school_count = int(row["전체 학교 수"])
    tf = count / total
    idf = math.log((1 + school_count) / (1 + food_school_count)) + 1.0
    value = tf * idf
    return (
        f"### {row['학교']}의 TF-IDF 1위: {row['음식']}\n"
        f"- TF = {count} ÷ {total} = **{tf:.4f}**\n"
        f"- IDF = ln((1 + {school_count}) ÷ (1 + {food_school_count})) + 1 "
        f"= **{idf:.4f}**\n"
        f"- TF-IDF 데이터 가치 점수 = {tf:.4f} × {idf:.4f} "
        f"= **{value:.4f}**"
    )


def school_food_frequencies(
    frame: pd.DataFrame, school_name: str, *, top_n: int = 15
) -> pd.DataFrame:
    """선택 학교에서 실제로 자주 나온 음식을 횟수 순서로 보여 준다."""

    if int(top_n) < 1:
        raise ValueError("표시할 음식 수는 1개 이상이어야 합니다.")
    counters, _ = _school_food_counters(frame)
    name = str(school_name)
    if name not in counters:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    rows = sorted(counters[name].items(), key=lambda item: (-item[1], item[0]))
    return pd.DataFrame(
        [
            {"순위": rank, "음식": food, "등장 횟수": int(count)}
            for rank, (food, count) in enumerate(rows[: int(top_n)], 1)
        ]
    )


def sample_school_foods(
    frame: pd.DataFrame,
    school_name: str,
    *,
    sample_size: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """선택 학교의 실제 음식에서 재현 가능한 평점 질문을 중복 없이 뽑는다."""

    if int(sample_size) < 1:
        raise ValueError("설문 음식 수는 1개 이상이어야 합니다.")
    counters, _ = _school_food_counters(frame)
    name = str(school_name)
    if name not in counters:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    foods = np.asarray(sorted(counters[name]), dtype=object)
    count = min(int(sample_size), len(foods))
    rng = np.random.default_rng(int(seed))
    selected = [str(value) for value in rng.choice(foods, size=count, replace=False)]
    return pd.DataFrame({"음식": selected, "평점": [3] * count})


def _validated_food_ratings(
    ratings: pd.DataFrame, available_foods: set[str]
) -> pd.DataFrame:
    if not isinstance(ratings, pd.DataFrame) or not {"음식", "평점"}.issubset(
        ratings.columns
    ):
        raise ValueError("음식과 평점 두 열이 있는 설문 표가 필요합니다.")
    result = ratings[["음식", "평점"]].copy()
    result["음식"] = result["음식"].astype(str).str.strip()
    if len(result) < 2:
        raise ValueError("역행렬 추천에는 평가 음식이 2개 이상 필요합니다.")
    if result["음식"].eq("").any() or result["음식"].duplicated().any():
        raise ValueError("평점 표의 음식은 비어 있거나 중복될 수 없습니다.")
    if not set(result["음식"]).issubset(available_foods):
        raise ValueError("평점 표에는 선택 학교의 실제 급식 음식만 사용할 수 있습니다.")
    numeric = pd.to_numeric(result["평점"], errors="coerce")
    if numeric.isna().any() or not numeric.between(1, 5).all():
        raise ValueError("음식 평점은 1에서 5 사이의 숫자여야 합니다.")
    result["평점"] = numeric.astype(float)
    return result.reset_index(drop=True)


def inverse_matrix_recommendations(
    frame: pd.DataFrame,
    school_name: str,
    ratings: pd.DataFrame,
    *,
    regularization: float = 0.1,
) -> pd.DataFrame:
    """30개 음식 평점에서 정규화 의사역행렬로 미평가 음식을 예측한다."""

    if not math.isfinite(float(regularization)) or float(regularization) <= 0:
        raise ValueError("정규화 값은 0보다 큰 숫자여야 합니다.")
    counters, _ = _school_food_counters(frame)
    name = str(school_name)
    if name not in counters:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    foods = sorted(counters[name])
    food_positions = {food: index for index, food in enumerate(foods)}
    survey = _validated_food_ratings(ratings, set(foods))
    vector_result = encode_texts(foods, method="tfidf")
    all_vectors = vector_result.matrix
    if all_vectors.shape[1] == 0:
        raise ValueError("음식 이름에서 TF-IDF 글자 특징을 만들 수 없습니다.")

    rated_names = survey["음식"].tolist()
    rated_indices = [food_positions[food] for food in rated_names]
    rated_vectors = all_vectors[rated_indices]
    rating_values = survey["평점"].to_numpy(dtype=float)
    centered = rating_values - 3.0
    lambda_value = float(regularization)
    gram = rated_vectors @ rated_vectors.T + lambda_value * np.eye(len(survey))
    coefficients = np.linalg.pinv(gram) @ centered
    weights = rated_vectors.T @ coefficients
    predictions = np.clip(3.0 + all_vectors @ weights, 1.0, 5.0)
    similarities = all_vectors @ rated_vectors.T
    contributions = similarities * coefficients

    rated_set = set(rated_names)
    records: list[dict[str, object]] = []
    for index, food in enumerate(foods):
        if food in rated_set:
            continue
        influence_index = int(np.argmax(np.abs(contributions[index])))
        records.append(
            {
                "음식": food,
                "예상 평점": round(float(predictions[index]), 2),
                "가장 영향 준 평가 음식": rated_names[influence_index],
                "그 음식 평점": float(rating_values[influence_index]),
                "유사도": round(float(similarities[index, influence_index]), 4),
                "등장 횟수": int(counters[name][food]),
            }
        )
    result = pd.DataFrame.from_records(
        records,
        columns=[
            "음식",
            "예상 평점",
            "가장 영향 준 평가 음식",
            "그 음식 평점",
            "유사도",
            "등장 횟수",
        ],
    )
    if not result.empty:
        result = result.sort_values(
            ["예상 평점", "등장 횟수", "음식"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    result.attrs.update(
        {
            "school_name": name,
            "rated_count": len(survey),
            "feature_count": int(all_vectors.shape[1]),
            "gram_shape": tuple(int(value) for value in gram.shape),
            "regularization": lambda_value,
            "formula": "w = Xᵀ pinv(XXᵀ + λI)(y - 3)",
            "vector_notice": vector_result.notice,
        }
    )
    return result


def food_map_coordinates(
    frame: pd.DataFrame,
    school_name: str,
    ratings: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    max_items: int = 50,
) -> pd.DataFrame:
    """직접 평가·예측·빈도 음식을 결정적 TF-IDF PCA 좌표에 놓는다."""

    if int(max_items) < 2:
        raise ValueError("음식 지도에는 음식이 2개 이상 필요합니다.")
    counters, _ = _school_food_counters(frame)
    name = str(school_name)
    if name not in counters:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    survey = _validated_food_ratings(ratings, set(counters[name]))
    recommendation_columns = {"음식", "예상 평점"}
    if not isinstance(recommendations, pd.DataFrame) or not recommendation_columns.issubset(
        recommendations.columns
    ):
        raise ValueError("음식 지도에 사용할 역행렬 추천 결과가 없습니다.")

    rated_names = survey["음식"].tolist()
    recommended_names = [
        str(value)
        for value in recommendations.sort_values(
            ["예상 평점", "음식"], ascending=[False, True]
        )["음식"].tolist()
    ]
    frequent_names = [
        food
        for food, _ in sorted(
            counters[name].items(), key=lambda item: (-item[1], item[0])
        )
    ]
    item_limit = int(max_items)
    selected: list[str] = []
    for food in rated_names:
        if food not in selected:
            selected.append(food)
        if len(selected) >= item_limit:
            break
    remaining_slots = item_limit - len(selected)
    frequency_candidates = [food for food in frequent_names if food not in selected]
    frequency_quota = (
        min(5, max(1, remaining_slots // 4), len(frequency_candidates))
        if remaining_slots > 0
        else 0
    )
    frequency_core_names = set(frequency_candidates[:frequency_quota])
    recommendation_candidates = [
        food
        for food in recommended_names
        if food not in selected and food not in frequency_core_names
    ]
    recommendation_quota = remaining_slots - len(frequency_core_names)
    selected.extend(recommendation_candidates[:recommendation_quota])
    selected.extend(frequency_candidates[:frequency_quota])
    for food in [*recommendation_candidates[recommendation_quota:], *frequent_names]:
        if len(selected) >= item_limit:
            break
        if food not in selected:
            selected.append(food)
    if len(selected) < 2:
        raise ValueError("음식 지도를 만들 실제 음식이 2개 이상 필요합니다.")

    vectors = encode_texts(selected, method="tfidf").matrix
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coordinates = np.zeros((len(selected), 2), dtype=float)
    component_count = min(2, len(singular_values))
    for component in range(component_count):
        loading = vt[component]
        reference = loading[int(np.argmax(np.abs(loading)))] if loading.size else 1.0
        sign = -1.0 if reference < 0 else 1.0
        coordinates[:, component] = (
            u[:, component] * singular_values[component] * sign
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("음식 지도의 2차원 좌표를 안정적으로 계산하지 못했습니다.")

    actual_ratings = dict(zip(survey["음식"], survey["평점"], strict=True))
    predicted_ratings = dict(
        zip(
            recommendations["음식"].astype(str),
            pd.to_numeric(recommendations["예상 평점"], errors="coerce"),
            strict=True,
        )
    )
    records = []
    for index, food in enumerate(selected):
        if food in actual_ratings:
            category = "직접 평가"
            rating = float(actual_ratings[food])
        elif food in frequency_core_names:
            category = "빈도 핵심"
            predicted = predicted_ratings.get(food, float("nan"))
            rating = float(predicted) if math.isfinite(predicted) else 3.0
        elif food in predicted_ratings and math.isfinite(predicted_ratings[food]):
            category = "역행렬 추천"
            rating = float(predicted_ratings[food])
        else:
            category = "빈도 핵심"
            rating = 3.0
        records.append(
            {
                "번호": index + 1,
                "음식": food,
                "X": round(float(coordinates[index, 0]), 6),
                "Y": round(float(coordinates[index, 1]), 6),
                "구분": category,
                "평점": round(rating, 2),
                "등장 횟수": int(counters[name][food]),
            }
        )
    result = pd.DataFrame.from_records(records)
    result.attrs["school_name"] = name
    result.attrs["vector_notice"] = "문자 n-gram TF-IDF를 SVD/PCA 두 축으로 줄였습니다."
    return result


def food_mbti(
    *,
    rice_vs_noodle: int,
    mild_vs_spicy: int,
    familiar_vs_new: int,
    dessert_vs_hearty: int,
) -> tuple[str, str]:
    """네 개의 식성 질문을 설명 가능한 네 글자 유형으로 바꾼다."""

    values = (rice_vs_noodle, mild_vs_spicy, familiar_vs_new, dessert_vs_hearty)
    if any(not 1 <= int(value) <= 5 for value in values):
        raise ValueError("Food MBTI 응답은 1에서 5 사이여야 합니다.")
    axes = [
        ("R", "E", "밥·국 중심", "면·간편식 중심"),
        ("M", "S", "순한 맛", "매운 맛"),
        ("F", "L", "익숙한 메뉴", "새로운 메뉴"),
        ("D", "H", "디저트", "든든한 식사"),
    ]
    letters = []
    descriptions = []
    for value, (left, right, left_text, right_text) in zip(values, axes):
        choose_right = int(value) >= 4
        letters.append(right if choose_right else left)
        descriptions.append(right_text if choose_right else left_text)
    return "".join(letters), " · ".join(descriptions)


def meal_buddies(
    profiles_text: str,
    *,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """최대 6명의 익명 취향 벡터로 가장 가까운 학생 쌍을 찾는다."""

    records = []
    for line in str(profiles_text or "").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError("각 줄을 익명코드|좋아하는 메뉴|피하는 메뉴로 입력하세요.")
        records.append(parts)
    if not 2 <= len(records) <= 6:
        raise ValueError("밥친구 분석에는 익명 학생 2명에서 6명이 필요합니다.")
    names = [record[0] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("익명 참여자 코드가 중복되었습니다.")
    texts = [f"{likes} 좋아함 {avoids} 피함" for _, likes, avoids in records]
    vectors = encode_texts(texts, method=method, embedder=embedder).matrix
    similarities = vectors @ vectors.T
    pair_records = [
        {
            "학생 1": names[left],
            "학생 2": names[right],
            "식성 유사도": round(float(similarities[left, right]), 4),
        }
        for left, right in combinations(range(len(names)), 2)
    ]
    pairs = pd.DataFrame(pair_records).sort_values(
        ["식성 유사도", "학생 1", "학생 2"], ascending=[False, True, True]
    ).reset_index(drop=True)
    matrix = pd.DataFrame(np.round(similarities, 4), columns=names)
    matrix.insert(0, "참여자", names)
    return pairs, matrix


def validate_feedback_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """익명 설문 CSV의 열과 값 범위를 검증한다."""

    missing = set(FEEDBACK_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"설문 데이터에 필요한 열이 없습니다: {', '.join(sorted(missing))}")
    result = frame[FEEDBACK_COLUMNS].copy()
    if result.empty:
        raise ValueError("설문 응답이 없습니다.")
    if result["participant_code"].nunique() > 6:
        raise ValueError("한 모둠은 익명 참여자 6명까지 분석할 수 있습니다.")
    if not result["participant_code"].astype(str).str.fullmatch(
        r"[가-힣A-Za-z0-9_-]{1,12}"
    ).all():
        raise ValueError("익명 참여자 코드는 1~12자의 글자·숫자로 입력하세요.")
    if not set(result["grade_band"].astype(str)).issubset(GRADE_BANDS):
        raise ValueError("학년 구간이 올바르지 않습니다.")
    if not set(result["feedback_tag"].astype(str)).issubset(FEEDBACK_TAGS):
        raise ValueError("피드백 태그가 올바르지 않습니다.")
    if not set(result["content_method"].astype(str)).issubset(
        {"tfidf", "embedding"}
    ):
        raise ValueError("콘텐츠 분석 방식이 올바르지 않습니다.")
    for column in CSV_TEXT_COLUMNS:
        if result[column].map(
            lambda value: str(value).lstrip().startswith(("=", "+", "-", "@"))
        ).any():
            raise ValueError("CSV 문자 값은 수식 기호(=, +, -, @)로 시작할 수 없습니다.")
    for column, low, high, message in (
        ("spice_level", 1, 5, "매운맛 선호도는 1에서 5 사이여야 합니다."),
        ("content_prediction", 0, 100, "콘텐츠 예측은 0에서 100 사이여야 합니다."),
        ("actual_rating", 1, 5, "실제 만족도는 1에서 5 사이여야 합니다."),
    ):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not values.between(low, high).all():
            raise ValueError(message)
        result[column] = values
    if result.duplicated(["participant_code", "menu_id"]).any():
        raise ValueError("같은 참여자와 식단의 응답이 중복되었습니다.")
    return result.reset_index(drop=True)


@dataclass(frozen=True)
class UserPrediction:
    rating: float
    common_neighbor_count: int
    fallback: str


def user_based_prediction(
    feedback: pd.DataFrame,
    *,
    target_participant: str,
    target_menu_id: str,
) -> UserPrediction:
    """대상 평점을 숨긴 뒤 유사 학생의 실제 리뷰로 1~5점을 예측한다."""

    data = validate_feedback_frame(feedback)
    visible = data.loc[
        ~(
            (data["participant_code"] == target_participant)
            & (data["menu_id"] == target_menu_id)
        )
    ]
    target_history = visible.loc[
        visible["participant_code"] == target_participant,
        ["menu_id", "actual_rating"],
    ].set_index("menu_id")["actual_rating"]
    candidate_rows = visible.loc[visible["menu_id"] == target_menu_id]
    weighted_sum = 0.0
    weight_total = 0.0
    common_neighbors = 0
    for _, candidate in candidate_rows.iterrows():
        other = str(candidate["participant_code"])
        other_history = visible.loc[
            (visible["participant_code"] == other)
            & (visible["menu_id"] != target_menu_id),
            ["menu_id", "actual_rating"],
        ].set_index("menu_id")["actual_rating"]
        common = target_history.index.intersection(other_history.index)
        if len(common) == 0:
            continue
        common_neighbors += 1
        differences = [
            abs(float(target_history[item]) - float(other_history[item]))
            for item in common
        ]
        similarity = max(0.0, 1.0 - float(np.mean(differences)) / 4.0)
        weighted_sum += similarity * float(candidate["actual_rating"])
        weight_total += similarity
    if weight_total > 0:
        return UserPrediction(
            rating=weighted_sum / weight_total,
            common_neighbor_count=common_neighbors,
            fallback="유사 학생 가중 평균",
        )
    if not candidate_rows.empty:
        return UserPrediction(
            rating=float(candidate_rows["actual_rating"].mean()),
            common_neighbor_count=common_neighbors,
            fallback="해당 식단 모둠 평균",
        )
    if not visible.empty:
        return UserPrediction(
            rating=float(visible["actual_rating"].mean()),
            common_neighbor_count=0,
            fallback="모든 리뷰 평균",
        )
    return UserPrediction(3.0, 0, "기본값")


def evaluate_recommenders(
    feedback: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """각 실제 리뷰를 가리고 콘텐츠·유저·혼합 추천 오차를 비교한다."""

    data = validate_feedback_frame(feedback)
    records = []
    for _, row in data.iterrows():
        user = user_based_prediction(
            data,
            target_participant=str(row["participant_code"]),
            target_menu_id=str(row["menu_id"]),
        )
        user_score = round(float(np.clip(user.rating * 20.0, 0.0, 100.0)), 1)
        content_score = float(row["content_prediction"])
        content_weight = 0.5 if user.common_neighbor_count > 0 else 0.8
        hybrid = round(
            content_weight * content_score + (1.0 - content_weight) * user_score,
            1,
        )
        actual = float(row["actual_rating"]) * 20.0
        records.append(
            {
                **row.to_dict(),
                "user_prediction": user_score,
                "hybrid_prediction": hybrid,
                "actual_score": actual,
                "common_neighbor_count": user.common_neighbor_count,
                "user_fallback": user.fallback,
                "content_error": abs(content_score - actual),
                "user_error": abs(user_score - actual),
                "hybrid_error": abs(hybrid - actual),
            }
        )
    detail = pd.DataFrame.from_records(records)
    summary = pd.DataFrame(
        [
            {"추천 방식": "콘텐츠 기반", "MAE": round(detail["content_error"].mean(), 2)},
            {"추천 방식": "유저 기반", "MAE": round(detail["user_error"].mean(), 2)},
            {"추천 방식": "혼합", "MAE": round(detail["hybrid_error"].mean(), 2)},
        ]
    )
    return detail, summary


def pareto_candidates(
    frame: pd.DataFrame,
    *,
    group_preferences: Sequence[str],
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> pd.DataFrame:
    """실제 원가·잔반 대신 공개한 대체지표의 비지배 식단을 찾는다."""

    if not group_preferences:
        raise ValueError("익명 집단 취향을 하나 이상 입력하세요.")
    score_columns = []
    vector_notices = []
    for preference in group_preferences:
        scored, vector_notice = predict_satisfaction(
            frame,
            likes_text=preference,
            avoids_text="",
            preferred_types=[],
            spice_level=3,
            method=method,
            embedder=embedder,
        )
        score_columns.append(scored["content_score"].to_numpy(dtype=float))
        vector_notices.append(vector_notice)
    result = frame.copy().reset_index(drop=True)
    group_scores = np.mean(np.vstack(score_columns), axis=0)
    result["모둠 예측 만족도"] = np.round(group_scores, 1)
    result["가성비 대체지표"] = np.round(
        np.clip(group_scores / result["dish_count"].clip(lower=1) * 3.0, 0, 100), 1
    )
    result["잔반 위험 대체지표"] = np.rint(100.0 - group_scores).astype(int)
    calories = pd.to_numeric(result["calories"], errors="coerce")
    calorie_center = float(calories.mean()) if calories.notna().any() else 0.0
    result["영양 편차 대체지표"] = np.round((calories - calorie_center).abs().fillna(0), 1)
    keep = []
    for index, row in result.iterrows():
        dominated = False
        for other_index, other in result.iterrows():
            if index == other_index:
                continue
            no_worse = (
                other["가성비 대체지표"] >= row["가성비 대체지표"]
                and other["잔반 위험 대체지표"] <= row["잔반 위험 대체지표"]
                and other["영양 편차 대체지표"] <= row["영양 편차 대체지표"]
            )
            strictly_better = (
                other["가성비 대체지표"] > row["가성비 대체지표"]
                or other["잔반 위험 대체지표"] < row["잔반 위험 대체지표"]
                or other["영양 편차 대체지표"] < row["영양 편차 대체지표"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    output = result.loc[keep].sort_values(
        ["잔반 위험 대체지표", "가성비 대체지표"], ascending=[True, False]
    )
    output.attrs["notice"] = (
        "수업용 대체지표입니다. 실제 식재료 원가나 잔반 측정값이 아닙니다."
    )
    output.attrs["vector_notice"] = " ".join(dict.fromkeys(vector_notices))
    return output.reset_index(drop=True)


def menu_pair_scores(
    frame: pd.DataFrame,
    *,
    preference_text: str,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> pd.DataFrame:
    """같은 날 나온 메뉴 쌍의 빈도와 취향 유사도로 탐색 점수를 만든다."""

    counts: Counter[tuple[str, str]] = Counter()
    for dishes in frame["dishes"]:
        unique = sorted(set(str(dish) for dish in dishes))
        counts.update(combinations(unique, 2))
    pair_texts = [" ".join(pair) for pair in counts]
    similarities, vector_result = cosine_scores(
        preference_text, pair_texts, method=method, embedder=embedder
    )
    records = [
        {
            "메뉴 조합": " + ".join(pair),
            "같이 나온 횟수": count,
            "취향 유사도": round(float(similarities[index]), 4),
            "조합 탐색 점수": round(count * 10 + float(similarities[index]) * 90, 1),
        }
        for index, (pair, count) in enumerate(counts.items())
    ]
    result = pd.DataFrame(records).sort_values(
        ["조합 탐색 점수", "메뉴 조합"], ascending=[False, True]
    )
    result.attrs["notice"] = (
        "동시 등장과 취향 유사도로 만든 수업용 추정치이며 실제 꿀조합·꽝조합 판정이 아닙니다."
    )
    result.attrs["vector_notice"] = vector_result.notice
    return result.reset_index(drop=True)


def feedback_to_csv(feedback: pd.DataFrame, path: str | Path) -> Path:
    """검증한 익명 설문을 사용자가 지정한 CSV로 내보낸다."""

    data = validate_feedback_frame(feedback)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8-sig", lineterminator="\n")
    return output


def feedback_from_csv(path: str | Path) -> pd.DataFrame:
    """내보낸 익명 설문 CSV를 같은 스키마로 읽고 다시 검증한다."""

    try:
        frame = pd.read_csv(
            Path(path),
            encoding="utf-8-sig",
            keep_default_na=False,
            dtype={
                "participant_code": str,
                "grade_band": str,
                "menu_id": str,
                "meal_date": str,
            },
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("익명 설문 CSV를 읽을 수 없습니다.") from exc
    return validate_feedback_frame(frame)


def analyze_feedback(
    feedback: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """실제 리뷰와 세 추천 방식의 오차를 모둠 단위로 요약한다."""

    detail, summary = evaluate_recommenders(feedback)
    tag_counts = (
        detail.groupby("feedback_tag", as_index=False)
        .size()
        .rename(columns={"feedback_tag": "피드백 태그", "size": "응답 수"})
        .sort_values(["응답 수", "피드백 태그"], ascending=[False, True])
        .reset_index(drop=True)
    )
    actual_average = float(detail["actual_rating"].mean())
    best_method = summary.sort_values(["MAE", "추천 방식"]).iloc[0]
    message = (
        f"익명 참여자 {detail['participant_code'].nunique()}명, 리뷰 {len(detail)}개를 "
        f"분석했습니다. 실제 만족도 평균은 5점 만점에 {actual_average:.2f}점입니다. "
        f"현재 가장 오차가 작은 방식은 {best_method['추천 방식']}"
        f"(MAE {best_method['MAE']:.2f})입니다. "
        "6명 이하의 작은 표본이므로 다른 학생 전체로 일반화하지 않습니다."
    )
    return message, detail, summary, tag_counts


def cluster_feedback(
    feedback: pd.DataFrame,
    *,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> pd.DataFrame:
    """학생별 취향 문장을 결정적 두 군집으로 묶는다."""

    data = validate_feedback_frame(feedback)
    profiles = (
        data.sort_values(["participant_code", "meal_date"])
        .groupby("participant_code", as_index=False)
        .agg(
            likes_text=("likes_text", "first"),
            avoids_text=("avoids_text", "first"),
            grade_band=("grade_band", "first"),
        )
    )
    if len(profiles) < 2:
        raise ValueError("식성 군집에는 익명 참여자 2명 이상이 필요합니다.")
    texts = [
        f"{row.likes_text} 좋아함 {row.avoids_text} 피함"
        for row in profiles.itertuples()
    ]
    matrix = encode_texts(texts, method=method, embedder=embedder).matrix
    distances_from_first = ((matrix - matrix[0]) ** 2).sum(axis=1)
    farthest = int(np.argmax(distances_from_first))
    if distances_from_first[farthest] == 0:
        labels = np.zeros(len(profiles), dtype=int)
    else:
        centroids = np.vstack([matrix[0], matrix[farthest]])
        labels = np.zeros(len(profiles), dtype=int)
        for _ in range(20):
            distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            new_centroids = centroids.copy()
            for cluster_id in (0, 1):
                members = matrix[new_labels == cluster_id]
                if len(members):
                    new_centroids[cluster_id] = members.mean(axis=0)
            if np.array_equal(labels, new_labels) and np.allclose(
                centroids, new_centroids
            ):
                labels = new_labels
                break
            labels, centroids = new_labels, new_centroids
    output = profiles.copy()
    output["food_cluster"] = [f"식성 군집 {int(label) + 1}" for label in labels]
    return output


def recommend_high_schools(
    frame: pd.DataFrame,
    preference_text: str,
    *,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> tuple[pd.DataFrame, str]:
    """교육 평가 없이 급식 취향 유사도만으로 고등학교를 정렬한다."""

    high = frame.loc[frame["school_kind"] == "고등학교"].copy()
    if high.empty:
        raise ValueError("고등학교 급식 데이터가 없습니다.")
    scores, vector_result = cosine_scores(
        preference_text,
        high["menu_text"].astype(str).tolist(),
        method=method,
        embedder=embedder,
    )
    high["similarity"] = scores
    records = []
    for school, group in high.groupby("school_name", sort=True):
        top_scores = group["similarity"].nlargest(min(5, len(group)))
        records.append(
            {
                "학교": school,
                "급식 취향 점수": round(float(top_scores.mean()) * 100.0, 1),
                "비교 급식 수": len(group),
            }
        )
    ranked = pd.DataFrame(records).sort_values(
        ["급식 취향 점수", "학교"], ascending=[False, True]
    ).reset_index(drop=True)
    notice = (
        f"{vector_result.notice} 이 순위는 급식 취향만 비교하며 교육의 질이나 "
        "진학 적합도를 평가하지 않습니다."
    )
    return ranked, notice
