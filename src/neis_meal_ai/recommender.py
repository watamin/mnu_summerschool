"""설명 가능한 개인 취향 기반 급식 추천과 식단 군집."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SAFETY_NOTICE = (
    "추천 결과는 취향 비교용입니다. 실제 식단과 알레르기 정보는 "
    "학교 급식표와 영양사 안내를 다시 확인하세요."
)

MENU_TYPE_KEYWORDS = {
    "밥": ("밥", "덮밥", "볶음밥", "비빔밥", "주먹밥"),
    "면": ("면", "국수", "우동", "라면", "스파게티", "파스타", "쫄면"),
    "국물": ("국", "탕", "찌개", "전골", "스프", "짬뽕"),
    "튀김": ("튀김", "돈까스", "커틀릿", "치킨", "꼬치"),
    "디저트": ("푸딩", "케이크", "과일", "바나나", "주스", "요구르트", "아이스", "우유"),
}

SPICY_KEYWORDS = {
    5: ("불닭", "마라", "매운", "핵", "아주매운"),
    4: ("짬뽕", "떡볶이", "고추", "낙지볶음"),
    3: ("김치", "제육", "비빔", "고추장"),
}


@dataclass(frozen=True)
class PreferenceProfile:
    """개인을 식별하지 않는 현재 실행용 취향 프로필."""

    likes: tuple[str, ...]
    avoids: tuple[str, ...]
    preferred_types: tuple[str, ...]
    spice_level: int
    allergy_codes: tuple[int, ...]


def _clean_terms(values: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in values:
        term = str(value).strip()
        if term and term not in cleaned:
            cleaned.append(term)
    return tuple(cleaned)


def validate_profile(profile: PreferenceProfile) -> PreferenceProfile:
    """프로필 범위를 확인하고 공백·중복을 정리한다."""

    likes = _clean_terms(profile.likes)
    avoids = _clean_terms(profile.avoids)
    preferred_types = _clean_terms(profile.preferred_types)
    allergies = tuple(sorted(set(int(code) for code in profile.allergy_codes)))
    if len(likes) > 5 or len(avoids) > 5:
        raise ValueError("좋아하거나 피하고 싶은 항목은 각각 최대 5개까지 입력하세요.")
    if not 1 <= int(profile.spice_level) <= 5:
        raise ValueError("매운맛 선호도는 1에서 5 사이여야 합니다.")
    unknown_types = set(preferred_types) - set(MENU_TYPE_KEYWORDS)
    if unknown_types:
        raise ValueError(f"지원하지 않는 메뉴 유형입니다: {', '.join(sorted(unknown_types))}")
    if any(code < 1 or code > 19 for code in allergies):
        raise ValueError("알레르기 주의 번호는 1에서 19 사이여야 합니다.")
    return PreferenceProfile(likes, avoids, preferred_types, int(profile.spice_level), allergies)


def _char_ngrams(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for word in text.casefold().split():
        padded = f" {word} "
        for size in (2, 3, 4):
            counts.update(padded[index : index + size] for index in range(len(padded) - size + 1))
    return counts


def _tfidf_similarity(menu_texts: list[str], query: str) -> np.ndarray:
    """작은 수업 데이터에 맞춘 문자 n-gram TF-IDF 코사인 유사도."""

    if not query.strip():
        return np.zeros(len(menu_texts), dtype=float)
    documents = [_char_ngrams(text) for text in [*menu_texts, query]]
    document_count = len(documents)
    document_frequency: Counter[str] = Counter()
    for counts in documents:
        document_frequency.update(counts.keys())
    idf = {
        term: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }

    def vector(counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {term: (count / total) * idf[term] for term, count in counts.items()}

    vectors = [vector(counts) for counts in documents]
    query_vector = vectors[-1]
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
    similarities: list[float] = []
    for menu_vector in vectors[:-1]:
        dot = sum(value * query_vector.get(term, 0.0) for term, value in menu_vector.items())
        menu_norm = math.sqrt(sum(value * value for value in menu_vector.values())) or 1.0
        similarities.append(dot / (menu_norm * query_norm))
    return np.asarray(similarities, dtype=float)


def _menu_types(menu_text: str) -> tuple[str, ...]:
    lowered = menu_text.casefold()
    return tuple(
        menu_type
        for menu_type, keywords in MENU_TYPE_KEYWORDS.items()
        if any(keyword.casefold() in lowered for keyword in keywords)
    )


def _spice_level(menu_text: str) -> int:
    lowered = menu_text.casefold()
    for level in (5, 4, 3):
        if any(keyword.casefold() in lowered for keyword in SPICY_KEYWORDS[level]):
            return level
    return 2


def cluster_meals(frame: pd.DataFrame, max_clusters: int = 3) -> pd.DataFrame:
    """영양 수치가 비슷한 식단을 작은 결정적 K-Means로 묶는다."""

    result = frame.copy()
    if len(result) < 3:
        result["cluster_name"] = "데이터 부족"
        return result

    columns = ["calories", "carbs_g", "protein_g", "fat_g", "dish_count"]
    features = result[columns].apply(pd.to_numeric, errors="coerce")
    nutrition_columns = ["calories", "carbs_g", "protein_g", "fat_g"]
    complete_nutrition_rows = int(features[nutrition_columns].notna().all(axis=1).sum())
    if complete_nutrition_rows < 3:
        result["cluster_name"] = "데이터 부족"
        return result
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    matrix = features.to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    standardized = (matrix - means) / stds

    unique_count = len(np.unique(standardized, axis=0))
    cluster_count = min(max(1, int(max_clusters)), len(result), unique_count)
    if cluster_count == 1:
        result["cluster_name"] = "중간 구성"
        return result

    order = np.argsort(features["calories"].to_numpy())
    positions = np.linspace(0, len(order) - 1, cluster_count).round().astype(int)
    centroids = standardized[order[positions]].copy()
    labels = np.zeros(len(result), dtype=int)
    for _ in range(30):
        distances = ((standardized[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(cluster_count):
            members = standardized[new_labels == cluster_id]
            if len(members):
                new_centroids[cluster_id] = members.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            labels = new_labels
            break
        labels, centroids = new_labels, new_centroids

    calorie_values = features["calories"].to_numpy(dtype=float)
    calorie_means = {
        cluster_id: float(calorie_values[labels == cluster_id].mean())
        for cluster_id in range(cluster_count)
        if np.any(labels == cluster_id)
    }
    sorted_clusters = sorted(calorie_means, key=calorie_means.get)
    if len(sorted_clusters) == 2:
        names = {
            sorted_clusters[0]: "상대적 가벼운 구성",
            sorted_clusters[1]: "상대적 든든한 구성",
        }
    else:
        names = {
            sorted_clusters[0]: "상대적 가벼운 구성",
            sorted_clusters[-1]: "상대적 든든한 구성",
            **{cluster_id: "중간 구성" for cluster_id in sorted_clusters[1:-1]},
        }
    result["cluster_name"] = [names[int(label)] for label in labels]
    return result


def _empty_result(frame: pd.DataFrame, excluded_count: int) -> pd.DataFrame:
    result = frame.iloc[0:0].copy()
    for column in ("score", "reason", "cluster_name", "safety_notice"):
        if column not in result:
            result[column] = pd.Series(dtype="object")
    result.attrs["excluded_count"] = excluded_count
    return result


def recommend_menus(
    frame: pd.DataFrame,
    profile: PreferenceProfile,
    top_n: int = 3,
) -> pd.DataFrame:
    """취향 유사도와 명시적 가감점을 합쳐 추천 순위를 만든다."""

    profile = validate_profile(profile)
    if top_n < 1:
        raise ValueError("추천 개수는 1개 이상이어야 합니다.")
    required = {"menu_text", "allergy_codes", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"급식 데이터에 필요한 열이 없습니다: {', '.join(sorted(missing))}")

    allergy_set = set(profile.allergy_codes)
    unsafe_mask = frame["allergy_codes"].apply(lambda codes: bool(allergy_set.intersection(set(codes))))
    excluded_count = int(unsafe_mask.sum())
    safe = frame.loc[~unsafe_mask].copy().reset_index(drop=True)
    if safe.empty:
        return _empty_result(frame, excluded_count)

    safe = cluster_meals(safe)
    query_parts = [*profile.likes, *profile.likes, *profile.preferred_types]
    query = " ".join(query_parts)
    similarities = _tfidf_similarity(safe["menu_text"].astype(str).tolist(), query)

    scores: list[float] = []
    reasons: list[str] = []
    for position, row in safe.iterrows():
        menu_text = str(row["menu_text"])
        lowered = menu_text.casefold()
        like_hits = [term for term in profile.likes if term.casefold() in lowered]
        avoid_hits = [term for term in profile.avoids if term.casefold() in lowered]
        menu_types = _menu_types(menu_text)
        type_hits = [menu_type for menu_type in profile.preferred_types if menu_type in menu_types]
        spice_difference = abs(profile.spice_level - _spice_level(menu_text))

        # 20점 기준점은 기피·매운맛 감점이 0점 하한에서도 보이게 하기 위한 공개 상수다.
        score = 20.0 + similarities[position] * 70.0
        score += 8.0 * len(like_hits)
        score += 5.0 * len(type_hits)
        score -= 18.0 * len(avoid_hits)
        score -= 3.0 * spice_difference
        scores.append(round(float(np.clip(score, 0.0, 100.0)), 1))

        reason_parts = [f"텍스트 유사도 {similarities[position]:.2f}"]
        if like_hits:
            reason_parts.append(f"좋아하는 키워드: {', '.join(like_hits)}")
        if type_hits:
            reason_parts.append(f"선호 유형: {', '.join(type_hits)}")
        if avoid_hits:
            reason_parts.append(f"피하고 싶은 키워드: {', '.join(avoid_hits)}")
        reason_parts.append(f"매운맛 차이 {spice_difference}")
        reasons.append(" · ".join(reason_parts))

    safe["score"] = scores
    safe["reason"] = reasons
    safe["safety_notice"] = SAFETY_NOTICE
    result = safe.sort_values(["score", "date"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    result.attrs["excluded_count"] = excluded_count
    return result
