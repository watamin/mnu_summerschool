from __future__ import annotations

import pandas as pd
import pytest

from neis_meal_ai.recommender import (
    PreferenceProfile,
    cluster_meals,
    recommend_menus,
    validate_profile,
)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-21",
                "school_name": "남악고등학교",
                "meal_type": "중식",
                "dishes": ["현미밥", "된장국", "제육볶음"],
                "menu_text": "현미밥 된장국 제육볶음",
                "allergy_codes": (5, 6, 10),
                "calories": 760.0,
                "carbs_g": 110.0,
                "protein_g": 32.0,
                "fat_g": 20.0,
                "dish_count": 3,
            },
            {
                "date": "2026-07-22",
                "school_name": "남악고등학교",
                "meal_type": "중식",
                "dishes": ["치즈스파게티", "옥수수스프", "자몽푸딩"],
                "menu_text": "치즈스파게티 옥수수스프 자몽푸딩",
                "allergy_codes": (1, 2, 5, 6, 10, 16),
                "calories": 930.0,
                "carbs_g": 140.0,
                "protein_g": 40.0,
                "fat_g": 29.0,
                "dish_count": 3,
            },
            {
                "date": "2026-07-23",
                "school_name": "남악고등학교",
                "meal_type": "중식",
                "dishes": ["쌀밥", "매운짬뽕국", "통새우튀김"],
                "menu_text": "쌀밥 매운짬뽕국 통새우튀김",
                "allergy_codes": (5, 6, 9, 17),
                "calories": 850.0,
                "carbs_g": 125.0,
                "protein_g": 35.0,
                "fat_g": 25.0,
                "dish_count": 3,
            },
            {
                "date": "2026-07-24",
                "school_name": "남악고등학교",
                "meal_type": "중식",
                "dishes": ["잔치국수", "오이무침", "바나나"],
                "menu_text": "잔치국수 오이무침 바나나",
                "allergy_codes": (1, 5, 6),
                "calories": 620.0,
                "carbs_g": 95.0,
                "protein_g": 22.0,
                "fat_g": 14.0,
                "dish_count": 3,
            },
        ]
    )


def test_validate_profile_rejects_identifying_or_excessive_values() -> None:
    with pytest.raises(ValueError, match="최대 5개"):
        validate_profile(
            PreferenceProfile(
                likes=("가", "나", "다", "라", "마", "바"),
                avoids=(),
                preferred_types=(),
                spice_level=3,
                allergy_codes=(),
            )
        )
    with pytest.raises(ValueError, match="1에서 5"):
        validate_profile(PreferenceProfile((), (), (), 6, ()))


def test_recommendation_prefers_matching_keyword_and_type(sample_frame: pd.DataFrame) -> None:
    profile = PreferenceProfile(
        likes=("스파게티", "치즈"),
        avoids=(),
        preferred_types=("면", "디저트"),
        spice_level=2,
        allergy_codes=(),
    )

    result = recommend_menus(sample_frame, profile, top_n=1)

    assert "치즈스파게티" in result.iloc[0]["menu_text"]
    assert 0 <= result.iloc[0]["score"] <= 100
    assert "좋아하는 키워드" in result.iloc[0]["reason"]


def test_allergy_match_is_excluded_and_counted(sample_frame: pd.DataFrame) -> None:
    profile = PreferenceProfile((), (), (), 3, (9,))

    result = recommend_menus(sample_frame, profile, top_n=10)

    assert all(9 not in codes for codes in result["allergy_codes"])
    assert result.attrs["excluded_count"] == 1


def test_avoid_keyword_reduces_score_and_is_explained(sample_frame: pd.DataFrame) -> None:
    neutral = recommend_menus(sample_frame, PreferenceProfile((), (), (), 3, ()), top_n=4)
    avoids = recommend_menus(sample_frame, PreferenceProfile((), ("오이",), (), 3, ()), top_n=4)

    neutral_score = neutral.loc[neutral["date"] == "2026-07-24", "score"].iloc[0]
    avoided_row = avoids.loc[avoids["date"] == "2026-07-24"].iloc[0]
    assert avoided_row["score"] < neutral_score
    assert "피하고 싶은 키워드: 오이" in avoided_row["reason"]


def test_all_candidates_can_be_safely_excluded(sample_frame: pd.DataFrame) -> None:
    profile = PreferenceProfile((), (), (), 3, (5,))
    result = recommend_menus(sample_frame, profile, top_n=3)

    assert result.empty
    assert result.attrs["excluded_count"] == 4
    assert "safety_notice" in result.columns


def test_cluster_meals_assigns_relative_non_medical_labels(sample_frame: pd.DataFrame) -> None:
    clustered = cluster_meals(sample_frame, max_clusters=3)

    assert clustered["cluster_name"].notna().all()
    assert set(clustered["cluster_name"]) <= {
        "상대적 가벼운 구성",
        "중간 구성",
        "상대적 든든한 구성",
    }
    lightest = clustered.loc[clustered["calories"].idxmin(), "cluster_name"]
    heaviest = clustered.loc[clustered["calories"].idxmax(), "cluster_name"]
    assert lightest == "상대적 가벼운 구성"
    assert heaviest == "상대적 든든한 구성"


def test_cluster_meals_marks_small_dataset_as_insufficient(sample_frame: pd.DataFrame) -> None:
    clustered = cluster_meals(sample_frame.iloc[:2], max_clusters=3)
    assert set(clustered["cluster_name"]) == {"데이터 부족"}


def test_cluster_meals_marks_missing_nutrition_as_insufficient(sample_frame: pd.DataFrame) -> None:
    missing = sample_frame.copy()
    missing[["calories", "carbs_g", "protein_g", "fat_g"]] = float("nan")

    clustered = cluster_meals(missing, max_clusters=3)

    assert set(clustered["cluster_name"]) == {"데이터 부족"}


def test_cluster_meals_requires_three_complete_nutrition_rows(sample_frame: pd.DataFrame) -> None:
    partial = sample_frame.copy()
    partial.loc[2:, "protein_g"] = float("nan")

    clustered = cluster_meals(partial, max_clusters=3)

    assert set(clustered["cluster_name"]) == {"데이터 부족"}


def test_neutral_profile_uses_documented_twenty_point_baseline(sample_frame: pd.DataFrame) -> None:
    result = recommend_menus(sample_frame, PreferenceProfile((), (), (), 2, ()), top_n=4)

    unspicy_score = result.loc[result["date"] == "2026-07-22", "score"].iloc[0]
    assert unspicy_score == 20.0
    assert result["score"].max() == 20.0
