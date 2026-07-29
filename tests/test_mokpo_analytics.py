from __future__ import annotations

import pandas as pd
import pytest

from neis_meal_ai.mokpo_analytics import (
    analyze_feedback,
    best_worst_menus,
    cluster_feedback,
    evaluate_recommenders,
    feedback_from_csv,
    feedback_to_csv,
    food_mbti,
    meal_buddies,
    pareto_candidates,
    predict_satisfaction,
    recommend_high_schools,
    school_statistics,
    signature_terms,
    user_based_prediction,
    validate_feedback_frame,
)


def _meal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-30",
                "school_name": "목포가람고등학교",
                "school_kind": "고등학교",
                "school_code": "7140001",
                "menu_text": "투움바스파게티 오이피클 요구르트",
                "dishes": ["투움바스파게티", "오이피클", "요구르트"],
                "calories": 750.0,
                "dish_count": 3,
            },
            {
                "date": "2026-07-31",
                "school_name": "목포가람고등학교",
                "school_kind": "고등학교",
                "school_code": "7140001",
                "menu_text": "매운 닭갈비덮밥 배추김치",
                "dishes": ["매운 닭갈비덮밥", "배추김치"],
                "calories": 820.0,
                "dish_count": 2,
            },
            {
                "date": "2026-07-30",
                "school_name": "목포나루고등학교",
                "school_kind": "고등학교",
                "school_code": "7140002",
                "menu_text": "치즈파스타 마늘빵 샐러드",
                "dishes": ["치즈파스타", "마늘빵", "샐러드"],
                "calories": 700.0,
                "dish_count": 3,
            },
            {
                "date": "2026-07-31",
                "school_name": "목포나루중학교",
                "school_kind": "중학교",
                "school_code": "8500002",
                "menu_text": "잡곡밥 된장국 고등어구이",
                "dishes": ["잡곡밥", "된장국", "고등어구이"],
                "calories": 680.0,
                "dish_count": 3,
            },
        ]
    )


def _feedback() -> pd.DataFrame:
    rows = []
    ratings = {
        "학생A": {"m1": 5, "m2": 4},
        "학생B": {"m1": 5, "m2": 3},
        "학생C": {"m1": 1, "m2": 1},
    }
    content = {
        "학생A": {"m1": 90.0, "m2": 80.0},
        "학생B": {"m1": 80.0, "m2": 70.0},
        "학생C": {"m1": 30.0, "m2": 40.0},
    }
    likes = {
        "학생A": "돈까스 파스타",
        "학생B": "파스타 돈까스",
        "학생C": "잡곡밥 생선",
    }
    avoids = {"학생A": "오이", "학생B": "오이", "학생C": "치즈"}
    for participant, menus in ratings.items():
        for menu_id, rating in menus.items():
            rows.append(
                {
                    "participant_code": participant,
                    "grade_band": "중2",
                    "menu_id": menu_id,
                    "meal_date": "2026-07-30" if menu_id == "m1" else "2026-07-31",
                    "source": "목포대 학생식당 검증 식단",
                    "menu_text": "부대찌개 돈까스" if menu_id == "m1" else "투움바스파게티",
                    "likes_text": likes[participant],
                    "avoids_text": avoids[participant],
                    "preferred_types": "면,튀김",
                    "spice_level": 3,
                    "content_method": "tfidf",
                    "content_prediction": content[participant][menu_id],
                    "actual_rating": rating,
                    "feedback_tag": "맛",
                }
            )
    return pd.DataFrame(rows)


def test_content_scores_put_liked_menu_above_avoided_menu() -> None:
    scored, notice = predict_satisfaction(
        _meal_frame(),
        likes_text="파스타, 치즈",
        avoids_text="닭갈비",
        preferred_types=["면"],
        spice_level=2,
        method="tfidf",
    )

    pasta = scored.loc[scored["menu_text"].str.contains("치즈파스타")].iloc[0]
    chicken = scored.loc[scored["menu_text"].str.contains("닭갈비")].iloc[0]
    assert 0 <= scored["content_score"].min() <= scored["content_score"].max() <= 100
    assert pasta["content_score"] > chicken["content_score"]
    assert "TF-IDF" in notice


def test_spice_difference_is_a_penalty_not_a_bonus() -> None:
    frame = _meal_frame().loc[lambda value: value["menu_text"].str.contains("매운")]

    close, _ = predict_satisfaction(
        frame,
        likes_text="닭갈비",
        avoids_text="",
        preferred_types=[],
        spice_level=4,
    )
    far, _ = predict_satisfaction(
        frame,
        likes_text="닭갈비",
        avoids_text="",
        preferred_types=[],
        spice_level=1,
    )

    assert close.iloc[0]["content_score"] > far.iloc[0]["content_score"]
    assert close.iloc[0]["content_score"] - far.iloc[0]["content_score"] == pytest.approx(7.5)
    assert close.iloc[0]["content_score"] == round(
        float(close.iloc[0]["content_similarity"]) * 70.0 - 2.5,
        1,
    )


def test_best_and_worst_menus_do_not_overlap() -> None:
    scored, _ = predict_satisfaction(
        _meal_frame(),
        likes_text="파스타",
        avoids_text="고등어",
        preferred_types=["면"],
        spice_level=2,
    )

    best, worst = best_worst_menus(scored, top_n=2)

    assert len(best) == 2
    assert len(worst) == 2
    assert set(best.index).isdisjoint(worst.index)


def test_school_statistics_and_signature_terms_are_school_specific() -> None:
    stats = school_statistics(_meal_frame())
    signatures = signature_terms(_meal_frame(), "목포나루고등학교", top_n=3)

    garam = stats.loc[stats["학교"] == "목포가람고등학교"].iloc[0]
    assert garam["급식 일수"] == 2
    assert garam["메뉴 항목 수"] == 5
    assert "치즈파스타" in set(signatures["시그니처 메뉴"])


def test_food_mbti_uses_all_four_question_axes() -> None:
    code, explanation = food_mbti(
        rice_vs_noodle=5,
        mild_vs_spicy=5,
        familiar_vs_new=1,
        dessert_vs_hearty=1,
    )

    assert code == "ESFD"
    assert all(axis in explanation for axis in ("면", "매운", "익숙", "디저트"))


def test_meal_buddies_returns_the_closest_anonymous_pair() -> None:
    pairs, matrix = meal_buddies(
        "학생A|파스타 치즈|오이\n학생B|치즈 파스타|오이\n학생C|잡곡밥 생선|치즈"
    )

    assert pairs.iloc[0]["학생 1"] == "학생A"
    assert pairs.iloc[0]["학생 2"] == "학생B"
    assert matrix.shape == (3, 4)


def test_feedback_rejects_duplicate_participant_menu_and_invalid_rating() -> None:
    duplicate = pd.concat([_feedback(), _feedback().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="중복"):
        validate_feedback_frame(duplicate)

    invalid_method = _feedback()
    invalid_method.loc[0, "content_method"] = "알 수 없는 방식"
    with pytest.raises(ValueError, match="분석 방식"):
        validate_feedback_frame(invalid_method)

    invalid = _feedback()
    invalid.loc[0, "actual_rating"] = 6
    with pytest.raises(ValueError, match="1에서 5"):
        validate_feedback_frame(invalid)


def test_user_based_prediction_hides_target_rating_and_weights_neighbors() -> None:
    prediction = user_based_prediction(
        _feedback(), target_participant="학생A", target_menu_id="m2"
    )

    assert prediction.rating == pytest.approx(3.0)
    assert prediction.common_neighbor_count == 2
    assert prediction.fallback == "유사 학생 가중 평균"


def test_user_based_prediction_uses_item_mean_without_common_reviews() -> None:
    prediction = user_based_prediction(
        _feedback(), target_participant="학생D", target_menu_id="m2"
    )

    assert prediction.rating == pytest.approx((4 + 3 + 1) / 3)
    assert prediction.common_neighbor_count == 0
    assert prediction.fallback == "해당 식단 모둠 평균"


def test_recommender_evaluation_compares_three_predictions_to_actual_reviews() -> None:
    detail, summary = evaluate_recommenders(_feedback())

    row = detail.loc[
        (detail["participant_code"] == "학생A") & (detail["menu_id"] == "m2")
    ].iloc[0]
    assert row["user_prediction"] == pytest.approx(60.0)
    assert row["hybrid_prediction"] == pytest.approx(70.0)
    assert row["actual_score"] == 80.0
    assert set(summary["추천 방식"]) == {"콘텐츠 기반", "유저 기반", "혼합"}


def test_pareto_candidates_exclude_a_strictly_dominated_menu() -> None:
    candidates = pareto_candidates(
        _meal_frame(), group_preferences=["파스타 치즈", "면 치즈"]
    )

    assert not candidates.empty
    assert "수업용 대체지표" in candidates.attrs["notice"]
    assert set(candidates["잔반 위험 대체지표"]).issubset(range(0, 101))


def test_feedback_csv_round_trip_preserves_the_anonymous_schema(tmp_path) -> None:
    output = feedback_to_csv(_feedback(), tmp_path / "feedback.csv")
    restored = feedback_from_csv(output)

    assert list(restored.columns) == list(_feedback().columns)
    assert restored.to_dict("records") == _feedback().to_dict("records")


def test_feedback_csv_preserves_empty_optional_preferences(tmp_path) -> None:
    feedback = _feedback()
    feedback["avoids_text"] = ""
    feedback["preferred_types"] = ""

    output = feedback_to_csv(feedback, tmp_path / "empty-preferences.csv")
    restored = feedback_from_csv(output)

    assert set(restored["avoids_text"]) == {""}
    assert set(restored["preferred_types"]) == {""}
    assert "nan" not in " ".join(restored.astype(str).to_numpy().ravel()).casefold()


def test_feedback_csv_rejects_spreadsheet_formula_text(tmp_path) -> None:
    feedback = _feedback()
    feedback.loc[0, "likes_text"] = "=HYPERLINK(\"https://example.com\")"

    with pytest.raises(ValueError, match="수식 기호"):
        feedback_to_csv(feedback, tmp_path / "unsafe.csv")

    assert not (tmp_path / "unsafe.csv").exists()


def test_feedback_analysis_reports_prediction_error_and_small_sample_limit() -> None:
    message, detail, summary, tags = analyze_feedback(_feedback())

    assert "6명 이하의 작은 표본" in message
    assert len(detail) == 6
    assert set(summary["추천 방식"]) == {"콘텐츠 기반", "유저 기반", "혼합"}
    assert tags.loc[tags["피드백 태그"] == "맛", "응답 수"].iloc[0] == 6


def test_feedback_clusters_keep_similar_students_together() -> None:
    clustered = cluster_feedback(_feedback())
    labels = clustered.set_index("participant_code")["food_cluster"]

    assert labels["학생A"] == labels["학생B"]
    assert labels["학생A"] != labels["학생C"]


def test_high_school_recommendation_uses_only_high_schools() -> None:
    ranked, notice = recommend_high_schools(_meal_frame(), "치즈 파스타")

    assert list(ranked["학교"])[0] == "목포나루고등학교"
    assert "급식 취향만" in notice
    assert "목포나루중학교" not in set(ranked["학교"])
