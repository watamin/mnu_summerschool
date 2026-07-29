from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd
import pytest
from matplotlib.figure import Figure

import mokpo_service
from neis_meal_ai.mokpo_data import MokpoDataset, load_mokpo_dataset, load_validation_menus
from neis_meal_ai.mokpo_ui import (
    _message_chatbot_component,
    add_feedback_callback,
    analyze_uploaded_feedback_callback,
    analyze_feedback_callback,
    create_mokpo_app,
    lab_callback,
    meal_chat_callback,
    mnu_prediction_callback,
    matrix_recommendation_callback,
    personal_recommendation_callback,
    register_mnu_predictions_callback,
    sample_foods_callback,
    school_analysis_callback,
    school_value_analysis_callback,
)
from neis_meal_ai.mokpo_analytics import feedback_to_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _data():
    dataset = load_mokpo_dataset(
        PROJECT_ROOT / "data" / "mokpo_schools.json",
        PROJECT_ROOT / "data" / "mokpo_meals_live.json",
    )
    validation, _ = load_validation_menus(
        PROJECT_ROOT / "data" / "mnu_cafeteria_2026_07_30_31.json"
    )
    return dataset, validation


def test_personal_callback_returns_best_and_worst_real_school_menus() -> None:
    dataset, _ = _data()
    school = dataset.meals["school_name"].iloc[0]

    message, best, worst = personal_recommendation_callback(
        dataset,
        school_name=school,
        likes_text="파스타, 돈까스",
        avoids_text="오이",
        preferred_types=["면", "튀김"],
        spice_level=2,
        method="tfidf",
    )

    assert "콘텐츠 기반" in message
    assert not best.empty
    assert not worst.empty
    assert set(best["날짜"]).isdisjoint(worst["날짜"])


def test_matrix_callbacks_sample_real_foods_and_return_map() -> None:
    dataset, _ = _data()
    school = max(
        dataset.meals.groupby("school_name"),
        key=lambda item: len({dish for dishes in item[1]["dishes"] for dish in dishes}),
    )[0]

    next_seed, sampled_names, sample_message, survey = sample_foods_callback(
        dataset, school_name=school, seed=11
    )
    survey.loc[0, "평점"] = 5
    survey.loc[1, "평점"] = 1
    message, best, worst, figure, coordinates, matrix_state = (
        matrix_recommendation_callback(
            dataset,
            school_name=school,
            sampled_foods=sampled_names,
            ratings=survey,
        )
    )

    assert next_seed == 12
    assert len(survey) == 30
    assert len(sampled_names) == 30
    assert survey["음식"].is_unique
    assert "실제 급식 음식 30개" in sample_message
    assert "pinv" in message
    assert not best.empty and not worst.empty
    assert set(best["음식"]).isdisjoint(worst["음식"])
    assert isinstance(figure, Figure)
    assert not coordinates[["X", "Y"]].isna().any().any()
    assert matrix_state["school_name"] == school
    assert len(matrix_state["records"]) >= len(best) + len(worst)


class RecordingNimClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ask(self, question, context, history=()):
        self.calls.append(
            {"question": question, "context": context, "history": list(history)}
        )
        return "TF와 IDF를 곱한 데이터 가치 점수가 가장 높기 때문입니다."


def test_message_chatbot_component_supports_gradio_4_and_6_signatures() -> None:
    legacy_calls: list[dict[str, str]] = []
    current_calls: list[dict[str, str]] = []

    class LegacyChatbot:
        def __init__(self, *, label: str, type: str = "tuples") -> None:
            legacy_calls.append({"label": label, "type": type})

    class CurrentChatbot:
        def __init__(self, *, label: str) -> None:
            current_calls.append({"label": label})

    _message_chatbot_component(
        SimpleNamespace(Chatbot=LegacyChatbot), label="이전 버전"
    )
    _message_chatbot_component(
        SimpleNamespace(Chatbot=CurrentChatbot), label="현재 버전"
    )

    assert legacy_calls == [{"label": "이전 버전", "type": "messages"}]
    assert current_calls == [{"label": "현재 버전"}]


def test_meal_chat_callback_grounds_answer_in_selected_school_values() -> None:
    dataset, _ = _data()
    school = dataset.meals["school_name"].iloc[0]
    client = RecordingNimClient()

    history, cleared = meal_chat_callback(
        dataset,
        school_name=school,
        question="왜 이 음식이 1위인가요?",
        history=[],
        nim_client=client,
        matrix_state=None,
    )

    assert cleared == ""
    assert history[-2] == {"role": "user", "content": "왜 이 음식이 1위인가요?"}
    assert history[-1]["role"] == "assistant"
    assert "데이터 가치 점수" in history[-1]["content"]
    assert len(client.calls) == 1
    context = client.calls[0]["context"]
    assert school in context
    assert all(term in context for term in ("TF", "IDF", "데이터 가치 점수"))


def test_meal_chat_callback_ignores_an_empty_question() -> None:
    dataset, _ = _data()
    client = RecordingNimClient()
    existing = [{"role": "assistant", "content": "질문해 주세요."}]

    history, cleared = meal_chat_callback(
        dataset,
        school_name=dataset.meals["school_name"].iloc[0],
        question="   ",
        history=existing,
        nim_client=client,
        matrix_state=None,
    )

    assert history == existing
    assert cleared == ""
    assert client.calls == []


def test_feedback_callback_adds_anonymous_review_and_rejects_duplicate() -> None:
    _, validation = _data()

    state, table, message = add_feedback_callback(
        validation,
        [],
        participant_code="학생A",
        grade_band="중2",
        menu_id="mnu-2026-07-30-lunch",
        likes_text="돈까스, 부대찌개",
        avoids_text="오이",
        preferred_types=["밥", "튀김"],
        spice_level=3,
        actual_rating=5,
        feedback_tag="맛",
        method="tfidf",
    )

    assert len(state) == 1
    assert len(table) == 1
    assert "저장하지 않고 현재 세션" in message
    assert 0 <= table.iloc[0]["콘텐츠 예측"] <= 100

    same_state, same_table, duplicate_message = add_feedback_callback(
        validation,
        state,
        participant_code="학생A",
        grade_band="중2",
        menu_id="mnu-2026-07-30-lunch",
        likes_text="돈까스",
        avoids_text="오이",
        preferred_types=["튀김"],
        spice_level=3,
        actual_rating=4,
        feedback_tag="맛",
        method="tfidf",
    )

    assert same_state == state
    assert len(same_table) == 1
    assert "중복" in duplicate_message


def test_mnu_prediction_callback_previews_both_dates_before_reviews() -> None:
    _, validation = _data()

    message, predictions = mnu_prediction_callback(
        validation,
        likes_text="돈까스, 파스타",
        avoids_text="오이",
        preferred_types=["면", "튀김"],
        spice_level=3,
        method="tfidf",
    )

    assert "식사 전에" in message
    assert list(predictions["날짜"]) == ["2026-07-30", "2026-07-31"]
    assert predictions["콘텐츠 예측"].between(0, 100).all()
    assert predictions["예측 근거"].str.len().gt(0).all()


def test_registered_prediction_is_kept_when_review_is_added_later() -> None:
    _, validation = _data()

    prediction_state, _, prediction_table = register_mnu_predictions_callback(
        validation,
        [],
        participant_code="학생A",
        grade_band="중2",
        likes_text="돈까스",
        avoids_text="오이",
        preferred_types=["튀김"],
        spice_level=3,
        method="tfidf",
    )
    before_score = float(prediction_table.iloc[0]["콘텐츠 예측"])
    assert prediction_state[0]["content_method"] == "tfidf"

    feedback_state, feedback_table, message = add_feedback_callback(
        validation,
        [],
        prediction_state=prediction_state,
        participant_code="학생A",
        grade_band="중2",
        menu_id="mnu-2026-07-30-lunch",
        likes_text="식사 뒤에 바꾼 취향",
        avoids_text="",
        preferred_types=["면"],
        spice_level=1,
        actual_rating=5,
        feedback_tag="맛",
        method="tfidf",
    )

    assert len(feedback_state) == 1
    assert float(feedback_table.iloc[0]["콘텐츠 예측"]) == before_score
    assert feedback_state[0]["likes_text"] == "돈까스"
    assert "식사 전에 등록한 예측" in message


def test_ui_review_requires_a_registered_pre_meal_prediction() -> None:
    _, validation = _data()

    state, table, message = add_feedback_callback(
        validation,
        [],
        prediction_state=[],
        participant_code="학생A",
        grade_band="중2",
        menu_id="mnu-2026-07-30-lunch",
        likes_text="돈까스",
        avoids_text="오이",
        preferred_types=["튀김"],
        spice_level=3,
        actual_rating=5,
        feedback_tag="맛",
        method="tfidf",
    )

    assert state == []
    assert table.empty
    assert "식사 전 예측" in message


def test_group_analysis_callback_compares_content_user_and_hybrid() -> None:
    _, validation = _data()
    state: list[dict] = []
    for participant, ratings, likes in (
        ("학생A", (5, 4), "돈까스 파스타"),
        ("학생B", (5, 3), "파스타 돈까스"),
        ("학생C", (1, 1), "잡곡밥 생선"),
    ):
        for menu_id, rating in zip(validation["menu_id"], ratings):
            state, _, _ = add_feedback_callback(
                validation,
                state,
                participant_code=participant,
                grade_band="중2",
                menu_id=menu_id,
                likes_text=likes,
                avoids_text="오이",
                preferred_types=["면"],
                spice_level=3,
                actual_rating=rating,
                feedback_tag="맛",
                method="tfidf",
            )

    message, detail, summary, tags, clusters, buddies = analyze_feedback_callback(
        state
    )

    assert "6명 이하의 작은 표본" in message
    assert len(detail) == 6
    assert set(summary["추천 방식"]) == {"콘텐츠 기반", "유저 기반", "혼합"}
    assert tags["응답 수"].sum() == 6
    assert clusters["익명 참여자"].nunique() == 3
    assert not buddies.empty


def test_teacher_can_merge_anonymous_csv_files_from_student_pcs(
    tmp_path: Path,
) -> None:
    _, validation = _data()
    paths = []
    for participant, ratings in (("학생A", (5, 4)), ("학생B", (4, 3))):
        state: list[dict] = []
        for menu_id, rating in zip(validation["menu_id"], ratings):
            state, _, _ = add_feedback_callback(
                validation,
                state,
                participant_code=participant,
                grade_band="중2",
                menu_id=menu_id,
                likes_text="돈까스 파스타",
                avoids_text="오이",
                preferred_types=["면"],
                spice_level=3,
                actual_rating=rating,
                feedback_tag="맛",
                method="tfidf",
            )
        path = feedback_to_csv(pd.DataFrame(state), tmp_path / f"{participant}.csv")
        paths.append(path)

    records, message, detail, summary, *_ = analyze_uploaded_feedback_callback(paths)

    assert len(records) == 4
    assert "익명 참여자 2명" in message
    assert len(detail) == 4
    assert set(summary["추천 방식"]) == {"콘텐츠 기반", "유저 기반", "혼합"}


def test_invalid_uploaded_csv_preserves_current_session_feedback(tmp_path: Path) -> None:
    _, validation = _data()
    state, _, _ = add_feedback_callback(
        validation,
        [],
        participant_code="학생A",
        grade_band="중2",
        menu_id="mnu-2026-07-30-lunch",
        likes_text="돈까스",
        avoids_text="오이",
        preferred_types=["튀김"],
        spice_level=3,
        actual_rating=5,
        feedback_tag="맛",
        method="tfidf",
    )
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("wrong,column\n1,2\n", encoding="utf-8")

    kept, message, *_ = analyze_uploaded_feedback_callback([invalid], state)

    assert kept == state
    assert "필요한 열" in message


def test_meal_lab_reports_embedding_fallback() -> None:
    dataset, _ = _data()

    class FailingEmbedder:
        device = "cuda"

        def encode(self, _texts):
            raise RuntimeError("모델 불러오기 실패")

    message, candidates, pairs = lab_callback(
        dataset,
        school_name=dataset.meals["school_name"].iloc[0],
        group_preferences_text="돈까스 파스타",
        method="embedding",
        embedder=FailingEmbedder(),
    )

    assert "TF-IDF" in message
    assert not candidates.empty
    assert not pairs.empty


def test_school_analysis_callback_returns_stats_signatures_and_ranking() -> None:
    dataset, _ = _data()
    school = dataset.schools.loc[
        dataset.schools["school_kind"] == "고등학교", "school_name"
    ].iloc[0]

    message, stats, signatures, ranking = school_analysis_callback(
        dataset, school_name=school, preference_text="파스타 돈까스", method="tfidf"
    )

    assert "급식 취향만" in message
    assert stats.iloc[0]["학교"] == school
    assert not signatures.empty
    assert not ranking.empty


def test_school_value_analysis_shows_real_formula_inputs_and_rankings() -> None:
    dataset, _ = _data()
    school = dataset.meals["school_name"].iloc[0]

    message, stats, frequencies, values, overall, ranking = (
        school_value_analysis_callback(
            dataset,
            school_name=school,
            preference_text="파스타 돈까스 치즈",
            method="tfidf",
        )
    )

    assert school in message
    assert all(term in message for term in ("TF =", "IDF =", "데이터 가치 점수"))
    assert stats.iloc[0]["학교"] == school
    assert not frequencies.empty
    assert not values.empty
    assert not overall.empty
    assert not ranking.empty
    assert values.iloc[0]["순위"] == 1


def test_mokpo_app_contains_food_value_matrix_map_and_nim_service() -> None:
    dataset, validation = _data()
    demo = create_mokpo_app(dataset, validation, nim_client=RecordingNimClient())
    config = json.dumps(demo.get_config_file(), ensure_ascii=False, default=str)

    assert isinstance(demo, gr.Blocks)
    for text in (
        "목포 급식 AI 탐험실",
        "학생 설문·개인 결과",
        "30개 음식 역행렬 추천",
        "모둠 피드백 분석",
        "학교별 가치 음식",
        "AI 식단 실험실",
        "NVIDIA NIM 데이터 해설",
        "31개교",
        "674",
        "실제 식단일 20260624~20260729",
        "음식 30개 뽑기",
        "역행렬 추천 계산",
        "TF-IDF 데이터 가치 점수",
        "음식 유사도 2차원 지도",
        "대화 지우기",
        "6명 이하의 작은 표본",
        "콘텐츠 기반 추천",
        "유저 기반 추천",
        "수업용 대체지표",
    ):
        assert text in config
    assert "학생은 코드를 작성하지 않고 익명 설문" not in config
    assert "실제 이름·학번·반·연락처·질병명" not in config


def test_app_excludes_catalog_schools_without_meal_rows() -> None:
    dataset, validation = _data()
    extra_school = pd.DataFrame(
        [
            {
                "school_name": "급식없는고등학교",
                "school_kind": "고등학교",
                "office_code": "Q10",
                "school_code": "7999999",
                "address": "전라남도 목포시 예시로 1",
            }
        ]
    )
    limited = MokpoDataset(
        schools=pd.concat([dataset.schools, extra_school], ignore_index=True),
        meals=dataset.meals,
        metadata={
            **dataset.metadata,
            "school_count": dataset.metadata["school_count"] + 1,
            "skipped_schools": ["급식없는고등학교"],
        },
    )

    config = json.dumps(
        create_mokpo_app(limited, validation).get_config_file(),
        ensure_ascii=False,
        default=str,
    )

    assert "급식없는고등학교" not in config
    assert "수집 제외 1개교" in config


def test_mokpo_service_is_local_only_and_does_not_launch_on_import() -> None:
    options = mokpo_service.local_launch_options()

    assert options["server_name"] == "127.0.0.1"
    assert options["share"] is False
    assert options["inbrowser"] is True


def test_mokpo_service_reuses_one_lazy_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_app(_dataset, _validation, *, embedder, nim_client):
        captured["embedder"] = embedder
        captured["nim_client"] = nim_client
        return "app"

    monkeypatch.setattr(mokpo_service, "create_mokpo_app", fake_create_app)

    assert mokpo_service.create_service_app() == "app"
    assert captured["embedder"].__class__.__name__ == "SentenceTransformerEmbedder"
    assert captured["nim_client"].__class__.__name__ == "NvidiaNimClient"
