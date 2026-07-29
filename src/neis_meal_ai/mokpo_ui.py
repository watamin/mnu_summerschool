"""목포 급식 AI 탐험실의 순수 콜백과 Gradio 화면."""

from __future__ import annotations

import inspect
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .mokpo_analytics import (
    FEEDBACK_COLUMNS,
    GRADE_BANDS,
    analyze_feedback,
    best_worst_menus,
    cluster_feedback,
    feedback_from_csv,
    feedback_to_csv,
    food_mbti,
    food_map_coordinates,
    global_food_value_explanation,
    global_food_values,
    inverse_matrix_recommendations,
    meal_buddies,
    menu_pair_scores,
    pareto_candidates,
    predict_satisfaction,
    recommend_high_schools,
    school_food_frequencies,
    school_food_value_explanation,
    school_food_values,
    school_statistics,
    sample_school_foods,
    signature_terms,
    validate_feedback_frame,
)
from .mokpo_data import MokpoDataset
from .nim_chat import NimChatError, NvidiaNimClient
from .recommender import MENU_TYPE_KEYWORDS
from .student_profile_ui import (
    export_ratings_callback,
    load_profile_callback,
    matrix_dashboard_callback,
    save_profile_callback,
)
from .student_profiles import StudentProfileStore
from .text_vectors import TextEmbedder


PREDICTION_COLUMNS = FEEDBACK_COLUMNS[:-2]


def _message_chatbot_component(gradio_module: object, *, label: str):
    """Gradio 4의 type 인자와 Gradio 6의 메시지 기본값을 함께 지원한다."""

    chatbot = getattr(gradio_module, "Chatbot")
    kwargs: dict[str, str] = {"label": str(label)}
    if "type" in inspect.signature(chatbot).parameters:
        kwargs["type"] = "messages"
    return chatbot(**kwargs)


def _student_rating_dataframe_component(gradio_module: object):
    """Gradio 4에서는 없는 고정 열 옵션을 지원 버전에서만 사용한다."""

    dataframe = getattr(gradio_module, "Dataframe")
    kwargs: dict[str, object] = {
        "headers": ["순서", "음식", "구분", "평점"],
        "datatype": ["number", "str", "str", "number"],
        "value": pd.DataFrame(columns=["순서", "음식", "구분", "평점"]),
        "label": "내 음식 30개 평가표",
        "interactive": True,
    }
    parameters = inspect.signature(dataframe).parameters
    if "static_columns" in parameters:
        kwargs["static_columns"] = [0, 1, 2]
    if "pinned_columns" in parameters:
        kwargs["pinned_columns"] = 3
    return dataframe(**kwargs)


def _recommendation_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        ["date", "school_name", "content_score", "menu_text", "prediction_reason"]
    ].rename(
        columns={
            "date": "날짜",
            "school_name": "학교",
            "content_score": "콘텐츠 예측",
            "menu_text": "메뉴",
            "prediction_reason": "예측 근거",
        }
    )


def food_map_figure(coordinates: pd.DataFrame):
    """한글 글꼴 설치와 무관한 번호형 음식 유사도 산점도를 그린다."""

    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize

    if coordinates.empty:
        raise ValueError("그릴 음식 지도 좌표가 없습니다.")
    required = {"번호", "음식", "X", "Y", "구분", "평점", "등장 횟수"}
    if not required.issubset(coordinates.columns):
        raise ValueError("음식 지도 좌표 표의 열이 올바르지 않습니다.")
    predicted = coordinates.loc[coordinates["구분"] == "역행렬 추천"]
    top_predicted = set(
        predicted.nlargest(min(5, len(predicted)), "평점")["음식"].astype(str)
    )
    norm = Normalize(vmin=1.0, vmax=5.0)
    figure, axis = plt.subplots(figsize=(10, 7))
    for row in coordinates.itertuples(index=False):
        directly_rated = row.구분 == "직접 평가"
        top_choice = str(row.음식) in top_predicted
        marker = "*" if top_choice else "o"
        size = 90 + min(int(row._6), 10) * 18 if hasattr(row, "_6") else 108
        point = axis.scatter(
            float(row.X),
            float(row.Y),
            c=[float(row.평점)],
            cmap="viridis",
            norm=norm,
            s=size,
            marker=marker,
            edgecolors="black" if directly_rated else "none",
            linewidths=1.4 if directly_rated else 0.0,
            alpha=0.88,
        )
        axis.annotate(
            str(int(row.번호)),
            (float(row.X), float(row.Y)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0, color="#d0d0d0", linewidth=0.8)
    axis.axvline(0, color="#d0d0d0", linewidth=0.8)
    axis.set_xlabel("TF-IDF PCA axis 1")
    axis.set_ylabel("TF-IDF PCA axis 2")
    axis.set_title("Food similarity map")
    figure.colorbar(point, ax=axis, label="Rating (1-5)")
    figure.tight_layout()
    return figure


def sample_foods_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    seed: int,
) -> tuple[int, list[str], str, pd.DataFrame]:
    """학교 실제 음식 30개와 다음 추첨 seed를 준비한다."""

    current_seed = int(seed or 0)
    survey = sample_school_foods(
        dataset.meals,
        str(school_name),
        sample_size=30,
        seed=current_seed,
    )
    foods = survey["음식"].astype(str).tolist()
    if len(survey) == 30:
        message = (
            f"### {school_name} 음식 취향 설문\n"
            "실제 급식 음식 30개를 뽑았습니다. 평점 열을 1~5점으로 바꿔 주세요."
        )
    else:
        message = (
            f"### {school_name} 음식 취향 설문\n"
            f"고유 음식이 30개보다 적어 실제 급식 음식 {len(survey)}개를 모두 표시했습니다."
        )
    return current_seed + 1, foods, message, survey


def matrix_recommendation_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    sampled_foods: Sequence[str],
    ratings: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame, object, pd.DataFrame, dict[str, object]]:
    """편집한 음식 평점을 검증하고 행렬 추천과 2차원 지도를 함께 만든다."""

    survey = ratings.copy() if isinstance(ratings, pd.DataFrame) else pd.DataFrame(
        ratings, columns=["음식", "평점"]
    )
    if not {"음식", "평점"}.issubset(survey.columns):
        raise ValueError("음식과 평점 두 열이 있는 설문 표가 필요합니다.")
    current_foods = survey["음식"].astype(str).str.strip().tolist()
    expected_foods = [str(value).strip() for value in (sampled_foods or [])]
    if current_foods != expected_foods:
        raise ValueError("음식 이름이나 순서를 바꾸지 말고 평점 열만 수정하세요.")
    recommendations = inverse_matrix_recommendations(
        dataset.meals,
        str(school_name),
        survey,
        regularization=0.1,
    )
    if recommendations.empty:
        raise ValueError("모든 실제 음식을 평가해 새로 예측할 음식이 없습니다.")
    result_count = min(10, max(1, len(recommendations) // 2))
    best = recommendations.head(result_count).reset_index(drop=True)
    worst = recommendations.sort_values(
        ["예상 평점", "등장 횟수", "음식"], ascending=[True, False, True]
    ).head(result_count).reset_index(drop=True)
    coordinates = food_map_coordinates(
        dataset.meals,
        str(school_name),
        survey,
        recommendations,
    )
    figure = food_map_figure(coordinates)
    gram_rows, gram_columns = recommendations.attrs["gram_shape"]
    message = (
        f"### {school_name} 역행렬 추천 결과\n"
        f"평가 {recommendations.attrs['rated_count']}개 × TF-IDF 특징 "
        f"{recommendations.attrs['feature_count']}개를 사용했습니다.\n\n"
        f"`G = X Xᵀ + 0.1I`는 {gram_rows}×{gram_columns} 행렬이고, "
        "`w = Xᵀ pinv(G)(y - 3)`로 취향 가중치를 계산했습니다.\n\n"
        "예상 평점은 `clip(3 + X_all w, 1, 5)`로 1~5점 사이에 두었습니다."
    )
    return (
        message,
        best,
        worst,
        figure,
        coordinates,
        {
            "school_name": str(school_name),
            "records": recommendations.to_dict(orient="records"),
        },
    )


def _school_chat_context(
    dataset: MokpoDataset,
    school_name: str,
    matrix_state: Mapping[str, object] | None,
) -> str:
    stats = school_statistics(dataset.meals)
    selected_stats = stats.loc[stats["학교"] == str(school_name)]
    if selected_stats.empty:
        raise ValueError("선택한 학교의 급식 데이터가 없습니다.")
    frequencies = school_food_frequencies(
        dataset.meals, str(school_name), top_n=10
    )
    values = school_food_values(dataset.meals, str(school_name), top_n=10)
    global_values = global_food_values(dataset.meals, top_n=10)
    parts = [
        (
            f"데이터 조회 기간={dataset.metadata['query_start']}~"
            f"{dataset.metadata['query_end']}, 실제 식단일="
            f"{dataset.metadata['actual_start']}~{dataset.metadata['actual_end']}, "
            f"전체 실제 중식 행={dataset.metadata['meal_row_count']}"
        ),
        "[선택 학교 통계]\n" + selected_stats.to_string(index=False),
        "[자주 나온 핵심 음식]\n" + frequencies.to_string(index=False),
        "[TF-IDF 데이터 가치 상위 음식]\n" + values.to_string(index=False),
        "[1위 계산식]\n" + school_food_value_explanation(values),
        "[학교와 상관없는 전체 음식 랭킹]\n"
        + global_values.to_string(index=False),
        "[전체 음식 1위 계산식]\n"
        + global_food_value_explanation(global_values),
    ]
    if matrix_state and str(matrix_state.get("school_name", "")) == str(school_name):
        records = matrix_state.get("records", [])
        matrix = pd.DataFrame.from_records(records if isinstance(records, list) else [])
        if {"음식", "예상 평점"}.issubset(matrix.columns) and not matrix.empty:
            best = matrix.sort_values(
                ["예상 평점", "음식"], ascending=[False, True]
            ).head(5)
            worst = matrix.sort_values(
                ["예상 평점", "음식"], ascending=[True, True]
            ).head(5)
            parts.extend(
                [
                    "[역행렬 예상 Best]\n" + best.to_string(index=False),
                    "[역행렬 예상 Worst]\n" + worst.to_string(index=False),
                ]
            )
    return "\n\n".join(parts)


def meal_chat_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    question: str,
    history: Sequence[Mapping[str, str]] | None,
    nim_client: NvidiaNimClient | None,
    matrix_state: Mapping[str, object] | None,
) -> tuple[list[dict[str, str]], str]:
    """선택 학교의 계산 근거로 NIM 답변을 만들고 대화 목록에 붙인다."""

    current_history = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in (history or [])
        if str(item.get("role", "")) in {"user", "assistant"}
    ]
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        return current_history, ""
    if nim_client is None:
        answer = "NVIDIA NIM 연결이 준비되지 않았습니다. API 키 파일을 확인해 주세요."
    else:
        try:
            context = _school_chat_context(dataset, str(school_name), matrix_state)
            answer = nim_client.ask(cleaned_question, context, current_history)
        except (NimChatError, ValueError) as exc:
            answer = str(exc)
    return [
        *current_history,
        {"role": "user", "content": cleaned_question},
        {"role": "assistant", "content": answer},
    ], ""


def personal_recommendation_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    likes_text: str,
    avoids_text: str,
    preferred_types: Sequence[str],
    spice_level: int,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """선택 학교의 실제 급식에서 콘텐츠 기반 Best·Worst를 만든다."""

    school_frame = dataset.meals.loc[
        dataset.meals["school_name"] == str(school_name)
    ].copy()
    scored, model_notice = predict_satisfaction(
        school_frame,
        likes_text=likes_text,
        avoids_text=avoids_text,
        preferred_types=preferred_types or [],
        spice_level=int(spice_level),
        method=method,
        embedder=embedder,
    )
    result_count = min(5, max(1, len(scored) // 2))
    best, worst = best_worst_menus(scored, top_n=result_count)
    message = (
        f"### 콘텐츠 기반 추천 결과\n{school_name}의 실제 중식 {len(scored)}일을 "
        f"비교했습니다. {model_notice}\n\n"
        "이 점수는 설문 전 예측이며 실제 만족도와 다를 수 있습니다."
    )
    return message, _recommendation_table(best), _recommendation_table(worst)


def _score_mnu_menus(
    validation_menus: pd.DataFrame,
    *,
    likes_text: str,
    avoids_text: str,
    preferred_types: Sequence[str],
    spice_level: int,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[pd.DataFrame, str]:

    prediction_frame = validation_menus.rename(columns={"meal_date": "date"}).copy()
    prediction_frame["dish_count"] = prediction_frame["dishes"].map(len)
    prediction_frame["calories"] = float("nan")
    scored, model_notice = predict_satisfaction(
        prediction_frame,
        likes_text=likes_text,
        avoids_text=avoids_text,
        preferred_types=preferred_types or [],
        spice_level=int(spice_level),
        method=method,
        embedder=embedder,
    )
    return scored, model_notice


def _mnu_prediction_table(scored: pd.DataFrame) -> pd.DataFrame:
    table = scored[
        ["date", "menu_text", "content_score", "prediction_reason"]
    ].rename(
        columns={
            "date": "날짜",
            "menu_text": "메뉴",
            "content_score": "콘텐츠 예측",
            "prediction_reason": "예측 근거",
        }
    )
    return table.reset_index(drop=True)


def mnu_prediction_callback(
    validation_menus: pd.DataFrame,
    *,
    likes_text: str,
    avoids_text: str,
    preferred_types: Sequence[str],
    spice_level: int,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[str, pd.DataFrame]:
    """7월 30·31일 목포대 식단의 식사 전 콘텐츠 예측표를 만든다."""

    scored, model_notice = _score_mnu_menus(
        validation_menus,
        likes_text=likes_text,
        avoids_text=avoids_text,
        preferred_types=preferred_types,
        spice_level=spice_level,
        method=method,
        embedder=embedder,
    )
    message = (
        "### 목포대 식사 전 예측표\n"
        "식사 전에 두 날짜의 예측값을 먼저 확인하고 기록하세요. "
        f"{model_notice}"
    )
    return message, _mnu_prediction_table(scored)


def register_mnu_predictions_callback(
    validation_menus: pd.DataFrame,
    state: list[dict] | None,
    *,
    participant_code: str,
    grade_band: str,
    likes_text: str,
    avoids_text: str,
    preferred_types: Sequence[str],
    spice_level: int,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[list[dict], str, pd.DataFrame]:
    """익명 학생의 이틀치 예측을 실제 리뷰 전에 세션에 고정한다."""

    records = list(state or [])
    current = pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)
    code = str(participant_code).strip()
    try:
        if re.fullmatch(r"[가-힣A-Za-z0-9_-]{1,12}", code) is None:
            raise ValueError("익명 참여자 코드는 1~12자의 글자·숫자로 입력하세요.")
        if grade_band not in GRADE_BANDS:
            raise ValueError("학년 구간이 올바르지 않습니다.")
        existing = current.loc[current["participant_code"] == code]
        if not existing.empty:
            table = existing[
                ["meal_date", "menu_text", "content_prediction"]
            ].rename(
                columns={
                    "meal_date": "날짜",
                    "menu_text": "메뉴",
                    "content_prediction": "콘텐츠 예측",
                }
            )
            table["예측 근거"] = "처음 등록한 값을 유지함"
            return (
                records,
                "### 이미 등록된 예측\n식사 전 예측은 공정한 비교를 위해 바꾸지 않습니다.",
                table.reset_index(drop=True),
            )
        if current["participant_code"].nunique() >= 6:
            raise ValueError("한 모둠은 익명 참여자 6명까지 등록할 수 있습니다.")
        scored, model_notice = _score_mnu_menus(
            validation_menus,
            likes_text=likes_text,
            avoids_text=avoids_text,
            preferred_types=preferred_types,
            spice_level=spice_level,
            method=method,
            embedder=embedder,
        )
    except ValueError as exc:
        return records, f"### 입력 확인\n{exc}", pd.DataFrame()

    new_records = []
    for row in scored.itertuples():
        new_records.append(
            {
                "participant_code": code,
                "grade_band": str(grade_band),
                "menu_id": str(row.menu_id),
                "meal_date": str(row.date),
                "source": str(row.source),
                "menu_text": str(row.menu_text),
                "likes_text": str(likes_text).strip(),
                "avoids_text": str(avoids_text).strip(),
                "preferred_types": ",".join(preferred_types or []),
                "spice_level": int(spice_level),
                "content_method": str(scored.attrs.get("vector_backend", method)),
                "content_prediction": float(row.content_score),
            }
        )
    combined = [*records, *new_records]
    message = (
        f"### 식사 전 예측 등록 완료\n{code}의 7월 30·31일 예측을 고정했습니다. "
        f"식사 후에도 이 값은 바뀌지 않습니다. {model_notice}"
    )
    return combined, message, _mnu_prediction_table(scored)


def _feedback_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "익명 참여자",
                "날짜",
                "메뉴",
                "콘텐츠 방식",
                "콘텐츠 예측",
                "실제 만족도",
                "태그",
            ]
        )
    return frame[
        [
            "participant_code",
            "meal_date",
            "menu_text",
            "content_method",
            "content_prediction",
            "actual_rating",
            "feedback_tag",
        ]
    ].rename(
        columns={
            "participant_code": "익명 참여자",
            "meal_date": "날짜",
            "menu_text": "메뉴",
            "content_method": "콘텐츠 방식",
            "content_prediction": "콘텐츠 예측",
            "actual_rating": "실제 만족도",
            "feedback_tag": "태그",
        }
    )


def add_feedback_callback(
    validation_menus: pd.DataFrame,
    state: list[dict] | None,
    *,
    participant_code: str,
    grade_band: str,
    menu_id: str,
    likes_text: str,
    avoids_text: str,
    preferred_types: Sequence[str],
    spice_level: int,
    actual_rating: int,
    feedback_tag: str,
    method: str,
    embedder: TextEmbedder | None = None,
    prediction_state: list[dict] | None = None,
) -> tuple[list[dict], pd.DataFrame, str]:
    """목포대 검증 식단 예측과 실제 리뷰를 현재 세션에만 추가한다."""

    current_records = list(state or [])
    current = pd.DataFrame.from_records(current_records, columns=FEEDBACK_COLUMNS)
    try:
        selected = validation_menus.loc[validation_menus["menu_id"] == menu_id]
        if len(selected) != 1:
            raise ValueError("목포대 검증 식단을 하나 선택하세요.")
        menu = selected.iloc[0]
        if prediction_state is not None:
            predictions = pd.DataFrame.from_records(
                prediction_state, columns=PREDICTION_COLUMNS
            )
            registered = predictions.loc[
                (predictions["participant_code"] == str(participant_code).strip())
                & (predictions["menu_id"] == str(menu_id))
            ]
            if len(registered) != 1:
                raise ValueError("먼저 이 익명 코드로 식사 전 예측을 등록하세요.")
            new_record = {
                **registered.iloc[0].to_dict(),
                "actual_rating": int(actual_rating),
                "feedback_tag": str(feedback_tag),
            }
        else:
            prediction_frame = pd.DataFrame(
                [
                    {
                        "date": menu["meal_date"],
                        "school_name": menu["school_name"],
                        "menu_text": menu["menu_text"],
                        "dishes": menu["dishes"],
                        "dish_count": len(menu["dishes"]),
                        "calories": float("nan"),
                    }
                ]
            )
            scored, _ = predict_satisfaction(
                prediction_frame,
                likes_text=likes_text,
                avoids_text=avoids_text,
                preferred_types=preferred_types or [],
                spice_level=int(spice_level),
                method=method,
                embedder=embedder,
            )
            new_record = {
                "participant_code": str(participant_code).strip(),
                "grade_band": str(grade_band),
                "menu_id": str(menu["menu_id"]),
                "meal_date": str(menu["meal_date"]),
                "source": str(menu["source"]),
                "menu_text": str(menu["menu_text"]),
                "likes_text": str(likes_text).strip(),
                "avoids_text": str(avoids_text).strip(),
                "preferred_types": ",".join(preferred_types or []),
                "spice_level": int(spice_level),
                "content_method": str(scored.attrs.get("vector_backend", method)),
                "content_prediction": float(scored.iloc[0]["content_score"]),
                "actual_rating": int(actual_rating),
                "feedback_tag": str(feedback_tag),
            }
        combined = pd.concat([current, pd.DataFrame([new_record])], ignore_index=True)
        validated = validate_feedback_frame(combined)
    except ValueError as exc:
        return current_records, _feedback_display(current), f"### 입력 확인\n{exc}"
    records = validated.to_dict("records")
    comparison_notice = (
        "식사 전에 등록한 예측과 비교합니다. "
        if prediction_state is not None
        else "현재 입력으로 계산한 콘텐츠 예측과 비교합니다. "
    )
    message = (
        f"### 리뷰 추가 완료\n{new_record['participant_code']}의 "
        f"{new_record['meal_date']} 응답을 추가했습니다. 실제 이름은 저장하지 않고 현재 세션에만 "
        f"보관합니다. {comparison_notice}"
    )
    return records, _feedback_display(validated), message


def analyze_feedback_callback(
    state: list[dict] | None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """현재 세션 리뷰로 세 추천 방식, 군집, 밥친구를 분석한다."""

    frame = pd.DataFrame.from_records(state or [], columns=FEEDBACK_COLUMNS)
    try:
        message, detail, summary, tags = analyze_feedback(frame)
        clusters = cluster_feedback(frame).rename(
            columns={
                "participant_code": "익명 참여자",
                "likes_text": "선호 메뉴",
                "avoids_text": "기피 메뉴",
                "grade_band": "학년",
                "food_cluster": "식성 군집",
            }
        )
        unique = (
            frame.sort_values(["participant_code", "meal_date"])
            .drop_duplicates("participant_code")
            .loc[:, ["participant_code", "likes_text", "avoids_text"]]
        )
        profile_text = "\n".join(
            f"{row.participant_code}|{row.likes_text}|{row.avoids_text}"
            for row in unique.itertuples()
        )
        buddies, _ = meal_buddies(profile_text)
    except ValueError as exc:
        empty = pd.DataFrame()
        return f"### 분석 안내\n{exc}", empty, empty, empty, empty, empty
    display_detail = detail[
        [
            "participant_code",
            "meal_date",
            "content_method",
            "content_prediction",
            "user_prediction",
            "hybrid_prediction",
            "actual_score",
            "user_fallback",
        ]
    ].rename(
        columns={
            "participant_code": "익명 참여자",
            "meal_date": "날짜",
            "content_method": "콘텐츠 방식",
            "content_prediction": "콘텐츠 예측",
            "user_prediction": "유저 기반 예측",
            "hybrid_prediction": "혼합 예측",
            "actual_score": "실제 리뷰 환산",
            "user_fallback": "유저 기반 근거",
        }
    )
    return message, display_detail, summary, tags, clusters, buddies


def analyze_uploaded_feedback_callback(
    file_paths: str | Path | Sequence[str | Path] | None,
    current_state: list[dict] | None = None,
) -> tuple[list[dict], str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_records = list(current_state or [])
    if not file_paths:
        empty = pd.DataFrame()
        return current_records, "### 업로드 안내\nCSV 파일을 하나 이상 선택하세요.", empty, empty, empty, empty, empty
    selected_paths = (
        list(file_paths)
        if isinstance(file_paths, Sequence) and not isinstance(file_paths, (str, Path))
        else [file_paths]
    )
    try:
        frames = [feedback_from_csv(path) for path in selected_paths]
        frame = validate_feedback_frame(pd.concat(frames, ignore_index=True))
    except ValueError as exc:
        empty = pd.DataFrame()
        return current_records, f"### 업로드 안내\n{exc}", empty, empty, empty, empty, empty
    records = frame.to_dict("records")
    return records, *analyze_feedback_callback(records)


def export_feedback_callback(
    state: list[dict] | None, output_directory: str | Path | None = None
) -> str | None:
    frame = pd.DataFrame.from_records(state or [], columns=FEEDBACK_COLUMNS)
    try:
        directory = Path(output_directory) if output_directory else Path(
            tempfile.mkdtemp(prefix="mokpo-meal-feedback-")
        )
        output = feedback_to_csv(frame, directory / "익명_급식_피드백.csv")
    except ValueError:
        return None
    return str(output)


def school_analysis_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    preference_text: str,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = school_statistics(dataset.meals)
    selected_stats = stats.loc[stats["학교"] == school_name].reset_index(drop=True)
    signatures = signature_terms(dataset.meals, school_name, top_n=10)
    ranking, notice = recommend_high_schools(
        dataset.meals,
        preference_text,
        method=method,
        embedder=embedder,
    )
    return f"### 학교 급식 비교\n{notice}", selected_stats, signatures, ranking


def school_value_analysis_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    preference_text: str,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """선택 학교의 빈도·TF-IDF 가치와 고교 추천 순위를 함께 계산한다."""

    stats = school_statistics(dataset.meals)
    selected_stats = stats.loc[stats["학교"] == str(school_name)].reset_index(
        drop=True
    )
    frequencies = school_food_frequencies(
        dataset.meals, str(school_name), top_n=15
    )
    values = school_food_values(dataset.meals, str(school_name), top_n=20)
    overall = global_food_values(dataset.meals, top_n=50)
    ranking, notice = recommend_high_schools(
        dataset.meals,
        str(preference_text or ""),
        method=str(method),
        embedder=embedder,
    )
    message = (
        f"{school_food_value_explanation(values)}\n\n"
        "값이 크다는 것은 영양가나 맛이 절대적으로 더 좋다는 뜻이 아니라, "
        "이 수집 자료에서 해당 학교를 설명하는 데 유용하다는 뜻입니다.\n\n"
        f"**학교 취향 비교:** {notice}"
    )
    return message, selected_stats, frequencies, values, overall, ranking


def food_mbti_callback(
    rice_vs_noodle: int,
    mild_vs_spicy: int,
    familiar_vs_new: int,
    dessert_vs_hearty: int,
) -> str:
    code, explanation = food_mbti(
        rice_vs_noodle=rice_vs_noodle,
        mild_vs_spicy=mild_vs_spicy,
        familiar_vs_new=familiar_vs_new,
        dessert_vs_hearty=dessert_vs_hearty,
    )
    return f"### Food MBTI: {code}\n{explanation}\n\n성격검사가 아닌 수업용 식성 분류입니다."


def lab_callback(
    dataset: MokpoDataset,
    *,
    school_name: str,
    group_preferences_text: str,
    method: str,
    embedder: TextEmbedder | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    school_frame = dataset.meals.loc[
        dataset.meals["school_name"] == school_name
    ].copy()
    preferences = [
        line.strip() for line in str(group_preferences_text or "").splitlines() if line.strip()
    ]
    candidates = pareto_candidates(
        school_frame,
        group_preferences=preferences,
        method=method,
        embedder=embedder,
    )
    pairs = menu_pair_scores(
        school_frame,
        preference_text=" ".join(preferences),
        method=method,
        embedder=embedder,
    ).head(10)
    message = (
        "### AI 식단 실험실\n"
        f"{candidates.attrs['vector_notice']}\n\n"
        f"{candidates.attrs['notice']}\n\n{pairs.attrs['notice']}"
    )
    candidate_columns = [
        "date",
        "menu_text",
        "모둠 예측 만족도",
        "가성비 대체지표",
        "잔반 위험 대체지표",
        "영양 편차 대체지표",
    ]
    return message, candidates[candidate_columns], pairs


def create_mokpo_app(
    dataset: MokpoDataset,
    validation_menus: pd.DataFrame,
    *,
    embedder: TextEmbedder | None = None,
    nim_client: NvidiaNimClient | None = None,
    profile_store: StudentProfileStore | None = None,
    classroom_mode: bool = False,
):
    """학교 음식 가치·추천·피드백 분석을 묶은 완성형 Gradio 앱을 만든다."""

    import gradio as gr

    meal_schools = dataset.meals[
        ["school_name", "school_kind"]
    ].drop_duplicates("school_name")
    schools = sorted(meal_schools["school_name"].astype(str).tolist())
    high_schools = sorted(
        meal_schools.loc[
            meal_schools["school_kind"] == "고등학교", "school_name"
        ].astype(str)
    )
    validation_choices = [
        (f"목포대 {row.meal_date} 점심", row.menu_id)
        for row in validation_menus.itertuples()
    ]
    source_text = (
        f"목포시 중·고교 목록 {dataset.metadata['school_count']}개교 · "
        f"급식 분석 {dataset.metadata['meal_school_count']}개교 · "
        f"수집 제외 {len(dataset.metadata['skipped_schools'])}개교 · "
        f"실제 중식 {dataset.metadata['meal_row_count']}행 · "
        f"조회 {dataset.metadata['query_start']}~{dataset.metadata['query_end']} · "
        f"실제 식단일 {dataset.metadata['actual_start']}~{dataset.metadata['actual_end']}"
    )
    global_ranking = global_food_values(dataset.meals, top_n=50)
    global_ranking_explanation = global_food_value_explanation(global_ranking)

    with gr.Blocks(title="목포 급식 AI 탐험실") as demo:
        feedback_state = gr.State([])
        prediction_state = gr.State([])
        survey_seed_state = gr.State(0)
        survey_foods_state = gr.State([])
        matrix_state = gr.State(None)
        gr.Markdown(
            "# 🍱 목포 급식 AI 탐험실\n"
            f"**사용 데이터:** {source_text}\n\n"
            "실제 학교 식단에서 자주 나온 음식과 학교를 잘 구별하는 음식, 개인 취향에 "
            "가까운 음식을 서로 다른 AI 방법으로 비교합니다.\n\n"
            "콘텐츠 기반 추천은 취향과 메뉴 글을 비교하고, 유저 기반 추천은 다른 익명 "
            "학생의 실제 리뷰를 비교합니다. 6명 이하의 작은 표본이므로 결과를 전체 학생에게 "
            "일반화하지 않습니다."
        )

        if profile_store is not None and classroom_mode:
            with gr.Tab("내 프로필·30개 평가"):
                gr.Markdown(
                    "## 로그인한 이름으로 평가 저장하기\n"
                    "목포 지역 학교 급식에서 뽑은 음식 30개를 1점(매우 싫음)부터 "
                    "5점(매우 좋음)까지 평가합니다. 저장 버튼을 여러 번 눌러도 같은 "
                    "프로필에 이어서 기록됩니다. 0은 아직 답하지 않은 문항입니다."
                )
                profile_message = gr.Markdown(
                    "평가표 불러오기를 누르면 로그인한 이름의 30개 질문이 열립니다."
                )
                with gr.Row():
                    profile_load_button = gr.Button("내 평가표 불러오기")
                    profile_save_button = gr.Button("평점 저장", variant="primary")
                profile_table = _student_rating_dataframe_component(gr)

        if profile_store is not None:
            with gr.Tab("학생 행렬분해 실험"):
                gr.Markdown(
                    "## 학생끼리 겹쳐 평가한 정보로 빈칸 예측하기\n"
                    "각 학생은 45개 음식 중 30개만 평가합니다. 검은 점은 학생이 직접 "
                    "평가한 셀, 흰 테두리는 행렬분해가 채운 셀입니다.\n\n"
                    "$$\\hat r_{ui}=\\mu+b_u+b_i+P_u\\cdot Q_i$$\n\n"
                    "**행렬분해 MAE**는 이미 아는 실제 평점 일부를 잠시 가리고 예측한 "
                    "절댓값 오차의 평균입니다. 진짜 빈칸에는 아직 정답이 없으므로 "
                    "정확도 계산에 쓰지 않습니다."
                )
                with gr.Row():
                    matrix_refresh_button = gr.Button(
                        "현재 저장값으로 행렬분해", variant="primary"
                    )
                    if not classroom_mode:
                        ratings_export_button = gr.Button("전체 실제 평점 CSV")
                        ratings_export_file = gr.File(
                            label="학생 평점 CSV", interactive=False
                        )
                class_matrix_message = gr.Markdown()
                profile_status_table = gr.Dataframe(
                    label="학생별 저장 현황", interactive=False
                )
                matrix_metrics_table = gr.Dataframe(
                    label="홀드아웃 검증 결과", interactive=False
                )
                with gr.Tab("실제 평점 행렬"):
                    observed_matrix_table = gr.Dataframe(
                        label="빈칸을 그대로 둔 관측 행렬", interactive=False
                    )
                with gr.Tab("빈칸 완성 행렬"):
                    completed_matrix_table = gr.Dataframe(
                        label="실제·예측 구분 완성 행렬", interactive=False
                    )
                matrix_heatmap = gr.Plot(label="실제와 예측 평점 열지도")
                matrix_recommendations = gr.Dataframe(
                    label="학생별 미평가 음식 예상 Best·Worst", interactive=False
                )
                with gr.Row():
                    student_map = gr.Plot(label="학생 식성 잠재벡터 지도")
                    student_coordinates = gr.Dataframe(
                        label="학생 지도 번호와 좌표", interactive=False
                    )

        with gr.Tab("학생 설문·개인 결과"):
            gr.Markdown("## 1. 실제 학교 급식 Best·Worst 예측")
            with gr.Row():
                school = gr.Dropdown(schools, value=schools[0], label="학교")
                model_method = gr.Radio(
                    [("TF-IDF", "tfidf"), ("GPU 임베딩", "embedding")],
                    value="tfidf",
                    label="콘텐츠 분석 방식",
                )
            with gr.Row():
                likes = gr.Textbox(label="좋아하는 메뉴", value="파스타, 돈까스")
                avoids = gr.Textbox(label="피하고 싶은 메뉴", value="오이")
            preferred = gr.CheckboxGroup(
                list(MENU_TYPE_KEYWORDS), value=["면", "튀김"], label="선호 유형"
            )
            spice = gr.Slider(1, 5, value=3, step=1, label="매운맛 선호")
            personal_button = gr.Button("학교 급식 콘텐츠 기반 추천", variant="primary")
            personal_message = gr.Markdown()
            with gr.Row():
                best_table = gr.Dataframe(label="예상 Best", interactive=False)
                worst_table = gr.Dataframe(label="예상 Worst", interactive=False)

            gr.Markdown(
                "## 2. 목포대 7월 30·31일 식사 전 예측\n"
                "익명 코드를 정한 뒤 실제 만족도를 매기기 전에 예측을 등록합니다. "
                "한번 등록한 값은 바꿀 수 없습니다."
            )
            with gr.Row():
                participant = gr.Textbox(label="익명 참여자 코드", value="학생A")
                grade = gr.Dropdown(
                    ["중1", "중2", "중3", "고1", "고2", "고3", "대학생"],
                    value="중2",
                    label="학년 구간",
                )
            mnu_prediction_button = gr.Button("목포대 이틀 식사 전 예측 등록")
            mnu_prediction_message = gr.Markdown()
            mnu_prediction_table = gr.Dataframe(
                label="7월 30·31일 콘텐츠 예측", interactive=False
            )

            gr.Markdown("## 3. 목포대 7월 30·31일 식사 후 실제 리뷰 등록")
            validation_menu = gr.Dropdown(
                validation_choices,
                value=validation_choices[0][1],
                label="목포대 검증 식단",
            )
            with gr.Row():
                actual_rating = gr.Slider(
                    1, 5, value=3, step=1, label="식사 후 실제 만족도"
                )
                feedback_tag = gr.Radio(
                    ["맛", "양", "매운맛", "메뉴 조합", "기타"],
                    value="맛",
                    label="피드백 태그",
                )
            add_button = gr.Button("예측값과 실제 리뷰를 모둠에 추가")
            add_message = gr.Markdown()
            feedback_table = gr.Dataframe(label="현재 모둠 익명 응답", interactive=False)

        with gr.Tab("30개 음식 역행렬 추천"):
            gr.Markdown(
                "## 학교 식단에서 뽑은 음식 30개를 평가해 보기\n"
                "각 음식의 평점을 1점(매우 싫음)부터 5점(매우 좋음)까지 바꾸면, "
                "아직 평가하지 않은 음식의 점수를 정규화 의사역행렬로 계산합니다.\n\n"
                "$$G = XX^T + 0.1I, \\quad "
                "w = X^T\\operatorname{pinv}(G)(y-3), \\quad "
                "\\hat y = \\operatorname{clip}(3 + X_{all}w, 1, 5)$$"
            )
            matrix_school = gr.Dropdown(
                schools, value=schools[0], label="평가할 학교"
            )
            with gr.Row():
                sample_button = gr.Button("음식 30개 뽑기", variant="primary")
                resample_button = gr.Button("다른 음식 다시 뽑기")
            sample_message = gr.Markdown()
            rating_table = gr.Dataframe(
                headers=["음식", "평점"],
                datatype=["str", "number"],
                value=pd.DataFrame(columns=["음식", "평점"]),
                label="30개 음식 평점표",
                interactive=True,
            )
            matrix_button = gr.Button("역행렬 추천 계산", variant="primary")
            matrix_message = gr.Markdown()
            with gr.Row():
                matrix_best = gr.Dataframe(label="미평가 음식 예상 Best", interactive=False)
                matrix_worst = gr.Dataframe(label="미평가 음식 예상 Worst", interactive=False)
            gr.Markdown(
                "### 음식 유사도 2차원 지도\n"
                "글자 N-그램 TF-IDF 벡터를 PCA 두 축으로 줄였습니다. 가까운 점은 이름에 "
                "비슷한 글자 조각이 많다는 뜻이며, 인과관계나 영양학적 유사성을 뜻하지는 않습니다."
            )
            food_map_plot = gr.Plot(label="음식 유사도 2차원 지도")
            food_map_table = gr.Dataframe(
                label="지도 번호와 음식 이름", interactive=False
            )

        with gr.Tab("모둠 피드백 분석"):
            gr.Markdown(
                "## 콘텐츠 기반·유저 기반·혼합 추천 검증\n"
                "각 실제 리뷰를 한 번씩 가린 뒤 예측하므로 정답을 미리 보는 누수를 막습니다."
            )
            with gr.Row():
                analyze_button = gr.Button("현재 모둠 분석", variant="primary")
                export_button = gr.Button("익명 CSV 만들기")
                export_file = gr.File(label="다운로드할 CSV", interactive=False)
            upload_file = gr.File(
                label="학생 PC에서 내려받은 익명 CSV(여러 파일 선택 가능)",
                type="filepath",
                file_count="multiple",
            )
            upload_button = gr.Button("CSV 합쳐서 분석")
            group_message = gr.Markdown()
            detail_table = gr.Dataframe(label="예측과 실제 리뷰 비교", interactive=False)
            with gr.Row():
                mae_table = gr.Dataframe(label="추천 방식별 MAE", interactive=False)
                tag_table = gr.Dataframe(label="피드백 태그", interactive=False)
            with gr.Row():
                cluster_table = gr.Dataframe(label="모둠 식성 군집", interactive=False)
                buddy_table = gr.Dataframe(label="밥친구", interactive=False)
            gr.Markdown("### Food MBTI\n성격검사가 아닌 수업용 식성 분류입니다.")
            with gr.Row():
                rice_noodle = gr.Slider(1, 5, value=3, step=1, label="밥·국 ↔ 면·간편식")
                mild_spicy = gr.Slider(1, 5, value=3, step=1, label="순한 맛 ↔ 매운 맛")
                familiar_new = gr.Slider(1, 5, value=3, step=1, label="익숙함 ↔ 새로움")
                dessert_hearty = gr.Slider(1, 5, value=3, step=1, label="디저트 ↔ 든든함")
            mbti_button = gr.Button("Food MBTI 확인")
            mbti_result = gr.Markdown()

        with gr.Tab("학교별 가치 음식"):
            gr.Markdown(
                "## 학교 상관없는 전체 음식 중요도 랭킹\n"
                f"{global_ranking_explanation}\n\n"
                "같은 음식은 여러 학교에 나와도 한 행으로 합칩니다. 이 순위는 전체 자료에서 "
                "얼마나 자주 등장하고 몇 개 학교에 퍼져 있는지를 함께 본 결과이며, 맛이나 "
                "영양 순위는 아닙니다."
            )
            overall_value_table = gr.Dataframe(
                value=global_ranking,
                label="학교 상관없는 전체 음식 중요도 랭킹",
                interactive=False,
            )
            gr.Markdown(
                "---\n## 선택 학교의 핵심 음식과 TF-IDF 데이터 가치 음식\n"
                "가장 자주 나온 음식은 **횟수**만 봅니다. 반면 데이터 가치 점수는 한 학교에서 "
                "자주 나오면서 다른 학교에는 덜 흔한 음식을 찾습니다.\n\n"
                "$$TF = \\frac{해당\\ 음식\\ 등장\\ 횟수}{학교\\ 전체\\ 음식\\ 수}$$\n"
                "$$IDF = \\ln\\left(\\frac{1+전체\\ 학교\\ 수}{1+해당\\ 음식이\\ 나온\\ 학교\\ 수}\\right)+1$$\n"
                "$$TF\text{-}IDF\\ 데이터\\ 가치\\ 점수 = TF \\times IDF$$"
            )
            school_map = gr.Dropdown(
                schools, value=schools[0], label="분석할 학교"
            )
            school_preference = gr.Textbox(
                label="급식 취향 문장", value="파스타 돈까스 치즈"
            )
            school_method = gr.Radio(
                [("TF-IDF", "tfidf"), ("GPU 임베딩", "embedding")],
                value="tfidf",
                label="분석 방식",
            )
            school_button = gr.Button("학교 핵심·가치 음식 계산", variant="primary")
            school_message = gr.Markdown()
            school_stats = gr.Dataframe(label="학교 통계", interactive=False)
            with gr.Row():
                frequency_table = gr.Dataframe(
                    label="가장 자주 나온 핵심 메뉴", interactive=False
                )
                value_table = gr.Dataframe(
                    label="TF-IDF 데이터 가치 점수 상세", interactive=False
                )
            with gr.Row():
                high_rank_table = gr.Dataframe(
                    label="급식 취향 기준 고등학교", interactive=False
                )

        with gr.Tab("AI 식단 실험실"):
            gr.Markdown(
                "## 영양사용 아이디어 실험\n"
                "가성비·잔반·꿀조합은 실제 측정값이 아니라 수업용 대체지표입니다."
            )
            lab_school = gr.Dropdown(schools, value=schools[0], label="학교")
            group_preferences = gr.Textbox(
                lines=5,
                label="익명 학생 취향(한 줄에 한 명)",
                value="파스타 치즈\n돈까스 튀김\n잡곡밥 생선",
            )
            lab_method = gr.Radio(
                [("TF-IDF", "tfidf"), ("GPU 임베딩", "embedding")],
                value="tfidf",
                label="분석 방식",
            )
            lab_button = gr.Button("Pareto 후보와 메뉴 조합 분석")
            lab_message = gr.Markdown()
            with gr.Row():
                pareto_table = gr.Dataframe(label="Pareto 후보", interactive=False)
                pair_table = gr.Dataframe(label="메뉴 조합 탐색", interactive=False)

        with gr.Tab("NVIDIA NIM 데이터 해설"):
            gr.Markdown(
                "## 계산 결과에 질문하기\n"
                "선택한 학교의 실제 통계·TF-IDF 값과 방금 계산한 역행렬 추천 결과를 "
                "근거로 설명합니다. 답변은 계산을 대신하는 새 예측이 아니라 결과를 읽는 "
                "도우미입니다. 질문과 아래 계산 근거는 NVIDIA NIM API로 전송됩니다."
            )
            chat_school = gr.Dropdown(
                schools, value=schools[0], label="질문할 학교"
            )
            chat_history = _message_chatbot_component(
                gr, label="NVIDIA NIM 급식 데이터 해설"
            )
            chat_question = gr.Textbox(
                label="질문",
                placeholder="예: 이 학교의 데이터 가치 1위 음식은 왜 1위인가요?",
            )
            with gr.Row():
                chat_button = gr.Button("데이터 근거로 답하기", variant="primary")
                chat_clear_button = gr.Button("대화 지우기")

        personal_button.click(
            lambda school_name, likes_text, avoids_text, types, spice_value, method_value: personal_recommendation_callback(
                dataset,
                school_name=school_name,
                likes_text=likes_text,
                avoids_text=avoids_text,
                preferred_types=types,
                spice_level=spice_value,
                method=method_value,
                embedder=embedder,
            ),
            inputs=[school, likes, avoids, preferred, spice, model_method],
            outputs=[personal_message, best_table, worst_table],
            api_name="personal_recommendation",
        )
        mnu_prediction_button.click(
            lambda state, code, grade_value, likes_value, avoids_value, types, spice_value, method_value: register_mnu_predictions_callback(
                validation_menus,
                state,
                participant_code=code,
                grade_band=grade_value,
                likes_text=likes_value,
                avoids_text=avoids_value,
                preferred_types=types,
                spice_level=spice_value,
                method=method_value,
                embedder=embedder,
            ),
            inputs=[
                prediction_state,
                participant,
                grade,
                likes,
                avoids,
                preferred,
                spice,
                model_method,
            ],
            outputs=[prediction_state, mnu_prediction_message, mnu_prediction_table],
            api_name="register_mnu_prediction",
        )
        for trigger in (sample_button, resample_button):
            trigger.click(
                lambda school_name, seed: sample_foods_callback(
                    dataset, school_name=school_name, seed=seed
                ),
                inputs=[matrix_school, survey_seed_state],
                outputs=[
                    survey_seed_state,
                    survey_foods_state,
                    sample_message,
                    rating_table,
                ],
                api_name=(
                    "sample_school_foods" if trigger is sample_button else False
                ),
            )
        matrix_button.click(
            lambda school_name, sampled_foods, ratings: matrix_recommendation_callback(
                dataset,
                school_name=school_name,
                sampled_foods=sampled_foods,
                ratings=ratings,
            ),
            inputs=[matrix_school, survey_foods_state, rating_table],
            outputs=[
                matrix_message,
                matrix_best,
                matrix_worst,
                food_map_plot,
                food_map_table,
                matrix_state,
            ],
            api_name="matrix_recommendation",
        )
        add_button.click(
            lambda state, predictions, code, grade_value, selected_menu, likes_value, avoids_value, types, spice_value, rating, tag, method_value: add_feedback_callback(
                validation_menus,
                state,
                prediction_state=predictions,
                participant_code=code,
                grade_band=grade_value,
                menu_id=selected_menu,
                likes_text=likes_value,
                avoids_text=avoids_value,
                preferred_types=types,
                spice_level=spice_value,
                actual_rating=rating,
                feedback_tag=tag,
                method=method_value,
                embedder=embedder,
            ),
            inputs=[
                feedback_state,
                prediction_state,
                participant,
                grade,
                validation_menu,
                likes,
                avoids,
                preferred,
                spice,
                actual_rating,
                feedback_tag,
                model_method,
            ],
            outputs=[feedback_state, feedback_table, add_message],
            api_name="add_feedback",
        )
        analyze_button.click(
            analyze_feedback_callback,
            inputs=[feedback_state],
            outputs=[
                group_message,
                detail_table,
                mae_table,
                tag_table,
                cluster_table,
                buddy_table,
            ],
            api_name="analyze_feedback",
        )
        export_button.click(
            export_feedback_callback,
            inputs=[feedback_state],
            outputs=[export_file],
            api_name="export_feedback",
        )
        upload_button.click(
            analyze_uploaded_feedback_callback,
            inputs=[upload_file, feedback_state],
            outputs=[
                feedback_state,
                group_message,
                detail_table,
                mae_table,
                tag_table,
                cluster_table,
                buddy_table,
            ],
            api_name="upload_feedback",
        )
        mbti_button.click(
            food_mbti_callback,
            inputs=[rice_noodle, mild_spicy, familiar_new, dessert_hearty],
            outputs=[mbti_result],
            api_name="food_mbti",
        )
        school_button.click(
            lambda school_name, preference, method_value: school_value_analysis_callback(
                dataset,
                school_name=school_name,
                preference_text=preference,
                method=method_value,
                embedder=embedder,
            ),
            inputs=[school_map, school_preference, school_method],
            outputs=[
                school_message,
                school_stats,
                frequency_table,
                value_table,
                overall_value_table,
                high_rank_table,
            ],
            api_name="school_food_values",
        )
        lab_button.click(
            lambda school_name, preferences, method_value: lab_callback(
                dataset,
                school_name=school_name,
                group_preferences_text=preferences,
                method=method_value,
                embedder=embedder,
            ),
            inputs=[lab_school, group_preferences, lab_method],
            outputs=[lab_message, pareto_table, pair_table],
            api_name="meal_lab",
        )
        chat_button.click(
            lambda school_name, question, history, current_matrix: meal_chat_callback(
                dataset,
                school_name=school_name,
                question=question,
                history=history,
                nim_client=nim_client,
                matrix_state=current_matrix,
            ),
            inputs=[chat_school, chat_question, chat_history, matrix_state],
            outputs=[chat_history, chat_question],
            api_name="nim_meal_chat",
        )
        chat_clear_button.click(
            lambda: ([], ""),
            outputs=[chat_history, chat_question],
            api_name=False,
        )
        if profile_store is not None and classroom_mode:
            def _load_authenticated_profile(request: gr.Request):
                return load_profile_callback(profile_store, request.username)

            def _save_authenticated_profile(ratings, request: gr.Request):
                return save_profile_callback(
                    profile_store, request.username, ratings
                )

            # ``from __future__ import annotations`` 때문에 Gradio가 문자열형
            # Request 주석을 일반 입력으로 세지 않도록 실제 클래스로 돌려놓는다.
            _load_authenticated_profile.__annotations__["request"] = gr.Request
            _save_authenticated_profile.__annotations__["request"] = gr.Request

            profile_load_button.click(
                _load_authenticated_profile,
                outputs=[profile_message, profile_table],
                api_name="load_student_profile",
            )
            profile_save_button.click(
                _save_authenticated_profile,
                inputs=[profile_table],
                outputs=[profile_message, profile_table],
                api_name="save_student_profile",
            )
            demo.load(
                _load_authenticated_profile,
                outputs=[profile_message, profile_table],
                api_name=False,
            )
        if profile_store is not None:
            matrix_refresh_button.click(
                lambda: matrix_dashboard_callback(
                    profile_store, anonymize=classroom_mode
                ),
                outputs=[
                    class_matrix_message,
                    profile_status_table,
                    observed_matrix_table,
                    completed_matrix_table,
                    matrix_metrics_table,
                    matrix_recommendations,
                    matrix_heatmap,
                    student_coordinates,
                    student_map,
                ],
                api_name="student_matrix_factorization",
            )
            if not classroom_mode:
                ratings_export_button.click(
                    lambda: export_ratings_callback(profile_store),
                    outputs=[ratings_export_file],
                    api_name="export_student_ratings",
                )
        gr.Markdown(
            "---\n추천 결과는 분석 실습용입니다. 실제 급식·알레르기·영양 판단은 "
            "학교 식단표와 영양사 안내를 먼저 확인하세요."
        )
    return demo
