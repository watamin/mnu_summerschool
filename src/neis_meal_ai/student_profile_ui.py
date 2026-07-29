"""학생 프로필 저장과 행렬분해 실험 화면의 콜백."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .matrix_factorization import MatrixAnalysis, analyze_rating_matrix
from .student_profiles import StudentProfileStore, validate_student_name


_SURVEY_COLUMNS = ["순서", "음식", "구분", "평점"]


def load_profile_callback(
    store: StudentProfileStore, username: object
) -> tuple[str, pd.DataFrame]:
    """로그인 학생의 저장된 평가표를 복원한다."""

    name = validate_student_name(username)
    survey = store.load_survey(name)
    answered = int((pd.to_numeric(survey["평점"], errors="coerce") > 0).sum())
    return (
        f"### {name} 학생의 평가표\n현재 **{answered}/30개**가 저장되어 있습니다. "
        "평점은 1점(매우 싫음)부터 5점(매우 좋음)까지 입력하세요.",
        survey,
    )


def _survey_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        frame = pd.DataFrame(value)
    if not set(_SURVEY_COLUMNS).issubset(frame.columns):
        raise ValueError("평가표의 순서·음식·구분·평점 열을 그대로 사용해 주세요.")
    return frame[_SURVEY_COLUMNS]


def save_profile_callback(
    store: StudentProfileStore, username: object, survey_value: object
) -> tuple[str, pd.DataFrame]:
    """화면의 30개 평점을 저장하고 DB의 최신 표를 다시 보여 준다."""

    name = validate_student_name(username)
    frame = _survey_frame(survey_value)
    assigned = store.load_survey(name)
    if (
        len(frame) != len(assigned)
        or frame["순서"].tolist() != assigned["순서"].tolist()
        or frame["음식"].astype(str).tolist() != assigned["음식"].tolist()
        or frame["구분"].astype(str).tolist() != assigned["구분"].tolist()
    ):
        raise ValueError("배정된 음식 이름과 순서를 바꾸지 말고 평점 열만 입력해 주세요.")
    result = store.save_ratings(name, frame)
    restored = store.load_survey(name)
    if result.complete:
        state = "30개 평가를 모두 저장했습니다. 이제 모둠 행렬분해에 사용할 수 있습니다."
    else:
        state = "중간 저장되었습니다. 나중에 같은 이름으로 로그인해 이어서 평가할 수 있습니다."
    message = (
        f"### 저장 완료: {result.saved_count}/{result.total_questions}\n"
        f"{state} 마지막 저장: `{result.updated_at}`"
    )
    return message, restored


def _observed_display(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.map(lambda value: "" if pd.isna(value) else f"{float(value):.0f} 실제")


def _completed_display(matrix: pd.DataFrame, analysis: MatrixAnalysis) -> pd.DataFrame:
    result = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(analysis.completed_matrix.iat[row, column])
            label = "예측" if analysis.prediction_mask.iat[row, column] else "실제"
            result.iat[row, column] = f"{value:.2f} {label}"
    return result


def matrix_heatmap_figure(analysis: MatrixAnalysis):
    """실제 셀과 예측 셀을 점 모양으로 구별한 평점 열지도를 그린다."""

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    values = analysis.completed_matrix.to_numpy(dtype=float)
    predicted = analysis.prediction_mask.to_numpy(dtype=bool)
    figure, axis = plt.subplots(figsize=(16, max(4.5, values.shape[0] * 0.8)))
    image = axis.imshow(values, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
    actual_y, actual_x = np.where(~predicted)
    predicted_y, predicted_x = np.where(predicted)
    axis.scatter(actual_x, actual_y, marker="o", s=15, c="black", alpha=0.75)
    axis.scatter(
        predicted_x,
        predicted_y,
        marker="o",
        s=22,
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
    )
    axis.set_xticks(range(values.shape[1]))
    axis.set_xticklabels([str(index) for index in range(1, values.shape[1] + 1)], fontsize=7)
    axis.set_yticks(range(values.shape[0]))
    axis.set_yticklabels([f"S{index}" for index in range(1, values.shape[0] + 1)])
    axis.set_xlabel("Food number (see completed matrix columns)")
    axis.set_ylabel("Student number (see status table)")
    axis.set_title("Observed and predicted rating matrix")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="black", label="Observed"),
            Line2D([0], [0], marker="o", color="white", markerfacecolor="none", label="Predicted"),
        ],
        loc="upper right",
    )
    figure.colorbar(image, ax=axis, label="Rating (1-5)")
    figure.tight_layout()
    return figure


def student_map_figure(coordinates: pd.DataFrame):
    """학생 잠재벡터의 앞 두 축을 번호형 산점도로 표시한다."""

    from matplotlib import pyplot as plt

    if coordinates.empty:
        raise ValueError("그릴 학생 좌표가 없습니다.")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(coordinates["X"], coordinates["Y"], s=100, color="#3b82f6")
    for number, row in enumerate(coordinates.itertuples(index=False), start=1):
        axis.annotate(
            f"S{number}",
            (float(row.X), float(row.Y)),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.axhline(0, color="#d0d0d0", linewidth=0.8)
    axis.axvline(0, color="#d0d0d0", linewidth=0.8)
    axis.set_xlabel("Latent preference axis 1")
    axis.set_ylabel("Latent preference axis 2")
    axis.set_title("Student taste map")
    figure.tight_layout()
    return figure


def matrix_dashboard_callback(store: StudentProfileStore):
    """현재 DB의 평점 행렬을 평가하고 화면용 결과 묶음을 반환한다."""

    status = store.status()
    matrix = store.rating_matrix()
    observed_display = _observed_display(matrix)
    empty = pd.DataFrame()
    if len(matrix.index) < 2:
        return (
            "행렬분해를 시작하려면 평점을 저장한 학생이 **2명 이상** 필요합니다.",
            status,
            observed_display,
            empty,
            empty,
            empty,
            None,
            empty,
            None,
        )
    try:
        analysis = analyze_rating_matrix(matrix)
    except ValueError as exc:
        return (
            f"아직 행렬분해를 계산할 수 없습니다: {exc}",
            status,
            observed_display,
            empty,
            empty,
            empty,
            None,
            empty,
            None,
        )

    actual_count = int(matrix.notna().sum().sum())
    missing_count = int(matrix.isna().sum().sum())
    evaluation = analysis.evaluation
    message = (
        f"### 계산 완료\n**{len(matrix)}명**, **{actual_count}개 실제 평점**, "
        f"**{missing_count}개 빈칸 예측**입니다. 검증에서는 알고 있던 "
        f"{evaluation.holdout_count}개를 잠시 가린 뒤 다시 맞혔습니다. "
        "MAE가 전체평균 기준선보다 작을수록 학생별 취향 구조를 더 잘 찾은 것입니다."
    )
    metrics = pd.DataFrame(
        [
            {"지표": "행렬분해 MAE", "값": round(evaluation.mae, 4), "뜻": "실제 평점과 예측 평점 차이의 절댓값 평균"},
            {"지표": "행렬분해 RMSE", "값": round(evaluation.rmse, 4), "뜻": "큰 오차에 더 큰 벌점을 주는 평균 오차"},
            {"지표": "전체평균 기준선 MAE", "값": round(evaluation.baseline_mae, 4), "뜻": "모두에게 같은 평균을 준 단순 예측의 오차"},
        ]
    )
    return (
        message,
        status,
        observed_display,
        _completed_display(matrix, analysis),
        metrics,
        analysis.best_worst,
        matrix_heatmap_figure(analysis),
        analysis.user_coordinates,
        student_map_figure(analysis.user_coordinates),
    )


def export_ratings_callback(store: StudentProfileStore) -> str:
    """교사가 내려받을 수 있는 전체 실제 평점 CSV 경로를 반환한다."""

    destination = Path(store.db_path).parent / "exports" / "student_ratings.csv"
    return str(store.export_ratings(destination))
