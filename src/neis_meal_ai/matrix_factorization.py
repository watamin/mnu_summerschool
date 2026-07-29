"""관측된 학생×음식 평점으로 실제 빈칸을 예측한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatrixFactorizationModel:
    """학생·음식 편향과 저차원 잠재벡터를 학습한 결과."""

    global_mean: float
    user_bias: np.ndarray
    item_bias: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_names: tuple[str, ...]
    item_names: tuple[str, ...]

    def predict_all(self) -> np.ndarray:
        raw = (
            self.global_mean
            + self.user_bias[:, None]
            + self.item_bias[None, :]
            + self.user_factors @ self.item_factors.T
        )
        return np.clip(raw, 1.0, 5.0)


@dataclass(frozen=True)
class EvaluationResult:
    """알고 있는 평점을 가려 계산한 모델·기준선 오차."""

    holdout_count: int
    mae: float
    rmse: float
    baseline_mae: float
    actual: tuple[float, ...]
    predicted: tuple[float, ...]


@dataclass(frozen=True)
class MatrixAnalysis:
    """화면에 필요한 완성 행렬, 추천, 검증 및 좌표."""

    completed_matrix: pd.DataFrame
    prediction_mask: pd.DataFrame
    best_worst: pd.DataFrame
    user_coordinates: pd.DataFrame
    evaluation: EvaluationResult
    model: MatrixFactorizationModel


def _validated_values(matrix: pd.DataFrame) -> np.ndarray:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("평점 행렬은 pandas DataFrame이어야 합니다.")
    if matrix.shape[0] < 2:
        raise ValueError("행렬분해에는 학생이 2명 이상 필요합니다.")
    if matrix.shape[1] < 2:
        raise ValueError("행렬분해에는 음식이 2개 이상 필요합니다.")
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError("학생 이름과 음식 이름은 중복될 수 없습니다.")
    try:
        values = matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("평점 행렬에는 숫자와 빈칸만 사용할 수 있습니다.") from exc
    observed = values[~np.isnan(values)]
    if observed.size < 4:
        raise ValueError("행렬분해에는 저장된 평점이 더 필요합니다.")
    if not np.isfinite(observed).all() or np.any((observed < 1) | (observed > 5)):
        raise ValueError("저장된 평점은 1부터 5 사이여야 합니다.")
    if np.any(np.sum(~np.isnan(values), axis=1) == 0):
        raise ValueError("모든 학생에게 저장된 평점이 하나 이상 있어야 합니다.")
    if np.any(np.sum(~np.isnan(values), axis=0) == 0):
        raise ValueError("모든 음식에 저장된 평점이 하나 이상 있어야 합니다.")
    return values


def fit_matrix_factorization(
    matrix: pd.DataFrame,
    *,
    rank: int = 3,
    seed: int = 42,
    epochs: int = 900,
    learning_rate: float = 0.018,
    regularization: float = 0.035,
) -> MatrixFactorizationModel:
    """결측값을 제외한 관측 평점만으로 편향 포함 행렬분해를 학습한다."""

    values = _validated_values(matrix)
    if int(rank) < 1:
        raise ValueError("잠재 차원은 1 이상이어야 합니다.")
    rank = min(int(rank), values.shape[0], values.shape[1])
    observed_positions = np.argwhere(~np.isnan(values))
    observed_values = values[~np.isnan(values)]
    global_mean = float(np.mean(observed_values))

    rng = np.random.default_rng(seed)
    user_bias = np.zeros(values.shape[0], dtype=float)
    item_bias = np.zeros(values.shape[1], dtype=float)
    user_factors = rng.normal(0.0, 0.08, size=(values.shape[0], rank))
    item_factors = rng.normal(0.0, 0.08, size=(values.shape[1], rank))

    rate = float(learning_rate)
    for _ in range(int(epochs)):
        order = rng.permutation(len(observed_positions))
        for observation_index in order:
            user, item = observed_positions[observation_index]
            rating = values[user, item]
            predicted = (
                global_mean
                + user_bias[user]
                + item_bias[item]
                + float(user_factors[user] @ item_factors[item])
            )
            error = rating - predicted
            user_bias[user] += rate * (error - regularization * user_bias[user])
            item_bias[item] += rate * (error - regularization * item_bias[item])
            old_user = user_factors[user].copy()
            user_factors[user] += rate * (
                error * item_factors[item] - regularization * user_factors[user]
            )
            item_factors[item] += rate * (
                error * old_user - regularization * item_factors[item]
            )
        rate *= 0.997

    return MatrixFactorizationModel(
        global_mean=global_mean,
        user_bias=user_bias,
        item_bias=item_bias,
        user_factors=user_factors,
        item_factors=item_factors,
        user_names=tuple(str(name) for name in matrix.index),
        item_names=tuple(str(name) for name in matrix.columns),
    )


def _holdout_split(
    matrix: pd.DataFrame, *, seed: int
) -> tuple[pd.DataFrame, list[tuple[int, int, float]]]:
    values = _validated_values(matrix)
    train = values.copy()
    remaining_by_user = np.sum(~np.isnan(train), axis=1).astype(int)
    remaining_by_item = np.sum(~np.isnan(train), axis=0).astype(int)
    rng = np.random.default_rng(seed)
    held_out: list[tuple[int, int, float]] = []

    for user in range(train.shape[0]):
        candidates = np.flatnonzero(~np.isnan(train[user]))
        candidates = rng.permutation(candidates)
        target = max(1, int(np.floor(len(candidates) * 0.2)))
        selected = 0
        for item in candidates:
            if selected >= target:
                break
            if remaining_by_user[user] <= 1 or remaining_by_item[item] <= 1:
                continue
            held_out.append((user, int(item), float(train[user, item])))
            train[user, item] = np.nan
            remaining_by_user[user] -= 1
            remaining_by_item[item] -= 1
            selected += 1

    if not held_out:
        raise ValueError(
            "검증용 실제 평점을 가릴 수 없습니다. 학생과 음식이 겹치는 평가를 더 저장해 주세요."
        )
    return pd.DataFrame(train, index=matrix.index, columns=matrix.columns), held_out


def evaluate_matrix_factorization(
    matrix: pd.DataFrame, *, rank: int = 3, seed: int = 42
) -> EvaluationResult:
    """관측 평점 일부를 가려 실제값과 예측값의 MAE·RMSE를 계산한다."""

    train, held_out = _holdout_split(matrix, seed=seed)
    model = fit_matrix_factorization(train, rank=rank, seed=seed)
    scores = model.predict_all()
    actual = np.array([rating for _, _, rating in held_out], dtype=float)
    predicted = np.array([scores[user, item] for user, item, _ in held_out])
    errors = actual - predicted
    training_values = train.to_numpy(dtype=float)
    baseline = float(np.nanmean(training_values))
    return EvaluationResult(
        holdout_count=len(held_out),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        baseline_mae=float(np.mean(np.abs(actual - baseline))),
        actual=tuple(float(value) for value in actual),
        predicted=tuple(float(value) for value in predicted),
    )


def analyze_rating_matrix(
    matrix: pd.DataFrame, *, rank: int = 3, seed: int = 42
) -> MatrixAnalysis:
    """전체 관측값으로 빈칸을 채우고 학생별 미평가 음식 순위를 만든다."""

    values = _validated_values(matrix)
    evaluation = evaluate_matrix_factorization(matrix, rank=rank, seed=seed)
    model = fit_matrix_factorization(matrix, rank=rank, seed=seed)
    scores = model.predict_all()
    prediction_mask = pd.DataFrame(
        np.isnan(values), index=matrix.index, columns=matrix.columns
    )
    completed_values = np.where(prediction_mask.to_numpy(), scores, values)
    completed = pd.DataFrame(
        completed_values, index=matrix.index, columns=matrix.columns
    ).round(2)

    recommendation_rows: list[dict[str, object]] = []
    for user_index, student in enumerate(matrix.index):
        missing_items = np.flatnonzero(prediction_mask.iloc[user_index].to_numpy())
        ranked = sorted(
            (
                (str(matrix.columns[item]), float(scores[user_index, item]))
                for item in missing_items
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        for label, foods in (
            ("예상 Best", ranked[:5]),
            ("예상 Worst", sorted(ranked, key=lambda pair: (pair[1], pair[0]))[:5]),
        ):
            for position, (food, score) in enumerate(foods, start=1):
                recommendation_rows.append(
                    {
                        "학생": str(student),
                        "구분": label,
                        "순위": position,
                        "음식": food,
                        "예상 평점": round(score, 2),
                    }
                )
    best_worst = pd.DataFrame.from_records(
        recommendation_rows,
        columns=["학생", "구분", "순위", "음식", "예상 평점"],
    )

    factors = model.user_factors
    x = factors[:, 0]
    y = factors[:, 1] if factors.shape[1] > 1 else np.zeros(len(factors))
    coordinates = pd.DataFrame(
        {"학생": [str(name) for name in matrix.index], "X": x, "Y": y}
    ).round({"X": 4, "Y": 4})
    return MatrixAnalysis(
        completed_matrix=completed,
        prediction_mask=prediction_mask,
        best_worst=best_worst,
        user_coordinates=coordinates,
        evaluation=evaluation,
        model=model,
    )
