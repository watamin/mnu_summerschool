from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neis_meal_ai.matrix_factorization import (
    analyze_rating_matrix,
    evaluate_matrix_factorization,
    fit_matrix_factorization,
)


def incomplete_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [5, 4, np.nan, 1, np.nan],
            [4, np.nan, 5, 2, 1],
            [1, 2, np.nan, 5, 4],
            [np.nan, 1, 2, 4, 5],
        ],
        index=["학생A", "학생B", "학생C", "학생D"],
        columns=["김치", "카레", "돈까스", "국수", "샐러드"],
        dtype=float,
    )


def clustered_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [5, 5, 4, 4, 2, 2, 1, 1],
            [5, 4, 5, 4, 2, 1, 2, 1],
            [4, 5, 4, 5, 1, 2, 1, 2],
            [1, 1, 2, 2, 4, 4, 5, 5],
            [1, 2, 1, 2, 5, 4, 5, 4],
            [2, 1, 2, 1, 4, 5, 4, 5],
        ],
        index=[f"학생{i}" for i in "ABCDEF"],
        columns=[f"음식{i}" for i in range(1, 9)],
        dtype=float,
    )


def test_fit_is_deterministic_and_predictions_stay_on_rating_scale() -> None:
    matrix = incomplete_matrix()

    first = fit_matrix_factorization(matrix, rank=3, seed=17)
    second = fit_matrix_factorization(matrix, rank=3, seed=17)
    first_scores = first.predict_all()
    second_scores = second.predict_all()

    np.testing.assert_allclose(first_scores, second_scores, atol=1e-12)
    assert first_scores.shape == matrix.shape
    assert np.isfinite(first_scores).all()
    assert float(first_scores.min()) >= 1.0
    assert float(first_scores.max()) <= 5.0


def test_analysis_preserves_observed_cells_and_fills_only_real_gaps() -> None:
    matrix = incomplete_matrix()

    analysis = analyze_rating_matrix(matrix, seed=9)

    observed = matrix.notna()
    np.testing.assert_allclose(
        analysis.completed_matrix.to_numpy()[observed.to_numpy()],
        matrix.to_numpy()[observed.to_numpy()],
    )
    assert analysis.completed_matrix.notna().all().all()
    assert analysis.prediction_mask.equals(matrix.isna())
    assert len(analysis.best_worst) == int(matrix.isna().sum().sum()) * 2
    assert set(analysis.best_worst["구분"]) == {"예상 Best", "예상 Worst"}
    assert list(analysis.user_coordinates.columns) == ["학생", "X", "Y"]
    assert analysis.user_coordinates["학생"].tolist() == matrix.index.tolist()


def test_holdout_metrics_are_finite_and_beat_global_mean_on_clustered_data() -> None:
    evaluation = evaluate_matrix_factorization(clustered_matrix(), seed=42)

    assert evaluation.holdout_count == 6
    assert np.isfinite(evaluation.mae)
    assert np.isfinite(evaluation.rmse)
    assert evaluation.mae < evaluation.baseline_mae
    assert evaluation.rmse >= evaluation.mae


def test_invalid_or_insufficient_rating_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="2명"):
        fit_matrix_factorization(
            pd.DataFrame([[5, 4]], index=["학생A"], columns=["음식1", "음식2"])
        )
    invalid = incomplete_matrix()
    invalid.loc["학생A", "김치"] = 8
    with pytest.raises(ValueError, match="1부터 5"):
        fit_matrix_factorization(invalid)
    too_sparse = pd.DataFrame(
        [[5, 4, np.nan, np.nan], [np.nan, np.nan, 1, 2]],
        index=["학생A", "학생B"],
        columns=["음식1", "음식2", "음식3", "음식4"],
    )
    with pytest.raises(ValueError, match="검증"):
        evaluate_matrix_factorization(too_sparse)
