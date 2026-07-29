from __future__ import annotations

import numpy as np
import pytest

from neis_meal_ai.text_vectors import cosine_scores, encode_texts


class FakeCudaEmbedder:
    device = "cuda"

    def encode(self, texts: list[str]) -> np.ndarray:
        mapping = {
            "치즈 파스타": [1.0, 0.0],
            "파스타 피자": [1.0, 0.0],
            "매운 닭갈비": [0.0, 1.0],
        }
        return np.asarray([mapping[text] for text in texts], dtype=float)


class BrokenEmbedder:
    device = "cuda"

    def encode(self, _texts: list[str]) -> np.ndarray:
        raise ImportError("sentence-transformers is unavailable")


def test_tfidf_vectors_are_normalized_and_empty_text_stays_zero() -> None:
    result = encode_texts(["치즈 파스타", "매운 닭갈비", ""], method="tfidf")

    assert result.backend == "tfidf"
    assert result.device == "cpu"
    assert result.matrix.shape[0] == 3
    assert np.linalg.norm(result.matrix[0]) == pytest.approx(1.0)
    assert np.linalg.norm(result.matrix[1]) == pytest.approx(1.0)
    assert np.linalg.norm(result.matrix[2]) == pytest.approx(0.0)


def test_tfidf_cosine_ranks_an_identical_korean_menu_first() -> None:
    scores, result = cosine_scores(
        "치즈 파스타",
        ["치즈 파스타", "고추장 닭갈비"],
        method="tfidf",
    )

    assert result.backend == "tfidf"
    assert scores[0] == pytest.approx(1.0)
    assert scores[0] > scores[1]


def test_embedding_mode_uses_the_embedder_device_and_cosine_vectors() -> None:
    scores, result = cosine_scores(
        "치즈 파스타",
        ["파스타 피자", "매운 닭갈비"],
        method="embedding",
        embedder=FakeCudaEmbedder(),
    )

    assert result.backend == "embedding"
    assert result.device == "cuda"
    assert result.matrix.shape == (3, 2)
    assert scores.tolist() == pytest.approx([1.0, 0.0])


def test_embedding_failure_falls_back_to_tfidf_with_an_explanation() -> None:
    scores, result = cosine_scores(
        "치즈 파스타",
        ["치즈 파스타", "매운 닭갈비"],
        method="embedding",
        embedder=BrokenEmbedder(),
    )

    assert result.backend == "tfidf"
    assert result.device == "cpu"
    assert "TF-IDF로 자동 전환" in result.notice
    assert scores[0] > scores[1]


def test_unknown_vector_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="분석 방식"):
        encode_texts(["급식"], method="unknown")
