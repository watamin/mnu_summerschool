"""수업용 TF-IDF와 선택적 Sentence Transformer 임베딩 경계."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class TextEmbedder(Protocol):
    device: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class VectorResult:
    matrix: np.ndarray
    backend: str
    device: str
    notice: str


def _char_ngrams(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for word in str(text or "").casefold().split():
        padded = f" {word} "
        for size in (2, 3, 4):
            counts.update(
                padded[index : index + size]
                for index in range(len(padded) - size + 1)
            )
    return counts


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("텍스트 벡터는 2차원 행렬이어야 합니다.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _tfidf_matrix(texts: Sequence[str]) -> np.ndarray:
    counters = [_char_ngrams(text) for text in texts]
    vocabulary = sorted({term for counter in counters for term in counter})
    matrix = np.zeros((len(counters), len(vocabulary)), dtype=float)
    if not vocabulary:
        return matrix
    document_frequency = Counter()
    for counter in counters:
        document_frequency.update(counter.keys())
    idf = {
        term: math.log((1 + len(counters)) / (1 + document_frequency[term])) + 1.0
        for term in vocabulary
    }
    positions = {term: index for index, term in enumerate(vocabulary)}
    for row_index, counter in enumerate(counters):
        total = sum(counter.values()) or 1
        for term, count in counter.items():
            matrix[row_index, positions[term]] = (count / total) * idf[term]
    return _normalize(matrix)


class SentenceTransformerEmbedder:
    """모델과 torch를 첫 임베딩 요청 때만 불러온다."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.device = "확인 전"
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=float,
        )


def encode_texts(
    texts: Sequence[str],
    *,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> VectorResult:
    """텍스트를 정규화 벡터로 바꾸고 실제 사용한 방식을 기록한다."""

    cleaned = [str(text or "").strip() for text in texts]
    if method == "tfidf":
        return VectorResult(
            matrix=_tfidf_matrix(cleaned),
            backend="tfidf",
            device="cpu",
            notice="문자 n-gram TF-IDF를 사용했습니다.",
        )
    if method != "embedding":
        raise ValueError("분석 방식은 tfidf 또는 embedding이어야 합니다.")
    selected = embedder or SentenceTransformerEmbedder()
    try:
        matrix = _normalize(np.asarray(selected.encode(cleaned), dtype=float))
    except Exception:
        return VectorResult(
            matrix=_tfidf_matrix(cleaned),
            backend="tfidf",
            device="cpu",
            notice=(
                "임베딩 모델을 사용할 수 없어 TF-IDF로 자동 전환했습니다. "
                "교사용 GPU 설치 안내를 확인하세요."
            ),
        )
    return VectorResult(
        matrix=matrix,
        backend="embedding",
        device=str(selected.device),
        notice=(
            f"다국어 MiniLM 384차원 임베딩을 {selected.device} 장치에서 사용했습니다."
        ),
    )


def cosine_scores(
    query: str,
    documents: Sequence[str],
    *,
    method: str = "tfidf",
    embedder: TextEmbedder | None = None,
) -> tuple[np.ndarray, VectorResult]:
    """질문 하나와 문서 목록을 같은 공간에 놓고 코사인 유사도를 구한다."""

    result = encode_texts(
        [str(query or ""), *[str(document or "") for document in documents]],
        method=method,
        embedder=embedder,
    )
    if len(documents) == 0:
        return np.asarray([], dtype=float), result
    scores = result.matrix[1:] @ result.matrix[0]
    return np.clip(scores.astype(float), -1.0, 1.0), result
