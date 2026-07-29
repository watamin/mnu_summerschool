"""NVIDIA NIM에 계산 근거만 보내는 한국어 급식 데이터 해설 경계."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import requests


DEFAULT_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NIM_MODEL = "meta/llama-3.1-8b-instruct"
_KEY_PATTERN = re.compile(r"nvapi-[A-Za-z0-9_.-]+")


class NimChatError(RuntimeError):
    """키·통신·응답 문제를 비밀값 없이 화면에 전달한다."""


def _validated_key(value: str) -> str:
    candidate = str(value or "").strip().strip('"').strip("'")
    if _KEY_PATTERN.fullmatch(candidate) is None:
        raise NimChatError("NVIDIA NIM API 키 형식을 확인해 주세요.")
    return candidate


def load_nvidia_api_key(
    *,
    key_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """환경변수 또는 외부 파일에서 키를 읽고 값은 노출하지 않는다."""

    environment = os.environ if environ is None else environ
    environment_key = str(environment.get("NVIDIA_API_KEY", "")).strip()
    if environment_key:
        return _validated_key(environment_key)
    configured_path = str(environment.get("NVIDIA_NIM_KEY_FILE", "")).strip()
    selected_path = Path(configured_path) if configured_path else (
        Path(key_path) if key_path is not None else None
    )
    if selected_path is None:
        raise NimChatError("NVIDIA NIM API 키 파일을 찾을 수 없습니다.")
    try:
        raw = selected_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise NimChatError("NVIDIA NIM API 키 파일을 읽을 수 없습니다.") from exc
    if "=" in raw:
        name, raw = raw.split("=", 1)
        if name.strip() not in {"NVIDIA_API_KEY", "nvidia_api_key", "nvidia_nim_api_key"}:
            raise NimChatError("NVIDIA NIM API 키 파일 형식을 확인해 주세요.")
    return _validated_key(raw)


def build_grounded_messages(
    question: str,
    context: str,
    history: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """근거와 최근 대화만 포함한 NIM Chat Completions 메시지를 만든다."""

    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        raise NimChatError("질문을 한 글자 이상 입력해 주세요.")
    if len(cleaned_question) > 500:
        raise NimChatError("질문은 500자 이내로 입력해 주세요.")
    cleaned_context = str(context or "").strip()
    if not cleaned_context:
        raise NimChatError("질문에 연결할 급식 데이터 근거가 없습니다.")
    system = (
        "너는 중학생에게 학교 급식 데이터와 AI 계산을 설명하는 한국어 도우미다. "
        "아래 근거 데이터 안의 숫자와 음식만 사용한다. 근거에 없는 사실은 "
        "'현재 데이터로는 알 수 없습니다'라고 답한다. 영양, 알레르기, 건강 안전을 "
        "판정하지 않는다. 먼저 결론을 말하고 TF, IDF, TF-IDF 또는 행렬 계산을 "
        "필요한 만큼 쉬운 말로 설명한다. 답은 8문장 이내로 쓴다.\n\n"
        f"[근거 데이터]\n{cleaned_context[:12000]}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for item in list(history or [])[-12:]:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:1200]})
    messages.append({"role": "user", "content": cleaned_question})
    return messages


class NvidiaNimClient:
    """NVIDIA의 OpenAI 호환 Chat Completions를 지연 호출한다."""

    def __init__(
        self,
        *,
        key_path: str | Path | None = None,
        base_url: str = DEFAULT_NIM_URL,
        model: str = DEFAULT_NIM_MODEL,
        timeout: float = 30.0,
        session: object | None = None,
    ) -> None:
        self.key_path = Path(key_path) if key_path is not None else None
        self.base_url = str(base_url)
        self.model = str(model)
        self.timeout = float(timeout)
        self.session = session or requests.Session()

    def ask(
        self,
        question: str,
        context: str,
        history: Sequence[Mapping[str, str]] = (),
    ) -> str:
        key = load_nvidia_api_key(key_path=self.key_path)
        messages = build_grounded_messages(question, context, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 700,
            "stream": False,
        }
        try:
            response = self.session.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise NimChatError(
                "NVIDIA NIM에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요."
            ) from exc
        status_code = int(getattr(response, "status_code", 0))
        if not 200 <= status_code < 300:
            if status_code in {401, 403}:
                message = "NVIDIA NIM 인증에 실패했습니다. API 키를 확인해 주세요."
            elif status_code == 429:
                message = "NVIDIA NIM 호출 한도에 도달했습니다. 잠시 뒤 다시 시도해 주세요."
            else:
                message = f"NVIDIA NIM 요청에 실패했습니다(상태 {status_code})."
            raise NimChatError(message)
        try:
            data = response.json()
            choices = data["choices"]
            answer = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise NimChatError("NVIDIA NIM 응답 형식을 확인할 수 없습니다.") from exc
        cleaned_answer = str(answer or "").strip()
        if not cleaned_answer:
            raise NimChatError("NVIDIA NIM이 빈 답변을 반환했습니다.")
        return cleaned_answer
