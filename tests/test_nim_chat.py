from __future__ import annotations

from pathlib import Path

import pytest

from neis_meal_ai.nim_chat import (
    DEFAULT_NIM_MODEL,
    DEFAULT_NIM_URL,
    NimChatError,
    NvidiaNimClient,
    build_grounded_messages,
    load_nvidia_api_key,
)


class FakeResponse:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self.payload


class RecordingSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def _complete_nim_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": DEFAULT_NIM_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "근거를 보면 돈까스입니다.",
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


@pytest.mark.parametrize(
    "contents",
    ["nvapi-test", "NVIDIA_API_KEY=nvapi-test\n"],
)
def test_load_nvidia_api_key_accepts_raw_and_assignment_files(
    tmp_path: Path, contents: str
) -> None:
    key_file = tmp_path / "nvidia_nim.txt"
    key_file.write_text(contents, encoding="utf-8")

    assert load_nvidia_api_key(key_path=key_file, environ={}) == "nvapi-test"


def test_load_nvidia_api_key_prefers_environment_and_never_echoes_bad_value(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "nvidia_nim.txt"
    key_file.write_text("nvapi-file", encoding="utf-8")
    assert (
        load_nvidia_api_key(
            key_path=key_file,
            environ={"NVIDIA_API_KEY": "nvapi-env"},
        )
        == "nvapi-env"
    )

    secret_bad_value = "this-must-never-appear"
    with pytest.raises(NimChatError) as exc_info:
        load_nvidia_api_key(
            key_path=key_file,
            environ={"NVIDIA_API_KEY": secret_bad_value},
        )
    assert secret_bad_value not in str(exc_info.value)


def test_build_grounded_messages_limits_history_and_requires_short_question() -> None:
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"이전 {index}"}
        for index in range(20)
    ]

    messages = build_grounded_messages(
        "TF-IDF 1위는 왜 높아요?",
        "학교=목포가람고등학교, TF=0.2, IDF=1.6931",
        history,
    )

    assert messages[0]["role"] == "system"
    assert "목포가람고등학교" in messages[0]["content"]
    assert len(messages) == 14
    assert messages[-1] == {"role": "user", "content": "TF-IDF 1위는 왜 높아요?"}
    with pytest.raises(NimChatError, match="500자"):
        build_grounded_messages("가" * 501, "근거", [])


def test_nim_client_sends_official_chat_payload_and_returns_answer(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "nvidia_nim.txt"
    key_file.write_text("nvapi-test", encoding="utf-8")
    session = RecordingSession(FakeResponse(_complete_nim_response()))
    client = NvidiaNimClient(key_path=key_file, session=session)

    answer = client.ask(
        "가장 가치가 높은 음식은?",
        "학교=목포가람고등학교, 음식=돈까스, TF-IDF=0.0579",
    )

    assert answer == "근거를 보면 돈까스입니다."
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == DEFAULT_NIM_URL
    assert call["timeout"] == 30.0
    assert call["headers"] == {
        "Authorization": "Bearer nvapi-test",
        "Content-Type": "application/json",
    }
    payload = call["json"]
    assert payload["model"] == DEFAULT_NIM_MODEL
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.7
    assert payload["max_tokens"] == 700
    assert payload["stream"] is False
    assert "TF-IDF=0.0579" in payload["messages"][0]["content"]


def test_nim_client_converts_http_and_malformed_responses_without_secret(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "nvidia_nim.txt"
    secret = "nvapi-secret"
    key_file.write_text(secret, encoding="utf-8")

    for response in (
        FakeResponse({"detail": secret}, status_code=401),
        FakeResponse({"choices": []}),
    ):
        client = NvidiaNimClient(
            key_path=key_file,
            session=RecordingSession(response),
        )
        with pytest.raises(NimChatError) as exc_info:
            client.ask("질문", "근거")
        assert secret not in str(exc_info.value)
