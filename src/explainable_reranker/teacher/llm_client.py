from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatModel(Protocol):
    """External chat-completion boundary used by the grounded LLM teacher.

    This is the single seam where the pipeline touches a real LLM provider. The
    rest of the teacher stack only depends on this protocol, so offline tests can
    inject :class:`ScriptedChatModel` while production injects a real client —
    :class:`AnthropicClaudeChatModel` (first-party Opus, the default used by
    ``scripts/collect_and_label.py``) or :class:`BedrockClaudeChatModel` for
    environments with Bedrock access instead.
    """

    def generate(self, *, system: str, user: str) -> str:
        """Return the raw assistant text for a single-turn completion."""


@dataclass
class ScriptedChatModel:
    """Deterministic dummy chat model used in tests and offline smoke runs.

    It stands in for the real Claude call by replaying queued responses. The
    `responses` argument may be:

    - a single ``str`` returned for every call,
    - a ``Sequence[str]`` consumed in order (raises once exhausted), or
    - a callable ``(system, user) -> str`` that synthesizes the response.

    Every prompt pair is recorded in :attr:`calls` so tests can assert on the
    self-consistency shuffling and 2-pass prompt wiring.
    """

    responses: "str | Sequence[str] | Callable[[str, str], str]"
    calls: list[tuple[str, str]] = field(default_factory=list)
    _cursor: int = 0

    def generate(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self.responses):
            return self.responses(system, user)
        if isinstance(self.responses, str):
            return self.responses
        if self._cursor >= len(self.responses):
            raise RuntimeError("ScriptedChatModel ran out of queued responses")
        response = self.responses[self._cursor]
        self._cursor += 1
        return response


class BedrockClaudeChatModel:
    """Real adapter skeleton for topa's Bedrock Claude (Opus 4.8).

    Credentials and the ``boto3`` dependency are resolved lazily on first call so
    importing the package stays dependency-free and unit-testable. The request
    body follows the Anthropic Messages API as exposed through
    ``bedrock-runtime.invoke_model``. This class is intentionally not exercised by
    the offline test-suite; it is the production wiring point.
    """

    DEFAULT_MODEL_ID = "anthropic.claude-opus-4-8"
    ANTHROPIC_VERSION = "bedrock-2023-05-31"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        region: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        client: object | None = None,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client

    def _bedrock_client(self) -> object:
        if self._client is None:
            try:
                import boto3  # imported lazily; only needed in the training env
            except ImportError as exc:  # pragma: no cover - exercised only in prod
                raise RuntimeError(
                    "boto3 is required for BedrockClaudeChatModel; install it in the "
                    "training/serving environment that has Bedrock access."
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _request_body(self, *, system: str, user: str) -> dict:
        return {
            "anthropic_version": self.ANTHROPIC_VERSION,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }

    @staticmethod
    def _extract_text(response_body: dict) -> str:
        blocks = response_body.get("content", [])
        return "".join(block.get("text", "") for block in blocks if isinstance(block, dict))

    def generate(self, *, system: str, user: str) -> str:  # pragma: no cover - prod path
        client = self._bedrock_client()
        response = client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(self._request_body(system=system, user=user)),
        )
        payload = response["body"].read()
        return self._extract_text(json.loads(payload))


class AnthropicClaudeChatModel:
    """Production teacher backed by the first-party Anthropic API (Opus 4.8).

    This is the "best case" label generator: it calls Claude Opus 4.8 directly
    via the official SDK. The SDK and credentials (``ANTHROPIC_API_KEY``) are
    resolved lazily on first call so importing the package stays dependency-free.

    Labeling emits a full listwise ranking plus grounded rationales, which can be
    long, so we stream and pull the final message (timeout-safe per the SDK
    guidance). Adaptive thinking is on — the teacher is doing real reasoning over
    many candidates — and only the text blocks are returned for JSON extraction.
    """

    DEFAULT_MODEL_ID = "claude-opus-4-8"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        max_tokens: int = 32000,
        effort: str = "high",
        client: object | None = None,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = client

    def _anthropic_client(self) -> object:
        if self._client is None:
            try:
                import anthropic  # lazy; only needed where labels are generated
            except ImportError as exc:  # pragma: no cover - exercised only in prod
                raise RuntimeError(
                    "anthropic SDK is required for AnthropicClaudeChatModel; install the "
                    "teacher extras: `pip install -e '.[teacher]'`."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, *, system: str, user: str) -> str:  # pragma: no cover - prod path
        client = self._anthropic_client()
        with client.messages.stream(
            model=self.model_id,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        return "".join(block.text for block in message.content if block.type == "text")


def extract_json_object(text: str) -> dict:
    """Parse the first top-level JSON object out of an LLM response.

    Real models often wrap JSON in prose or ```json fences, so we slice from the
    first ``{`` to the last ``}`` before parsing. Raises ``ValueError`` when no
    object is present, which the grounded teacher uses to trigger a retry.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in chat-model response")
    return json.loads(text[start : end + 1])
