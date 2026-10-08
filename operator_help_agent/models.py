"""ドライバーに依存しない中立な型。

Agent はこの型で会話履歴を管理し、各ドライバーが自分の形式（Messages API / claude -p のプロンプト）に変換する。
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    tool_calls: list[ToolCall]
    # ドライバー固有の生データ。API ドライバーはレスポンスの content（thinking ブロックを含む）を入れ、
    # 次のリクエストでそのまま返す
    raw: Any = None


@dataclass(frozen=True)
class ToolResultsTurn:
    results: list[ToolResult]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


@dataclass(frozen=True)
class FinalAnswer:
    kind: Literal["answer", "escalation"]
    message: str
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Usage:
    """LLM を1回呼んだときのトークン量と費用。"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # claude -p は費用を返すが、Messages API は返さないので None
    cost_usd: float | None = None
    model: str | None = None


@dataclass(frozen=True)
class DriverStep:
    """ドライバーが返す次の一手と、そのときのトークン量。"""

    step: AssistantTurn | FinalAnswer
    usage: Usage | None = None


@dataclass(frozen=True)
class AgentResult:
    final: FinalAnswer
    history: list[Turn]


# 最終回答の JSON Schema。API ドライバーの structured output と CLI ドライバーの次の一手スキーマで共有する
FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["answer", "escalation"],
            "description": "answer: 回答案 / escalation: 人間へのエスカレーションが必要",
        },
        "message": {
            "type": "string",
            "description": (
                "オペレーター向けの回答案（システムプロンプトの回答案テンプレートに沿う）、"
                "またはエスカレーションが必要な理由と宛先。"
                "全体の見出し（【回答案】など）は付けない"
            ),
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "根拠にしたマニュアルのファイル名や過去チケットID",
        },
    },
    "required": ["kind", "message", "sources"],
    "additionalProperties": False,
}


class DriverError(Exception):
    """LLM から有効な次の一手を得られなかった（refusal, max_tokens, 出力の不正など）。"""
