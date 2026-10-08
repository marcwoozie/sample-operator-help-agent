"""Anthropic API ドライバー。ネイティブの Tool Use と structured output を使う。"""

import json

import anthropic

from operator_help_agent.models import (
    FINAL_ANSWER_SCHEMA,
    AssistantTurn,
    DriverError,
    DriverStep,
    FinalAnswer,
    ToolCall,
    ToolResultsTurn,
    Turn,
    Usage,
    UserTurn,
)
from operator_help_agent.tools import ToolSpec

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000
EFFORT = "medium"
# 安全分類器が拒否したとき、サーバー側で別モデルに切り替えて応答させる
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicApiDriver:
    def __init__(self, client=None, model: str = MODEL):
        # client を差し替えられるようにして、テストでは偽クライアントを渡す
        self.client = client if client is not None else anthropic.Anthropic()
        self.model = model

    def next_step(
        self, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> DriverStep:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[_to_message(turn) for turn in history],
            tools=[_to_api_tool(tool) for tool in tools],
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": FINAL_ANSWER_SCHEMA},
            },
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

        return DriverStep(step=_parse_step(response), usage=_parse_usage(response))


def _parse_step(response) -> AssistantTurn | FinalAnswer:
    # G7: 有効な次の一手を得られない場合は DriverError
    match response.stop_reason:
        case "tool_use":
            calls = [
                ToolCall(id=b.id, name=b.name, input=dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ]
            return AssistantTurn(tool_calls=calls, raw=response.content)
        case "end_turn":
            return _parse_final(response.content)
        case "refusal":
            category = getattr(response.stop_details, "category", None)
            raise DriverError(f"LLM が応答を拒否しました（category: {category}）")
        case other:
            raise DriverError(f"想定外の stop_reason です: {other}")


def _parse_usage(response) -> Usage:
    usage = response.usage
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        model=response.model,
    )


def _to_api_tool(tool: ToolSpec) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "strict": True,
    }


def _to_message(turn: Turn) -> dict:
    match turn:
        case UserTurn(text=text):
            return {"role": "user", "content": text}
        case AssistantTurn(raw=raw) if raw is not None:
            # thinking ブロックを含め、レスポンスの content を変更せずに返す
            return {"role": "assistant", "content": raw}
        case AssistantTurn(tool_calls=calls):
            return {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                    for c in calls
                ],
            }
        case ToolResultsTurn(results=results):
            # 並列で呼ばれたツールの結果は、1つの user メッセージにまとめて返す
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in results
                ],
            }
    raise TypeError(f"未知の Turn です: {turn!r}")


def _parse_final(content) -> FinalAnswer:
    text = "".join(b.text for b in content if b.type == "text")
    try:
        data = json.loads(text)
        return FinalAnswer(kind=data["kind"], message=data["message"], sources=list(data["sources"]))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise DriverError(f"最終回答を JSON として解釈できません: {text[:200]!r}") from e
