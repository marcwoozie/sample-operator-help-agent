import json
from types import SimpleNamespace

import pytest

from operator_help_agent.drivers.api import AnthropicApiDriver
from operator_help_agent.models import (
    FINAL_ANSWER_SCHEMA,
    AssistantTurn,
    DriverError,
    FinalAnswer,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    Usage,
    UserTurn,
)
from operator_help_agent.tools import ToolSpec

SEARCH_TOOL = ToolSpec(
    name="search",
    description="検索する",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=lambda query: query,
)


class FakeClient:
    """client.beta.messages.create の呼び出しを記録し、決めたレスポンスを返す。"""

    def __init__(self, response):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._response


def block(type, **fields):
    return SimpleNamespace(type=type, **fields)


USAGE = SimpleNamespace(
    input_tokens=1200, output_tokens=300, cache_creation_input_tokens=None, cache_read_input_tokens=800
)


def response(stop_reason, content, stop_details=None, usage=USAGE):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        stop_details=stop_details,
        usage=usage,
        model="claude-sonnet-5-5",
    )


def final_json(kind="answer", message="回答", sources=("a.md",)):
    return json.dumps({"kind": kind, "message": message, "sources": list(sources)}, ensure_ascii=False)


END_TURN = response("end_turn", [block("text", text=final_json())])


def run(history, resp=END_TURN, tools=(SEARCH_TOOL,)):
    client = FakeClient(resp)
    driver = AnthropicApiDriver(client=client)
    driver_step = driver.next_step("システム", history, list(tools))
    return driver_step.step, client.requests[0]


# --- リクエストの組み立て ---


def test_request_contains_model_system_and_output_settings():
    _, request = run([UserTurn("質問")])

    assert request["model"] == "claude-sonnet-5-5"
    assert request["system"] == "システム"
    assert request["max_tokens"] == 16000
    assert request["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": FINAL_ANSWER_SCHEMA},
    }


def test_request_enables_server_side_fallback():
    _, request = run([UserTurn("質問")])

    assert request["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request["betas"]


def test_tools_are_converted_to_strict_api_tools():
    _, request = run([UserTurn("質問")])

    assert request["tools"] == [
        {
            "name": "search",
            "description": "検索する",
            "input_schema": SEARCH_TOOL.input_schema,
            "strict": True,
        }
    ]


def test_history_is_converted_to_messages():
    raw_content = [block("thinking", thinking=""), block("tool_use", id="t1", name="search", input={"query": "q"})]
    history = [
        UserTurn("質問"),
        AssistantTurn(tool_calls=[ToolCall("t1", "search", {"query": "q"})], raw=raw_content),
        ToolResultsTurn(
            [ToolResult("t1", "結果"), ToolResult("t2", "失敗", is_error=True)]
        ),
    ]

    _, request = run(history)

    assert request["messages"] == [
        {"role": "user", "content": "質問"},
        # thinking ブロックを含め、レスポンスの content を変更せずに返す
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "結果", "is_error": False},
                {"type": "tool_result", "tool_use_id": "t2", "content": "失敗", "is_error": True},
            ],
        },
    ]


def test_assistant_turn_without_raw_is_rebuilt_from_tool_calls():
    history = [
        UserTurn("質問"),
        AssistantTurn(tool_calls=[ToolCall("t1", "search", {"query": "q"})]),
        ToolResultsTurn([ToolResult("t1", "結果")]),
    ]

    _, request = run(history)

    assert request["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {"query": "q"}}],
    }


# --- レスポンスの解釈 ---


def test_tool_use_response_becomes_assistant_turn():
    content = [
        block("thinking", thinking=""),
        block("tool_use", id="t1", name="search", input={"query": "解約"}),
        block("tool_use", id="t2", name="search", input={"query": "返金"}),
    ]

    step, _ = run([UserTurn("質問")], response("tool_use", content))

    assert isinstance(step, AssistantTurn)
    assert step.tool_calls == [
        ToolCall("t1", "search", {"query": "解約"}),
        ToolCall("t2", "search", {"query": "返金"}),
    ]
    assert step.raw is content


def test_end_turn_response_becomes_final_answer():
    content = [
        block("thinking", thinking=""),
        block("text", text=final_json("escalation", "請求チームへ", [])),
    ]

    step, _ = run([UserTurn("質問")], response("end_turn", content))

    assert step == FinalAnswer(kind="escalation", message="請求チームへ", sources=[])


# --- G7: 有効な次の一手を得られない場合 ---


def test_refusal_raises_driver_error_with_category():
    resp = response(
        "refusal",
        [],
        stop_details=SimpleNamespace(category="cyber", explanation="..."),
    )

    with pytest.raises(DriverError, match="cyber"):
        run([UserTurn("質問")], resp)


def test_max_tokens_raises_driver_error():
    with pytest.raises(DriverError, match="max_tokens"):
        run([UserTurn("質問")], response("max_tokens", [block("text", text="{")]))


def test_invalid_final_json_raises_driver_error():
    with pytest.raises(DriverError):
        run([UserTurn("質問")], response("end_turn", [block("text", text="JSONではない")]))


# --- トークン量 ---


def test_returns_usage_of_the_response():
    driver = AnthropicApiDriver(client=FakeClient(END_TURN))

    driver_step = driver.next_step("システム", [UserTurn("質問")], [SEARCH_TOOL])

    # キャッシュの値が None のときは 0 として扱う。Messages API は費用を返さない
    assert driver_step.usage == Usage(
        input_tokens=1200,
        output_tokens=300,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=800,
        cost_usd=None,
        model="claude-sonnet-5-5",
    )
