import json

import pytest

from operator_help_agent.drivers.cli import NEXT_STEP_SCHEMA, ClaudeCliDriver, CommandResult
from operator_help_agent.models import (
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
    name="search_manuals",
    description="マニュアルを検索する",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=lambda query: query,
)


class FakeRunner:
    """run_command の呼び出しを記録し、決めた実行結果を返す。"""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, args, input):
        self.calls.append({"args": args, "input": input})
        return self.result


def cli_output(structured_output, **overrides):
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "structured_output": structured_output,
        **overrides,
    }
    return CommandResult(returncode=0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")


FINAL_OUTPUT = cli_output(
    {"action": "final", "final": {"kind": "answer", "message": "回答", "sources": ["a.md"]}}
)


def run(history, result=FINAL_OUTPUT, tools=(SEARCH_TOOL,), system="システム"):
    runner = FakeRunner(result)
    driver = ClaudeCliDriver(run_command=runner)
    driver_step = driver.next_step(system, history, list(tools))
    return driver_step.step, runner.calls[0]


def flag_value(args, flag):
    return args[args.index(flag) + 1]


# --- コマンドの組み立て ---


def test_command_uses_print_mode_with_structured_output():
    _, call = run([UserTurn("質問")])
    args = call["args"]

    assert args[:2] == ["claude", "-p"]
    assert flag_value(args, "--output-format") == "json"
    assert json.loads(flag_value(args, "--json-schema")) == NEXT_STEP_SCHEMA
    assert flag_value(args, "--system-prompt") == "システム"
    assert flag_value(args, "--model") == "sonnet"
    assert "--no-session-persistence" in args


def test_command_disables_builtin_tools_mcp_skills_and_settings():
    _, call = run([UserTurn("質問")])
    args = call["args"]

    assert flag_value(args, "--tools") == ""
    assert flag_value(args, "--setting-sources") == ""
    assert "--strict-mcp-config" in args
    assert "--disable-slash-commands" in args


# --- プロンプトの組み立て ---


def test_prompt_lists_tools_with_schema():
    _, call = run([UserTurn("質問")])
    prompt = call["input"]

    assert "search_manuals" in prompt
    assert "マニュアルを検索する" in prompt
    assert json.dumps(SEARCH_TOOL.input_schema, ensure_ascii=False) in prompt


def test_prompt_contains_question_tool_calls_and_results_in_order():
    history = [
        UserTurn("解約したら返金ある？"),
        AssistantTurn(tool_calls=[ToolCall("cli-1", "search_manuals", {"query": "解約 返金"})]),
        ToolResultsTurn([ToolResult("cli-1", "日割り返金はしない")]),
        AssistantTurn(tool_calls=[ToolCall("cli-2", "read_manual", {"name": "x.md"})]),
        ToolResultsTurn([ToolResult("cli-2", "ファイルがありません", is_error=True)]),
    ]

    _, call = run(history)
    prompt = call["input"]

    expected_in_order = [
        "解約したら返金ある？",
        'search_manuals {"query": "解約 返金"}',
        "日割り返金はしない",
        'read_manual {"name": "x.md"}',
        "ファイルがありません",
    ]
    positions = [prompt.index(text) for text in expected_in_order]
    assert positions == sorted(positions)
    assert "エラー" in prompt[positions[3]:]


# --- 出力の解釈 ---


def test_call_tools_output_becomes_assistant_turn_with_generated_ids():
    result = cli_output(
        {
            "action": "call_tools",
            "tool_calls": [
                {"name": "search_manuals", "input": {"query": "解約"}},
                {"name": "search_past_tickets", "input": {"query": "返金"}},
            ],
        }
    )

    step, _ = run([UserTurn("質問")], result)

    assert step == AssistantTurn(
        tool_calls=[
            ToolCall("cli-1", "search_manuals", {"query": "解約"}),
            ToolCall("cli-2", "search_past_tickets", {"query": "返金"}),
        ]
    )


def test_generated_ids_continue_after_existing_calls():
    history = [
        UserTurn("質問"),
        AssistantTurn(tool_calls=[ToolCall("cli-1", "a", {}), ToolCall("cli-2", "b", {})]),
        ToolResultsTurn([ToolResult("cli-1", "x"), ToolResult("cli-2", "y")]),
    ]
    result = cli_output({"action": "call_tools", "tool_calls": [{"name": "c", "input": {}}]})

    step, _ = run(history, result)

    assert step.tool_calls[0].id == "cli-3"


def test_final_output_becomes_final_answer():
    step, _ = run([UserTurn("質問")])

    assert step == FinalAnswer(kind="answer", message="回答", sources=["a.md"])


# --- G7: 有効な次の一手を得られない場合 ---


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(returncode=1, stdout="", stderr="not logged in"),
        CommandResult(returncode=0, stdout="JSONではない", stderr=""),
        cli_output(None, is_error=True, subtype="error_during_execution"),
        cli_output(None),
        cli_output({"action": "call_tools", "tool_calls": []}),
        cli_output({"action": "final"}),
        cli_output({"action": "final", "final": {"kind": "answer"}}),
    ],
    ids=[
        "nonzero_exit",
        "invalid_json",
        "is_error",
        "missing_structured_output",
        "empty_tool_calls",
        "missing_final",
        "incomplete_final",
    ],
)
def test_invalid_output_raises_driver_error(result):
    with pytest.raises(DriverError):
        run([UserTurn("質問")], result)


# --- トークン量 ---


def test_returns_usage_and_cost_from_cli_output():
    result = cli_output(
        {"action": "final", "final": {"kind": "answer", "message": "回答", "sources": ["a.md"]}},
        usage={
            "input_tokens": 1500,
            "output_tokens": 200,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 3000,
        },
        total_cost_usd=0.0123,
        modelUsage={"claude-sonnet-5-5": {"inputTokens": 1500}},
    )
    driver = ClaudeCliDriver(run_command=FakeRunner(result))

    driver_step = driver.next_step("システム", [UserTurn("質問")], [SEARCH_TOOL])

    assert driver_step.usage == Usage(
        input_tokens=1500,
        output_tokens=200,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=3000,
        cost_usd=0.0123,
        model="claude-sonnet-5-5",
    )


def test_usage_is_none_when_cli_output_has_no_usage():
    driver = ClaudeCliDriver(run_command=FakeRunner(FINAL_OUTPUT))

    assert driver.next_step("システム", [UserTurn("質問")], [SEARCH_TOOL]).usage is None


def test_prompt_tells_llm_not_to_call_tools_directly():
    # ツールを Claude Code のツールとして直接呼ぼうとして失敗するのを防ぐ
    _, call = run([UserTurn("質問")])

    assert "直接呼び出すことはできません" in call["input"]
    assert '"call_tools"' in call["input"]
