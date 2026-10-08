import pytest

from operator_help_agent.agent import Agent
from operator_help_agent.models import (
    AssistantTurn,
    DriverStep,
    Usage,
    FinalAnswer,
    ToolCall,
    ToolResultsTurn,
    UserTurn,
)
from operator_help_agent.tools import ToolError, ToolSpec

SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


class FakeDriver:
    """決められた次の一手を順に返す。呼ばれたときの引数を記録する。"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def next_step(self, system, history, tools):
        self.calls.append({"system": system, "history": list(history), "tools": tools})
        step = self.steps.pop(0)
        return step if isinstance(step, DriverStep) else DriverStep(step)


def make_tool(name, handler, requires_approval=False):
    return ToolSpec(
        name=name,
        description=f"{name} のテスト用ツール",
        input_schema=SCHEMA,
        handler=handler,
        requires_approval=requires_approval,
    )


@pytest.fixture
def echo_tool():
    received = []

    def handler(**kwargs):
        received.append(kwargs)
        return f"echo: {kwargs}"

    return make_tool("echo", handler), received


def answer(message="回答", sources=("manual.md",)):
    return FinalAnswer(kind="answer", message=message, sources=list(sources))


def calls(*tool_calls):
    return AssistantTurn(tool_calls=list(tool_calls))


def always_approve(call):
    return True


# --- 基本のループ ---


def test_returns_final_answer_without_tools():
    driver = FakeDriver([answer("即答")])
    agent = Agent(driver, tools=[], approver=always_approve)

    result = agent.run("質問")

    assert result.final == answer("即答")
    assert result.history == [UserTurn("質問")]


def test_passes_system_prompt_and_tools_to_driver(echo_tool):
    tool, _ = echo_tool
    driver = FakeDriver([answer()])
    agent = Agent(driver, tools=[tool], approver=always_approve, system="システム")

    agent.run("質問")

    assert driver.calls[0]["system"] == "システム"
    assert driver.calls[0]["tools"] == [tool]


def test_executes_tool_and_feeds_result_back(echo_tool):
    tool, received = echo_tool
    turn = calls(ToolCall(id="c1", name="echo", input={"x": 1}))
    driver = FakeDriver([turn, answer()])
    agent = Agent(driver, tools=[tool], approver=always_approve)

    result = agent.run("質問")

    assert received == [{"x": 1}]
    second_history = driver.calls[1]["history"]
    assert second_history[0] == UserTurn("質問")
    assert second_history[1] is turn
    results_turn = second_history[2]
    assert isinstance(results_turn, ToolResultsTurn)
    assert results_turn.results[0].call_id == "c1"
    assert results_turn.results[0].content == "echo: {'x': 1}"
    assert results_turn.results[0].is_error is False
    assert result.history == second_history


def test_returns_all_results_of_parallel_calls_in_one_turn(echo_tool):
    tool, received = echo_tool
    turn = calls(
        ToolCall(id="c1", name="echo", input={"n": 1}),
        ToolCall(id="c2", name="echo", input={"n": 2}),
    )
    driver = FakeDriver([turn, answer()])
    agent = Agent(driver, tools=[tool], approver=always_approve)

    result = agent.run("質問")

    results_turns = [t for t in result.history if isinstance(t, ToolResultsTurn)]
    assert len(results_turns) == 1
    assert [r.call_id for r in results_turns[0].results] == ["c1", "c2"]


# --- G1: ループ回数の上限 ---


def test_stops_and_escalates_when_max_steps_reached(echo_tool):
    tool, _ = echo_tool
    endless = [calls(ToolCall(id=f"c{i}", name="echo", input={})) for i in range(10)]
    driver = FakeDriver(endless)
    agent = Agent(driver, tools=[tool], approver=always_approve, max_steps=3)

    result = agent.run("質問")

    assert len(driver.calls) == 3
    assert result.final.kind == "escalation"
    assert "スーパーバイザー" in result.final.message


# --- G2: 承認ゲート ---


def test_runs_approval_required_tool_when_approved():
    executed = []
    tool = make_tool("danger", lambda: executed.append(True) or "done", requires_approval=True)
    approvals = []

    def approver(call):
        approvals.append(call)
        return True

    call = ToolCall(id="c1", name="danger", input={})
    driver = FakeDriver([calls(call), answer()])
    agent = Agent(driver, tools=[tool], approver=approver)

    result = agent.run("質問")

    assert approvals == [call]
    assert executed == [True]
    assert result.history[2].results[0].content == "done"


def test_skips_approval_required_tool_when_declined():
    executed = []
    tool = make_tool("danger", lambda: executed.append(True) or "done", requires_approval=True)
    driver = FakeDriver([calls(ToolCall(id="c1", name="danger", input={})), answer()])
    agent = Agent(driver, tools=[tool], approver=lambda call: False)

    result = agent.run("質問")

    assert executed == []
    tool_result = result.history[2].results[0]
    assert tool_result.is_error is True
    assert "却下" in tool_result.content


def test_does_not_ask_approval_for_safe_tools(echo_tool):
    tool, _ = echo_tool

    def approver(call):
        raise AssertionError("承認不要のツールで approver が呼ばれた")

    driver = FakeDriver([calls(ToolCall(id="c1", name="echo", input={})), answer()])
    agent = Agent(driver, tools=[tool], approver=approver)

    agent.run("質問")


# --- G3: ツールのエラーはループを止めずに LLM に返す ---


def _raise(exc):
    def handler(**kwargs):
        raise exc

    return handler


@pytest.mark.parametrize(
    ("tools", "call", "expected"),
    [
        ([], ToolCall(id="c1", name="nope", input={}), "nope"),
        (
            [make_tool("bad", _raise(ToolError("ファイルがありません")))],
            ToolCall(id="c1", name="bad", input={}),
            "ファイルがありません",
        ),
        (
            [make_tool("boom", _raise(RuntimeError("予期しない失敗")))],
            ToolCall(id="c1", name="boom", input={}),
            "予期しない失敗",
        ),
        (
            [make_tool("strict", lambda query: query)],
            ToolCall(id="c1", name="strict", input={"unexpected": 1}),
            "strict",
        ),
    ],
    ids=["unknown_tool", "tool_error", "unexpected_exception", "invalid_arguments"],
)
def test_returns_tool_failures_as_error_results(tools, call, expected):
    driver = FakeDriver([calls(call), answer()])
    agent = Agent(driver, tools=tools, approver=always_approve)

    result = agent.run("質問")

    tool_result = result.history[2].results[0]
    assert tool_result.is_error is True
    assert expected in tool_result.content
    assert result.final == answer()


# --- G4: 根拠のない回答はエスカレーションに変換 ---


def test_converts_answer_without_sources_to_escalation():
    driver = FakeDriver([answer("たぶん返金できます", sources=[])])
    agent = Agent(driver, tools=[], approver=always_approve)

    result = agent.run("質問")

    assert result.final.kind == "escalation"
    assert "たぶん返金できます" in result.final.message
    assert result.final.sources == []


def test_keeps_escalation_without_sources():
    escalation = FinalAnswer(kind="escalation", message="スーパーバイザーへ", sources=[])
    driver = FakeDriver([escalation])
    agent = Agent(driver, tools=[], approver=always_approve)

    assert agent.run("質問").final == escalation


# --- 進捗イベント ---


def test_emits_progress_events_in_order(echo_tool):
    from operator_help_agent.agent import StepFinished, StepStarted, ToolFinished, ToolStarted

    tool, _ = echo_tool
    call = ToolCall(id="c1", name="echo", input={"x": 1})
    driver = FakeDriver([calls(call), answer()])
    events = []
    agent = Agent(driver, tools=[tool], approver=always_approve, on_event=events.append)

    agent.run("質問")

    assert [type(e) for e in events] == [
        StepStarted,
        StepFinished,
        ToolStarted,
        ToolFinished,
        StepStarted,
        StepFinished,
    ]
    assert [e.step for e in events if isinstance(e, StepStarted)] == [1, 2]
    assert events[2].call is call
    assert events[3].result.content == "echo: {'x': 1}"


def test_emits_step_finished_even_when_driver_fails():
    from operator_help_agent.agent import StepFinished
    from operator_help_agent.models import DriverError

    class FailingDriver:
        def next_step(self, system, history, tools):
            raise DriverError("失敗")

    events = []
    agent = Agent(FailingDriver(), tools=[], approver=always_approve, on_event=events.append)

    with pytest.raises(DriverError):
        agent.run("質問")

    assert isinstance(events[-1], StepFinished)


def test_tool_started_is_emitted_after_approval():
    from operator_help_agent.agent import ToolStarted

    order = []
    tool = make_tool("danger", lambda: "done", requires_approval=True)
    driver = FakeDriver([calls(ToolCall(id="c1", name="danger", input={})), answer()])

    def on_event(event):
        if isinstance(event, ToolStarted):
            order.append("started")

    def approver(call):
        order.append("approval")
        return True

    Agent(driver, tools=[tool], approver=approver, on_event=on_event).run("質問")

    assert order == ["approval", "started"]


def test_step_finished_carries_usage():
    from operator_help_agent.agent import StepFinished

    usage = Usage(input_tokens=100, output_tokens=20, cost_usd=0.01, model="sonnet")
    driver = FakeDriver([DriverStep(answer(), usage)])
    events = []

    Agent(driver, tools=[], approver=always_approve, on_event=events.append).run("質問")

    finished = [e for e in events if isinstance(e, StepFinished)]
    assert finished == [StepFinished(step=1, usage=usage)]
