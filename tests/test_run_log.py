import json
from datetime import datetime

from operator_help_agent.agent import StepFinished, StepStarted, ToolFinished, ToolStarted
from operator_help_agent.models import ToolCall, ToolResult, Usage
from operator_help_agent.run_log import RunLogger, append_jsonl

TIMESTAMP = datetime(2026, 10, 7, 14, 30, 0)


class FakeClock:
    def __init__(self, *times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


def usage(input_tokens, output_tokens, cost_usd=None, cache_read=0):
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cost_usd=cost_usd,
        model="claude-sonnet-5-5",
    )


def feed_two_steps(logger, first_usage, second_usage):
    search = ToolCall("c1", "search_manuals", {"query": "個人情報を含むかもしれない語"})
    read = ToolCall("c2", "read_manual", {"name": "x.md"})
    logger(StepStarted(1))
    logger(StepFinished(1, first_usage))
    logger(ToolStarted(search))
    logger(ToolFinished(search, ToolResult("c1", "結果")))
    logger(ToolFinished(read, ToolResult("c2", "ありません", is_error=True)))
    logger(StepStarted(2))
    logger(StepFinished(2, second_usage))


def build(logger, **overrides):
    return logger.build_record(
        **{
            "driver": "cli",
            "model": "sonnet",
            "outcome": "answer",
            "duration_seconds": 8.44,
            "timestamp": TIMESTAMP,
            **overrides,
        }
    )


def test_record_contains_per_call_usage_and_totals():
    logger = RunLogger(clock=FakeClock(0.0, 3.0, 3.5, 8.0))
    feed_two_steps(logger, usage(1000, 100, 0.01, cache_read=500), usage(2000, 200, 0.02))

    record = build(logger)

    assert record == {
        "timestamp": "2026-10-07T14:30:00",
        "driver": "cli",
        "model": "sonnet",
        "outcome": "answer",
        "error_type": None,
        "duration_seconds": 8.44,
        "resolved": None,
        "followup_escalated": False,
        "steps": 2,
        "tool_calls": ["search_manuals", "read_manual"],
        "tool_errors": 1,
        "usage": {
            "input_tokens": 3000,
            "output_tokens": 300,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 500,
            "cost_usd": 0.03,
        },
        "llm_calls": [
            {
                "step": 1,
                "model": "claude-sonnet-5-5",
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 500,
                "cost_usd": 0.01,
                "duration_seconds": 3.0,
            },
            {
                "step": 2,
                "model": "claude-sonnet-5-5",
                "input_tokens": 2000,
                "output_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cost_usd": 0.02,
                "duration_seconds": 4.5,
            },
        ],
    }


def test_record_does_not_contain_tool_inputs():
    logger = RunLogger(clock=FakeClock(0.0, 1.0, 2.0, 3.0))
    feed_two_steps(logger, usage(1, 1), usage(1, 1))

    serialized = json.dumps(build(logger), ensure_ascii=False)

    assert "個人情報を含むかもしれない語" not in serialized
    assert "x.md" not in serialized


def test_total_cost_is_none_when_any_call_has_no_cost():
    logger = RunLogger(clock=FakeClock(0.0, 1.0, 2.0, 3.0))
    feed_two_steps(logger, usage(1, 1, cost_usd=0.01), usage(1, 1, cost_usd=None))

    assert build(logger)["usage"]["cost_usd"] is None


def test_failed_step_is_recorded_without_usage():
    logger = RunLogger(clock=FakeClock(0.0, 2.0))
    logger(StepStarted(1))
    logger(StepFinished(1, None))

    record = build(logger, outcome="error", error_type="DriverError")

    assert record["outcome"] == "error"
    assert record["error_type"] == "DriverError"
    assert record["llm_calls"] == [{"step": 1, "duration_seconds": 2.0}]
    assert record["usage"]["input_tokens"] == 0
    assert record["usage"]["cost_usd"] is None


def test_append_jsonl_creates_directory_and_appends_lines(tmp_path):
    path = tmp_path / "logs" / "runs.jsonl"

    append_jsonl(path, {"n": 1, "text": "日本語"})
    append_jsonl(path, {"n": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"n": 1, "text": "日本語"}, {"n": 2}]
    assert "日本語" in lines[0]  # ASCII にエスケープしない


def test_per_call_cost_is_rounded():
    logger = RunLogger(clock=FakeClock(0.0, 1.0))
    logger(StepStarted(1))
    logger(StepFinished(1, usage(1, 1, cost_usd=0.007183800000000001)))

    assert build(logger)["llm_calls"][0]["cost_usd"] == 0.007184


def test_record_contains_follow_up_result():
    logger = RunLogger(clock=FakeClock())

    record = build(logger, resolved=False, followup_escalated=True)

    assert record["resolved"] is False
    assert record["followup_escalated"] is True
