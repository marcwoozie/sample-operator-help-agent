import json
from pathlib import Path

import pytest

from evals.grader import CaseRun, EvalCase, grade, load_cases
from operator_help_agent.models import (
    AssistantTurn,
    FinalAnswer,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    UserTurn,
)

CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "cases.jsonl"


def answer_case(**overrides):
    return EvalCase(
        **{
            "id": "A01",
            "question": "日割り返金ある？",
            "expected_kind": "answer",
            "must_cite_any": ["refund.md", "T-1001"],
            "must_include": ["次回請求日"],
            **overrides,
        }
    )


def escalation_case(**overrides):
    return EvalCase(
        **{
            "id": "E01",
            "question": "二重請求",
            "expected_kind": "escalation",
            "expected_team": "請求チーム",
            **overrides,
        }
    )


def run_with(final, history=None, error=None):
    return CaseRun(final=final, history=history or [UserTurn("q")], error=error, seconds=1.0)


def failure_types(result):
    return [f.type for f in result.failures]


# --- 評価セットの読み込み ---


def test_load_cases_reads_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "A01", "question": "q", "expected_kind": "answer"}) + "\n\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases == [EvalCase(id="A01", question="q", expected_kind="answer")]


@pytest.mark.parametrize(
    "lines",
    [
        [{"id": "A01", "question": "q", "expected_kind": "maybe"}],
        [{"id": "A01", "expected_kind": "answer"}],
        [
            {"id": "A01", "question": "q", "expected_kind": "answer"},
            {"id": "A01", "question": "q2", "expected_kind": "answer"},
        ],
    ],
    ids=["invalid_kind", "missing_question", "duplicate_id"],
)
def test_load_cases_rejects_invalid_cases(tmp_path, lines):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(path)


def test_bundled_cases_are_valid():
    cases = load_cases(CASES_PATH)

    assert len(cases) >= 20
    assert {c.expected_kind for c in cases} == {"answer", "escalation"}


# --- 合格 ---


def test_passes_when_answer_meets_all_expectations():
    final = FinalAnswer(kind="answer", message="次回請求日の前日まで見られます", sources=["refund.md"])

    result = grade(answer_case(), run_with(final))

    assert result.passed is True
    assert result.failures == []


def test_passes_escalation_when_team_in_message():
    final = FinalAnswer(kind="escalation", message="請求チームへエスカレーションしてください", sources=[])

    assert grade(escalation_case(), run_with(final)).passed is True


def test_passes_escalation_when_team_in_create_escalation_call():
    call = ToolCall("c1", "create_escalation", {"team": "請求チーム", "summary": "s", "reason": "r"})
    history = [UserTurn("q"), AssistantTurn([call]), ToolResultsTurn([ToolResult("c1", "起票しました")])]
    final = FinalAnswer(kind="escalation", message="起票しました（ESC-0001）", sources=[])

    assert grade(escalation_case(), run_with(final, history)).passed is True


# --- 失敗の分類 ---


def test_error_run_fails_with_error():
    result = grade(answer_case(), run_with(None, error="LLM が応答を拒否しました"))

    assert failure_types(result) == ["error"]
    assert "拒否" in result.failures[0].detail


def test_answer_for_escalation_case_is_missed_escalation():
    final = FinalAnswer(kind="answer", message="返金します", sources=["refund.md"])

    result = grade(escalation_case(), run_with(final))

    # kind が違えば内容（宛先など）は判定しない
    assert failure_types(result) == ["missed_escalation"]


def test_escalation_for_answer_case_is_over_escalation():
    final = FinalAnswer(kind="escalation", message="スーパーバイザーへ", sources=[])

    assert failure_types(grade(answer_case(), run_with(final))) == ["over_escalation"]


def test_wrong_team():
    final = FinalAnswer(kind="escalation", message="技術サポートへ", sources=[])

    assert failure_types(grade(escalation_case(), run_with(final))) == ["wrong_team"]


def test_missing_citation():
    final = FinalAnswer(kind="answer", message="次回請求日まで", sources=["other.md"])

    assert failure_types(grade(answer_case(), run_with(final))) == ["missing_citation"]


def test_citation_matches_partially():
    final = FinalAnswer(kind="answer", message="次回請求日まで", sources=["T-1001（過去事例）"])

    assert grade(answer_case(), run_with(final)).passed is True


def test_missing_content_reports_each_missing_phrase():
    case = answer_case(must_include=["次回請求日", "180日"])
    final = FinalAnswer(kind="answer", message="返金はありません", sources=["refund.md"])

    result = grade(case, run_with(final))

    assert failure_types(result) == ["missing_content", "missing_content"]
    assert [f.detail for f in result.failures] == ["次回請求日", "180日"]


def test_reports_multiple_failure_types_together():
    final = FinalAnswer(kind="answer", message="返金はありません", sources=["other.md"])

    assert failure_types(grade(answer_case(), run_with(final))) == [
        "missing_citation",
        "missing_content",
    ]


# --- 再質問 ---


def history_with_ask():
    call = ToolCall("c1", "ask_operator", {"question": "契約はどこから？"})
    return [UserTurn("q"), AssistantTurn([call]), ToolResultsTurn([ToolResult("c1", "App Store")])]


def test_load_cases_reads_clarification_fields(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "C01",
                "question": "q",
                "expected_kind": "answer",
                "expect_clarification": True,
                "operator_replies": ["App Store"],
            }
        ),
        encoding="utf-8",
    )

    case = load_cases(path)[0]

    assert case.expect_clarification is True
    assert case.operator_replies == ["App Store"]


def test_missed_clarification():
    final = FinalAnswer(kind="answer", message="次回請求日まで", sources=["refund.md"])

    result = grade(answer_case(expect_clarification=True), run_with(final))

    assert failure_types(result) == ["missed_clarification"]


def test_unnecessary_clarification():
    final = FinalAnswer(kind="answer", message="次回請求日まで", sources=["refund.md"])

    result = grade(answer_case(expect_clarification=False), run_with(final, history_with_ask()))

    assert failure_types(result) == ["unnecessary_clarification"]


def test_clarification_is_checked_even_when_kind_differs():
    final = FinalAnswer(kind="escalation", message="スーパーバイザーへ", sources=[])

    result = grade(answer_case(expect_clarification=True), run_with(final))

    assert failure_types(result) == ["over_escalation", "missed_clarification"]


def test_clarification_is_not_checked_when_unspecified():
    final = FinalAnswer(kind="answer", message="次回請求日まで", sources=["refund.md"])

    assert grade(answer_case(), run_with(final, history_with_ask())).passed is True
