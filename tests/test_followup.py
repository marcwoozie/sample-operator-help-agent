import json

import pytest

from operator_help_agent.followup import FollowUpResult, follow_up
from operator_help_agent.models import FinalAnswer
from operator_help_agent.tools import build_tools

ANSWER = FinalAnswer(kind="answer", message="180日間保持されます", sources=["refund.md"])
ESCALATION = FinalAnswer(kind="escalation", message="請求チームへ", sources=[])


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "manuals").mkdir()
    (tmp_path / "past_tickets.json").write_text("[]", encoding="utf-8")
    return tmp_path


@pytest.fixture
def escalation_tool(data_dir):
    return next(t for t in build_tools(data_dir) if t.name == "create_escalation")


def run_follow_up(final, escalation_tool, answers, approve=True):
    answers = list(answers)
    prompts, printed, approvals = [], [], []

    def input_fn(prompt):
        prompts.append(prompt)
        return answers.pop(0)

    def approver(call):
        approvals.append(call)
        return approve

    result = follow_up(
        question="解約したら視聴履歴は消えますか？",
        final=final,
        escalation_tool=escalation_tool,
        approver=approver,
        input_fn=input_fn,
        print_fn=lambda *args: printed.append(" ".join(map(str, args))),
    )
    return result, prompts, "\n".join(printed), approvals


def read_escalations(data_dir):
    path = data_dir / "escalations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_does_not_ask_for_escalation_result(escalation_tool):
    result, prompts, _, _ = run_follow_up(ESCALATION, escalation_tool, answers=[])

    assert result == FollowUpResult(resolved=None)
    assert prompts == []


@pytest.mark.parametrize("answer", ["y", "Y", " y "])
def test_resolved(escalation_tool, data_dir, answer):
    result, prompts, _, _ = run_follow_up(ANSWER, escalation_tool, answers=[answer])

    assert result == FollowUpResult(resolved=True)
    assert "解決しましたか" in prompts[0]
    assert read_escalations(data_dir) == []


@pytest.mark.parametrize("answer", ["", "skip", "x"])
def test_skipped(escalation_tool, data_dir, answer):
    result, _, _, _ = run_follow_up(ANSWER, escalation_tool, answers=[answer])

    assert result == FollowUpResult(resolved=None)
    assert read_escalations(data_dir) == []


def test_unresolved_escalates_to_supervisor_by_default(escalation_tool, data_dir):
    result, _, output, approvals = run_follow_up(
        ANSWER, escalation_tool, answers=["n", "180日を過ぎていると言われた", ""]
    )

    assert result == FollowUpResult(resolved=False, escalated=True)
    assert approvals[0].name == "create_escalation"
    records = read_escalations(data_dir)
    assert len(records) == 1
    assert records[0]["team"] == "スーパーバイザー"
    assert records[0]["summary"] == "解約したら視聴履歴は消えますか？"
    assert records[0]["reason"] == "一次回答で解決しなかった: 180日を過ぎていると言われた"
    assert "ESC-0001" in output


def test_unresolved_escalates_to_selected_team(escalation_tool, data_dir):
    run_follow_up(ANSWER, escalation_tool, answers=["n", "まだ再生できない", "4"])

    assert read_escalations(data_dir)[0]["team"] == "技術サポート"


def test_asks_team_again_on_invalid_number(escalation_tool, data_dir):
    _, _, output, _ = run_follow_up(ANSWER, escalation_tool, answers=["n", "理由", "9", "abc", "1"])

    assert read_escalations(data_dir)[0]["team"] == "請求チーム"
    assert output.count("番号") >= 2


def test_unresolved_without_approval_does_not_escalate(escalation_tool, data_dir):
    result, _, output, _ = run_follow_up(ANSWER, escalation_tool, answers=["n", "理由", ""], approve=False)

    assert result == FollowUpResult(resolved=False, escalated=False)
    assert read_escalations(data_dir) == []
    assert "起票しませんでした" in output


def test_team_selection_uses_given_marker(escalation_tool):
    prompts = []
    answers = ["n", "理由", ""]

    follow_up(
        question="質問",
        final=ANSWER,
        escalation_tool=escalation_tool,
        approver=lambda call: False,
        input_fn=lambda prompt: (prompts.append(prompt), answers.pop(0))[1],
        print_fn=lambda *args: None,
        marker="\033[5m>\033[0m ",
    )

    assert prompts[-1] == "\033[5m>\033[0m "
