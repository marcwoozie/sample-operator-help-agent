import json
import threading

import pytest

from evals.__main__ import main
from evals.grader import EvalCase
from evals.runner import format_report, make_scripted_replies, run_eval, summarize, to_json
from operator_help_agent.models import (
    AgentResult,
    AssistantTurn,
    DriverError,
    FinalAnswer,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    UserTurn,
)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    (root / "manuals").mkdir(parents=True)
    (root / "manuals" / "refund.md").write_text("# 返金\n", encoding="utf-8")
    (root / "past_tickets.json").write_text("[]", encoding="utf-8")
    return root


class FakeAgent:
    """質問ごとに決めた結果を返す。例外を入れておくと送出する。"""

    def __init__(self, outcomes, seen_dirs=None, data_dir=None):
        self.outcomes = outcomes
        if seen_dirs is not None:
            seen_dirs.append(data_dir)

    def run(self, question):
        outcome = self.outcomes[question]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def result_of(final, tool_calls=0):
    history = [UserTurn("q")]
    for i in range(tool_calls):
        call = ToolCall(f"c{i}", "search_manuals", {"query": "x"})
        history += [AssistantTurn([call]), ToolResultsTurn([ToolResult(call.id, "r")])]
    return AgentResult(final=final, history=history)


ANSWER = FinalAnswer(kind="answer", message="回答", sources=["refund.md"])
ESCALATION = FinalAnswer(kind="escalation", message="請求チームへ", sources=[])

CASES = [
    EvalCase(id="A01", question="qa1", expected_kind="answer"),
    EvalCase(id="A02", question="qa2", expected_kind="answer"),
    EvalCase(id="E01", question="qe1", expected_kind="escalation", expected_team="請求チーム"),
    EvalCase(id="E02", question="qe2", expected_kind="escalation", expected_team="請求チーム"),
]

OUTCOMES = {
    "qa1": result_of(ANSWER, tool_calls=2),
    "qa2": result_of(ESCALATION, tool_calls=1),  # 過剰エスカレーション
    "qe1": result_of(ESCALATION, tool_calls=3),
    "qe2": DriverError("拒否されました"),
}


def make_fake_agent(outcomes=OUTCOMES, seen_dirs=None):
    return lambda data_dir, ask: FakeAgent(outcomes, seen_dirs, data_dir)


# --- 実行 ---


def test_run_eval_grades_each_case_in_order(data_dir):
    results = run_eval(CASES, make_fake_agent(), data_dir, concurrency=4)

    assert [r.case.id for r in results] == ["A01", "A02", "E01", "E02"]
    assert [r.grade.passed for r in results] == [True, False, True, False]
    assert [f.type for f in results[1].grade.failures] == ["over_escalation"]


def test_run_eval_records_errors_without_stopping(data_dir):
    outcomes = {**OUTCOMES, "qa1": RuntimeError("想定外")}

    results = run_eval(CASES, make_fake_agent(outcomes), data_dir)

    assert results[0].run.final is None
    assert "想定外" in results[0].run.error
    assert results[3].run.error == "拒否されました"
    assert results[2].grade.passed is True


def test_run_eval_gives_each_case_a_copy_of_data_dir(data_dir):
    seen_dirs = []
    copied = []

    def make_agent(case_data_dir, ask):
        # コピーはケースの実行後に削除されるので、Agent を作る時点で確認する
        copied.append((case_data_dir / "manuals" / "refund.md").exists())
        return FakeAgent(OUTCOMES, seen_dirs, case_data_dir)

    run_eval(CASES, make_agent, data_dir)

    assert len(set(seen_dirs)) == len(CASES)
    assert data_dir not in seen_dirs
    assert all(copied)
    assert not any(d.exists() for d in seen_dirs)


def test_run_eval_runs_cases_concurrently(data_dir):
    barrier = threading.Barrier(2, timeout=5)

    class WaitingAgent:
        def run(self, question):
            # 2ケースが同時に走っていないと、ここでタイムアウトする
            barrier.wait()
            return result_of(ANSWER)

    cases = CASES[:2]
    results = run_eval(cases, lambda data_dir, ask: WaitingAgent(), data_dir, concurrency=2)

    assert all(r.grade.passed for r in results)


# --- 集計 ---


def test_summarize_computes_metrics(data_dir):
    summary = summarize(run_eval(CASES, make_fake_agent(), data_dir))

    assert summary.total == 4
    assert summary.passed == 2
    assert summary.pass_rate == 0.5
    assert summary.missed_escalation_rate == 0.0  # E01 は正解、E02 はエラー
    assert summary.over_escalation_rate == 0.5  # A02
    assert summary.errors == 1
    assert summary.avg_tool_calls == pytest.approx((2 + 1 + 3 + 0) / 4)
    assert summary.failure_counts == {"over_escalation": 1, "error": 1}


def test_summarize_rates_are_none_without_target_cases(data_dir):
    summary = summarize(run_eval(CASES[:1], make_fake_agent(), data_dir))

    assert summary.missed_escalation_rate is None
    assert summary.over_escalation_rate == 0.0


# --- 出力 ---


def test_to_json_contains_case_details_and_summary(data_dir):
    results = run_eval(CASES, make_fake_agent(), data_dir)

    data = to_json(results, summarize(results), meta={"driver": "cli", "model": "sonnet"})

    assert data["meta"] == {"driver": "cli", "model": "sonnet"}
    assert data["summary"]["pass_rate"] == 0.5
    first = data["cases"][0]
    assert first["id"] == "A01"
    assert first["question"] == "qa1"
    assert first["passed"] is True
    assert first["final"] == {"kind": "answer", "message": "回答", "sources": ["refund.md"]}
    assert first["tool_calls"] == [{"name": "search_manuals", "input": {"query": "x"}}] * 2
    assert data["cases"][1]["failures"] == [{"type": "over_escalation", "detail": "answer を期待したが escalation"}]
    json.dumps(data, ensure_ascii=False)  # そのまま保存できる


def test_format_report_shows_each_case_and_metrics(data_dir):
    results = run_eval(CASES, make_fake_agent(), data_dir)

    report = format_report(results, summarize(results))

    assert "✓ A01" in report
    assert "✗ A02" in report
    assert "over_escalation" in report
    assert "合格率: 50.0% (2/4)" in report
    assert "過剰エスカレーション率: 50.0%" in report


# --- エントリポイント ---


def test_main_runs_selected_cases_and_saves_results(tmp_path, data_dir):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in CASES), encoding="utf-8"
    )
    results_dir = tmp_path / "results"
    output = []
    received = []

    def make_agent_factory(driver, model):
        received.append((driver, model))
        return make_fake_agent(), "sonnet"

    code = main(
        ["--driver", "cli", "--cases", str(cases_path), "--only", "A01", "E01"],
        make_agent_factory=make_agent_factory,
        data_dir=data_dir,
        results_dir=results_dir,
        print_fn=output.append,
    )

    assert code == 0
    assert received == [("cli", None)]
    saved = list(results_dir.glob("*_cli_sonnet.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text(encoding="utf-8"))
    assert [c["id"] for c in data["cases"]] == ["A01", "E01"]
    assert "合格率: 100.0% (2/2)" in "\n".join(output)


def test_main_rejects_unknown_case_ids(tmp_path, data_dir):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(CASES[0].__dict__), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            ["--cases", str(cases_path), "--only", "Z99"],
            make_agent_factory=lambda driver, model: (make_fake_agent(), "m"),
            data_dir=data_dir,
            results_dir=tmp_path / "results",
            print_fn=lambda *args: None,
        )


# --- 再質問 ---


def test_scripted_replies_are_returned_in_order_then_unknown():
    ask = make_scripted_replies(["App Store", "E-201"])

    assert [ask("q1"), ask("q2"), ask("q3")] == ["App Store", "E-201", "わかりません"]


def test_run_eval_answers_clarification_with_case_replies(data_dir):
    replies = []

    class AskingAgent:
        def __init__(self, ask):
            self.ask = ask

        def run(self, question):
            replies.append(self.ask("契約はどこから？"))
            return result_of(ANSWER)

    cases = [EvalCase(id="C01", question="q", expected_kind="answer", operator_replies=["App Store"])]

    run_eval(cases, lambda data_dir, ask: AskingAgent(ask), data_dir)

    assert replies == ["App Store"]


def test_summarize_computes_clarification_rate(data_dir):
    call = ToolCall("c1", "ask_operator", {"question": "契約は？"})
    asked = AgentResult(
        final=ANSWER,
        history=[UserTurn("q"), AssistantTurn([call]), ToolResultsTurn([ToolResult("c1", "App Store")])],
    )
    outcomes = {**OUTCOMES, "qa1": asked}

    results = run_eval(CASES, make_fake_agent(outcomes), data_dir)
    summary = summarize(results)

    assert summary.clarification_rate == 0.25
    assert "再質問率: 25.0% (1/4)" in format_report(results, summary)
