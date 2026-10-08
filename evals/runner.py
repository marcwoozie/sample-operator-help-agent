"""評価セットの実行と集計。"""

import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from evals.grader import CaseRun, EvalCase, Grade, asked_operator, grade
from operator_help_agent.models import AgentResult, AssistantTurn, Turn


class RunnableAgent(Protocol):
    def run(self, question: str) -> AgentResult: ...


# 再質問に答える関数（確認事項を受け取り、答えを返す）
Asker = Callable[[str], str]
# data_dir（ケース専用のコピー）と再質問に答える関数を受け取って Agent を作る関数
AgentFactory = Callable[[Path, Asker], RunnableAgent]


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    run: CaseRun
    grade: Grade


@dataclass(frozen=True)
class Summary:
    total: int
    passed: int
    pass_rate: float
    missed_escalation_rate: float | None
    over_escalation_rate: float | None
    errors: int
    clarification_rate: float
    avg_tool_calls: float
    avg_seconds: float
    failure_counts: dict[str, int]


def make_scripted_replies(replies: list[str]) -> Asker:
    """ケースに書いた答えを順に返す。尽きたら「わかりません」と返す。"""
    remaining = list(replies)

    def ask(question: str) -> str:
        return remaining.pop(0) if remaining else "わかりません"

    return ask


def run_eval(
    cases: list[EvalCase], make_agent: AgentFactory, data_dir: Path, concurrency: int = 4
) -> list[CaseResult]:
    def run_one(case: EvalCase) -> CaseResult:
        run = _run_case(case, make_agent, Path(data_dir))
        return CaseResult(case, run, grade(case, run))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # map は入力の順に結果を返す
        return list(executor.map(run_one, cases))


def _run_case(case: EvalCase, make_agent: AgentFactory, data_dir: Path) -> CaseRun:
    with tempfile.TemporaryDirectory() as tmp:
        # 起票が本物の data/ に書かれないよう、ケースごとにコピーを使う
        case_data_dir = Path(tmp) / "data"
        shutil.copytree(data_dir, case_data_dir, ignore=shutil.ignore_patterns("escalations.jsonl"))
        agent = make_agent(case_data_dir, make_scripted_replies(case.operator_replies))

        started = time.perf_counter()
        try:
            result = agent.run(case.question)
        except Exception as e:
            return CaseRun(final=None, history=[], error=str(e), seconds=time.perf_counter() - started)
        return CaseRun(
            final=result.final, history=result.history, error=None, seconds=time.perf_counter() - started
        )


def summarize(results: list[CaseResult]) -> Summary:
    failure_types = [f.type for r in results for f in r.grade.failures]
    counts = Counter(failure_types)

    def rate(failure_type: str, expected_kind: str) -> float | None:
        targets = sum(1 for r in results if r.case.expected_kind == expected_kind)
        return counts[failure_type] / targets if targets else None

    total = len(results)
    passed = sum(1 for r in results if r.grade.passed)
    return Summary(
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        missed_escalation_rate=rate("missed_escalation", "escalation"),
        over_escalation_rate=rate("over_escalation", "answer"),
        errors=counts["error"],
        clarification_rate=(
            sum(1 for r in results if asked_operator(r.run.history)) / total if total else 0.0
        ),
        avg_tool_calls=sum(len(_tool_calls(r.run.history)) for r in results) / total if total else 0.0,
        avg_seconds=sum(r.run.seconds for r in results) / total if total else 0.0,
        failure_counts=dict(counts),
    )


def to_json(results: list[CaseResult], summary: Summary, meta: dict) -> dict:
    return {
        "meta": meta,
        "summary": asdict(summary),
        "cases": [
            {
                "id": r.case.id,
                "question": r.case.question,
                "expected_kind": r.case.expected_kind,
                "passed": r.grade.passed,
                "failures": [asdict(f) for f in r.grade.failures],
                "final": asdict(r.run.final) if r.run.final else None,
                "error": r.run.error,
                "tool_calls": [{"name": c.name, "input": c.input} for c in _tool_calls(r.run.history)],
                "seconds": round(r.run.seconds, 2),
            }
            for r in results
        ],
    }


def format_report(results: list[CaseResult], summary: Summary) -> str:
    lines = []
    for r in results:
        mark = "✓" if r.grade.passed else "✗"
        actual = r.run.final.kind if r.run.final else "error"
        line = f"{mark} {r.case.id}  {r.case.expected_kind} → {actual}  ({r.run.seconds:.1f}s)"
        if r.grade.failures:
            line += "  " + ", ".join(f"{f.type}: {f.detail}" for f in r.grade.failures)
        lines.append(line)

    lines += [
        "",
        f"合格率: {_percent(summary.pass_rate)} ({summary.passed}/{summary.total})",
        f"エスカレーション見逃し率: {_percent(summary.missed_escalation_rate)}",
        f"過剰エスカレーション率: {_percent(summary.over_escalation_rate)}",
        f"エラー: {summary.errors}",
        f"再質問率: {_percent(summary.clarification_rate)} "
        f"({round(summary.clarification_rate * summary.total)}/{summary.total})",
        f"平均ツール呼び出し回数: {summary.avg_tool_calls:.1f}",
        f"平均所要時間: {summary.avg_seconds:.1f}s",
    ]
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _tool_calls(history: list[Turn]):
    return [call for turn in history if isinstance(turn, AssistantTurn) for call in turn.tool_calls]
