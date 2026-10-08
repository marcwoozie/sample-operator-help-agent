"""評価セットの読み込みと、コードによる採点。"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from operator_help_agent.models import AssistantTurn, FinalAnswer, Turn
from operator_help_agent.tools import ASK_OPERATOR

FailureType = Literal[
    "error",
    "missed_escalation",
    "over_escalation",
    "wrong_team",
    "missing_citation",
    "missing_content",
    "missed_clarification",
    "unnecessary_clarification",
]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_kind: Literal["answer", "escalation"]
    expected_team: str | None = None
    must_cite_any: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    # True: 再質問すべき / False: 再質問すべきでない / None: 判定しない
    expect_clarification: bool | None = None
    # 再質問されたときに返す答え（順に返す）
    operator_replies: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class CaseRun:
    """1ケースを実行した結果。失敗した場合は final が None で error に理由が入る。"""

    final: FinalAnswer | None
    history: list[Turn]
    error: str | None
    seconds: float


@dataclass(frozen=True)
class Failure:
    type: FailureType
    detail: str


@dataclass(frozen=True)
class Grade:
    case_id: str
    failures: list[Failure]

    @property
    def passed(self) -> bool:
        return not self.failures


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase(**json.loads(line))
        except TypeError as e:
            raise ValueError(f"{path}:{line_no}: ケースの形式が不正です: {e}") from e
        if case.expected_kind not in ("answer", "escalation"):
            raise ValueError(f"{path}:{line_no}: expected_kind が不正です: {case.expected_kind}")
        cases.append(case)

    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: ケースIDが重複しています: {', '.join(duplicates)}")
    return cases


def grade(case: EvalCase, run: CaseRun) -> Grade:
    if run.final is None:
        return Grade(case.id, [Failure("error", run.error or "不明なエラー")])

    final = run.final
    if final.kind != case.expected_kind:
        # kind が違えば、内容の判定はしない（再質問の判定はする）
        failure_type = "missed_escalation" if case.expected_kind == "escalation" else "over_escalation"
        failures = [Failure(failure_type, f"{case.expected_kind} を期待したが {final.kind}")]
        return Grade(case.id, failures + _check_clarification(case, run.history))

    failures: list[Failure] = []

    if case.expected_team and case.expected_team not in _escalation_targets(final, run.history):
        failures.append(Failure("wrong_team", f"{case.expected_team} が見つからない"))

    if case.must_cite_any and not any(
        expected in source for expected in case.must_cite_any for source in final.sources
    ):
        failures.append(Failure("missing_citation", f"sources={final.sources}"))

    failures += [
        Failure("missing_content", phrase) for phrase in case.must_include if phrase not in final.message
    ]

    return Grade(case.id, failures + _check_clarification(case, run.history))


def asked_operator(history: list[Turn]) -> bool:
    return any(call.name == ASK_OPERATOR for call in _tool_calls(history))


def _check_clarification(case: EvalCase, history: list[Turn]) -> list[Failure]:
    asked = asked_operator(history)
    if case.expect_clarification is True and not asked:
        return [Failure("missed_clarification", "再質問すべきだが、しなかった")]
    if case.expect_clarification is False and asked:
        return [Failure("unnecessary_clarification", "再質問は不要だが、した")]
    return []


def _tool_calls(history: list[Turn]):
    return [call for turn in history if isinstance(turn, AssistantTurn) for call in turn.tool_calls]


def _escalation_targets(final: FinalAnswer, history: list[Turn]) -> str:
    """宛先を探す対象のテキスト。create_escalation で指定した team と、最終回答の message。"""
    teams = [
        str(call.input.get("team", ""))
        for call in _tool_calls(history)
        if call.name == "create_escalation"
    ]
    return "\n".join([*teams, final.message])
