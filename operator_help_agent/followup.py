"""回答後の解決ヒアリングと、未解決時のエスカレーション。LLM は呼ばない。"""

from dataclasses import dataclass
from typing import Callable

from operator_help_agent.agent import Approver
from operator_help_agent.models import FinalAnswer, ToolCall
from operator_help_agent.tools import TEAMS, ToolSpec

DEFAULT_TEAM = "スーパーバイザー"


@dataclass(frozen=True)
class FollowUpResult:
    # True: 解決 / False: 未解決 / None: スキップ、または聞いていない
    resolved: bool | None
    escalated: bool = False


def follow_up(
    question: str,
    final: FinalAnswer,
    escalation_tool: ToolSpec,
    approver: Approver,
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    marker: str = "> ",
) -> FollowUpResult:
    if final.kind != "answer":
        return FollowUpResult(resolved=None)

    answer = input_fn("\nこの回答で解決しましたか？ [y/n/Enter=スキップ] ").strip().lower()
    if answer == "y":
        return FollowUpResult(resolved=True)
    if answer != "n":
        return FollowUpResult(resolved=None)

    reason = input_fn("解決しなかった内容を入力してください> ").strip()
    team = _choose_team(input_fn, print_fn, marker)
    call = ToolCall(
        id="followup-1",
        name=escalation_tool.name,
        input={"team": team, "summary": question, "reason": f"一次回答で解決しなかった: {reason}"},
    )
    # G2: 起票の前に承認を求める
    if not approver(call):
        print_fn("起票しませんでした。")
        return FollowUpResult(resolved=False, escalated=False)

    print_fn(escalation_tool.handler(**call.input))
    return FollowUpResult(resolved=False, escalated=True)


def _choose_team(input_fn: Callable[[str], str], print_fn: Callable[..., None], marker: str) -> str:
    options = "  ".join(f"{i}. {team}" for i, team in enumerate(TEAMS, start=1))
    while True:
        print_fn(f"エスカレーション先を選んでください（Enter={DEFAULT_TEAM}）")
        print_fn(f"  {options}")
        choice = input_fn(marker).strip()
        if not choice:
            return DEFAULT_TEAM
        if choice.isdigit() and 1 <= int(choice) <= len(TEAMS):
            return TEAMS[int(choice) - 1]
        print_fn(f"1〜{len(TEAMS)} の番号を入力してください。")
