"""実行ログ（トークン量・費用・所要時間）を組み立てて JSONL に追記する。

質問・回答・ツールの引数などの本文は記録しない（実務で個人情報が混ざることを想定）。
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from operator_help_agent.agent import AgentEvent, StepFinished, StepStarted, ToolFinished
from operator_help_agent.models import Usage

TOKEN_FIELDS = [
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
]


class RunLogger:
    """Agent の進捗イベントを受け取り、LLM 呼び出しごとのトークン量とツール名を集める。"""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self._step_started: dict[int, float] = {}
        self._calls: list[tuple[int, Usage | None, float]] = []
        self._tool_names: list[str] = []
        self._tool_errors = 0

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case StepStarted(step=step):
                self._step_started[step] = self.clock()
            case StepFinished(step=step, usage=usage):
                duration = self.clock() - self._step_started.pop(step)
                self._calls.append((step, usage, duration))
            case ToolFinished(call=call, result=result):
                self._tool_names.append(call.name)
                self._tool_errors += result.is_error

    def build_record(
        self,
        *,
        driver: str,
        model: str | None,
        outcome: str,
        duration_seconds: float,
        timestamp: datetime,
        error_type: str | None = None,
        resolved: bool | None = None,
        followup_escalated: bool = False,
    ) -> dict:
        usages = [usage for _, usage, _ in self._calls if usage is not None]
        costs = [u.cost_usd for u in usages]
        has_all_costs = bool(costs) and len(usages) == len(self._calls) and None not in costs
        return {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "driver": driver,
            "model": model,
            "outcome": outcome,
            "error_type": error_type,
            "duration_seconds": round(duration_seconds, 3),
            "resolved": resolved,
            "followup_escalated": followup_escalated,
            "steps": len(self._calls),
            "tool_calls": list(self._tool_names),
            "tool_errors": self._tool_errors,
            "usage": {
                **{field: sum(getattr(u, field) for u in usages) for field in TOKEN_FIELDS},
                # 1回でも費用がわからない呼び出しがあれば、合計も不明とする
                "cost_usd": round(sum(costs), 6) if has_all_costs else None,
            },
            "llm_calls": [_call_record(step, usage, duration) for step, usage, duration in self._calls],
        }


def _call_record(step: int, usage: Usage | None, duration: float) -> dict:
    record: dict = {"step": step}
    if usage is not None:
        fields = asdict(usage)
        record["model"] = fields.pop("model")
        if fields["cost_usd"] is not None:
            fields["cost_usd"] = round(fields["cost_usd"], 6)
        record.update(fields)
    record["duration_seconds"] = round(duration, 3)
    return record


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
