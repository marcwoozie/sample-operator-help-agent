"""エージェントループとガードレール。"""

from dataclasses import dataclass
from typing import Callable

from operator_help_agent.drivers.base import LLMDriver
from operator_help_agent.models import (
    AgentResult,
    AssistantTurn,
    DriverStep,
    FinalAnswer,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    Turn,
    Usage,
    UserTurn,
)
from operator_help_agent.prompts import SYSTEM_PROMPT
from operator_help_agent.tools import ToolError, ToolSpec

Approver = Callable[[ToolCall], bool]


# --- 進捗イベント（表示側がループの進み具合を知るためのもの） ---


@dataclass(frozen=True)
class StepStarted:
    """LLM に次の一手を聞き始めた。"""

    step: int


@dataclass(frozen=True)
class StepFinished:
    """LLM から次の一手が返ってきた（失敗した場合も通知する。そのとき usage は None）。"""

    step: int
    usage: Usage | None = None


@dataclass(frozen=True)
class ToolStarted:
    """ツールの実行を始めた（承認が必要なツールは承認後）。"""

    call: ToolCall


@dataclass(frozen=True)
class ToolFinished:
    """ツールの結果が出た（却下・エラーを含む）。"""

    call: ToolCall
    result: ToolResult


AgentEvent = StepStarted | StepFinished | ToolStarted | ToolFinished


class Agent:
    def __init__(
        self,
        driver: LLMDriver,
        tools: list[ToolSpec],
        approver: Approver,
        max_steps: int = 8,
        system: str = SYSTEM_PROMPT,
        on_event: Callable[[AgentEvent], None] = lambda event: None,
    ):
        self.driver = driver
        self.tools = tools
        self.approver = approver
        self.max_steps = max_steps
        self.system = system
        self.on_event = on_event
        self._tools_by_name = {tool.name: tool for tool in tools}

    def run(self, question: str) -> AgentResult:
        history: list[Turn] = [UserTurn(question)]

        # G1: 上限に達したら打ち切る
        for step_no in range(1, self.max_steps + 1):
            self.on_event(StepStarted(step_no))
            driver_step: DriverStep | None = None
            try:
                driver_step = self.driver.next_step(self.system, history, self.tools)
            finally:
                self.on_event(StepFinished(step_no, driver_step.usage if driver_step else None))
            step = driver_step.step

            if isinstance(step, FinalAnswer):
                return AgentResult(final=self._validate(step), history=history)

            results = [self._execute_and_notify(call) for call in step.tool_calls]
            history = [*history, step, ToolResultsTurn(results)]

        return AgentResult(
            final=FinalAnswer(
                kind="escalation",
                message=(
                    f"{self.max_steps} 回の調査で回答をまとめられませんでした。"
                    "スーパーバイザーへエスカレーションしてください。"
                ),
            ),
            history=history,
        )

    def _execute_and_notify(self, call: ToolCall) -> ToolResult:
        result = self._execute(call)
        self.on_event(ToolFinished(call, result))
        return result

    def _execute(self, call: ToolCall) -> ToolResult:
        # G3: どんな失敗も is_error の結果として LLM に返し、ループは止めない
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            return ToolResult(call.id, f"ツール '{call.name}' は存在しません。", is_error=True)

        # G2: 副作用のあるツールはオペレーターの承認を得てから実行する
        if tool.requires_approval and not self.approver(call):
            return ToolResult(call.id, "オペレーターが実行を却下しました。", is_error=True)

        self.on_event(ToolStarted(call))
        try:
            return ToolResult(call.id, tool.handler(**call.input))
        except ToolError as e:
            return ToolResult(call.id, str(e), is_error=True)
        except TypeError as e:
            return ToolResult(call.id, f"ツール '{call.name}' の引数が不正です: {e}", is_error=True)
        except Exception as e:
            return ToolResult(call.id, f"ツールの実行に失敗しました: {e}", is_error=True)

    def _validate(self, final: FinalAnswer) -> FinalAnswer:
        # G4: 根拠のない回答は出さない
        if final.kind == "answer" and not final.sources:
            return FinalAnswer(
                kind="escalation",
                message=(
                    "根拠となる情報を確認できなかったため、エスカレーションが必要です。\n"
                    f"（LLMの回答案: {final.message}）"
                ),
            )
        return final
