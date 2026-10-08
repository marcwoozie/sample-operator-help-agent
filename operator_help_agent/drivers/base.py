"""LLM ドライバーのインターフェース。"""

from typing import Protocol

from operator_help_agent.models import DriverStep, Turn
from operator_help_agent.tools import ToolSpec


class LLMDriver(Protocol):
    def next_step(
        self, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> DriverStep:
        """履歴を見て、次の一手（ツール呼び出し or 最終回答）とトークン量を返す。

        有効な次の一手を得られない場合は DriverError を送出する。
        """
        ...
