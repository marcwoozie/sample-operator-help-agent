"""実行中の進み具合を端末に表示する。"""

import json
import sys
import threading
import time
from typing import Callable, Protocol, TextIO

from operator_help_agent.agent import AgentEvent, StepFinished, StepStarted, ToolFinished, ToolStarted
from operator_help_agent.tools import ASK_OPERATOR

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CLEAR_LINE = "\r\033[K"


def render_frame(frame: str, message: str, elapsed: float) -> str:
    return f"{frame} {message} ({elapsed:.1f}s)"


class Spinner:
    """別スレッドで同じ行を書き換え続けるスピナー。端末でないとき（パイプ・テスト）は何も出さない。"""

    def __init__(self, stream: TextIO = sys.stderr, interval: float = 0.1, enabled: bool | None = None):
        self.stream = stream
        self.interval = interval
        self.enabled = stream.isatty() if enabled is None else enabled
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, message: str) -> None:
        if not self.enabled:
            return
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, args=(message,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self.stream.write(CLEAR_LINE)
        self.stream.flush()

    def _spin(self, message: str) -> None:
        started = time.perf_counter()
        i = 0
        while not self._stop_event.is_set():
            frame = FRAMES[i % len(FRAMES)]
            self.stream.write("\r" + render_frame(frame, message, time.perf_counter() - started))
            self.stream.flush()
            i += 1
            self._stop_event.wait(self.interval)


class SpinnerLike(Protocol):
    def start(self, message: str) -> None: ...
    def stop(self) -> None: ...


class ProgressPrinter:
    """Agent の進捗イベントを受け取り、スピナーとツール呼び出しの経過を表示する。"""

    def __init__(self, print_fn: Callable[..., None], spinner: SpinnerLike):
        self.print_fn = print_fn
        self.spinner = spinner

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case StepStarted():
                self.spinner.start("Thinking...")
            case StepFinished():
                self.spinner.stop()
            case ToolStarted(call=call) if call.name != ASK_OPERATOR:
                # 再質問は [確認] として別に表示する
                self.print_fn(f"  → {call.name} {json.dumps(call.input, ensure_ascii=False)}")
            case ToolFinished(call=call, result=result) if result.is_error:
                self.print_fn(f"    ✗ {call.name}: {result.content}")
