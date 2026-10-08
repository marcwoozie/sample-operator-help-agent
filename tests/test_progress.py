import io
import time

from operator_help_agent.agent import StepFinished, StepStarted, ToolFinished, ToolStarted
from operator_help_agent.models import ToolCall, ToolResult
from operator_help_agent.progress import CLEAR_LINE, ProgressPrinter, Spinner, render_frame

# --- Spinner ---


def test_render_frame_shows_message_and_elapsed_seconds():
    assert render_frame("⠋", "Thinking...", 3.25) == "⠋ Thinking... (3.2s)"


def test_disabled_spinner_writes_nothing():
    stream = io.StringIO()
    spinner = Spinner(stream=stream, enabled=False)

    spinner.start("考え中")
    spinner.stop()

    assert stream.getvalue() == ""


def test_spinner_is_disabled_when_stream_is_not_a_terminal():
    assert Spinner(stream=io.StringIO()).enabled is False


def test_enabled_spinner_animates_and_clears_line_on_stop():
    stream = io.StringIO()
    spinner = Spinner(stream=stream, interval=0.01, enabled=True)

    spinner.start("考え中")
    time.sleep(0.05)
    spinner.stop()

    output = stream.getvalue()
    assert "考え中" in output
    assert output.count("\r") >= 2  # 同じ行を書き換えている
    assert output.endswith(CLEAR_LINE)


def test_stop_without_start_is_safe():
    Spinner(stream=io.StringIO(), enabled=True).stop()


# --- ProgressPrinter ---


class FakeSpinner:
    def __init__(self):
        self.calls = []

    def start(self, message):
        self.calls.append(("start", message))

    def stop(self):
        self.calls.append(("stop",))


def make_printer():
    spinner = FakeSpinner()
    printed = []
    printer = ProgressPrinter(print_fn=lambda *args: printed.append(" ".join(map(str, args))), spinner=spinner)
    return printer, spinner, printed


def test_spinner_runs_while_waiting_for_llm():
    printer, spinner, _ = make_printer()

    printer(StepStarted(1))
    printer(StepFinished(1))

    assert spinner.calls == [("start", "Thinking..."), ("stop",)]


def test_prints_tool_call_when_started():
    printer, _, printed = make_printer()

    printer(ToolStarted(ToolCall("c1", "search_manuals", {"query": "解約 返金"})))

    assert printed == ['  → search_manuals {"query": "解約 返金"}']


def test_prints_only_failed_tool_results():
    printer, _, printed = make_printer()
    call = ToolCall("c1", "read_manual", {"name": "x.md"})

    printer(ToolFinished(call, ToolResult("c1", "本文")))
    printer(ToolFinished(call, ToolResult("c1", "ファイルがありません", is_error=True)))

    assert printed == ["    ✗ read_manual: ファイルがありません"]


def test_does_not_print_ask_operator_call():
    # 再質問は [確認] として別に表示するので、→ の行は出さない
    printer, _, printed = make_printer()

    printer(ToolStarted(ToolCall("c1", "ask_operator", {"question": "契約はどこから？"})))

    assert printed == []
