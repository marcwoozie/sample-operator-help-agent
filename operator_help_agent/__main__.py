"""CLI エントリポイント。

uv run python -m operator_help_agent --driver api|cli "質問"
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from operator_help_agent.agent import Agent, Approver
from operator_help_agent.followup import FollowUpResult, follow_up
from operator_help_agent.drivers.base import LLMDriver
from operator_help_agent.models import DriverError, ToolCall
from operator_help_agent.progress import ProgressPrinter, Spinner
from operator_help_agent.run_log import RunLogger, append_jsonl
from operator_help_agent.tools import build_tools, make_ask_operator_tool

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_LOG_FILE = PROJECT_DIR / "logs" / "runs.jsonl"
DRIVERS = ["api", "cli"]
LABELS = {"answer": "回答案", "escalation": "エスカレーションが必要"}
# LLM が回答文の先頭に付けてしまう見出し（[回答案] や 【回答案】 など）
DUPLICATED_HEADING = re.compile(r"^\s*[\[【]\s*(?:" + "|".join(LABELS.values()) + r")\s*[\]】]\s*")


def strip_heading(message: str) -> str:
    """回答文の先頭にある見出しを取り除く。CLI 側でも同じ見出しを表示するため。"""
    return DUPLICATED_HEADING.sub("", message, count=1)


def create_driver(name: str, model: str | None = None) -> LLMDriver:
    # model を省略したら各ドライバーの既定モデルを使う
    options = {"model": model} if model else {}
    match name:
        case "api":
            from operator_help_agent.drivers.api import AnthropicApiDriver

            return AnthropicApiDriver(**options)
        case "cli":
            from operator_help_agent.drivers.cli import ClaudeCliDriver

            return ClaudeCliDriver(**options)
    raise ValueError(f"未知のドライバーです: {name}（指定できるもの: {', '.join(DRIVERS)}）")


def make_approver(input_fn: Callable[[str], str], print_fn: Callable[..., None]) -> Approver:
    def approve(call: ToolCall) -> bool:
        print_fn(f"\n[承認が必要] {call.name}")
        print_fn(json.dumps(call.input, ensure_ascii=False, indent=2))
        return input_fn("実行しますか？ [y/N] ").strip().lower() == "y"

    return approve


def input_marker(blink: bool) -> str:
    """入力待ちの「>」。端末では点滅させる（ANSI の点滅。対応していない端末では普通に表示される）。"""
    return "\033[5m>\033[0m " if blink else "> "


def make_asker(
    input_fn: Callable[[str], str], print_fn: Callable[..., None], marker: str = "> "
) -> Callable[[str], str]:
    def ask(question: str) -> str:
        print_fn(f"\n[確認] {question}")
        return input_fn(marker)

    return ask


def main(
    argv: list[str] | None = None,
    driver_factory: Callable[[str, str | None], LLMDriver] = create_driver,
    data_dir: Path = DEFAULT_DATA_DIR,
    log_file: Path = DEFAULT_LOG_FILE,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
    blink: bool | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="オペレーター向け社内ヘルプエージェント（サンプル）")
    parser.add_argument("question", nargs="?", help="質問（省略すると入力を求める）")
    parser.add_argument("--driver", choices=DRIVERS, default="api", help="LLM の呼び出し方式")
    parser.add_argument("--model", help="モデル名（省略時: api は claude-sonnet-5-5、cli は sonnet）")
    parser.add_argument("--max-steps", type=int, default=8, help="ループ回数の上限")
    parser.add_argument("--log-file", type=Path, help=f"実行ログの追記先（省略時: {DEFAULT_LOG_FILE}）")
    args = parser.parse_args(argv)
    log_file = args.log_file or log_file

    marker = input_marker(sys.stdout.isatty() if blink is None else blink)
    question = args.question or input_fn(marker)
    driver = driver_factory(args.driver, args.model)
    tools = build_tools(data_dir)
    approver = make_approver(input_fn, print_fn)
    spinner = Spinner()
    progress = ProgressPrinter(print_fn, spinner)
    logger = RunLogger()

    def on_event(event):
        progress(event)
        logger(event)

    agent = Agent(
        driver=driver,
        tools=[*tools, make_ask_operator_tool(make_asker(input_fn, print_fn, marker))],
        approver=approver,
        max_steps=args.max_steps,
        on_event=on_event,
    )

    def write_log(
        outcome: str,
        duration: float,
        error_type: str | None = None,
        follow_up_result: FollowUpResult = FollowUpResult(resolved=None),
    ) -> None:
        record = logger.build_record(
            driver=args.driver,
            model=getattr(driver, "model", None),
            outcome=outcome,
            error_type=error_type,
            duration_seconds=duration,
            timestamp=datetime.now(),
            resolved=follow_up_result.resolved,
            followup_escalated=follow_up_result.escalated,
        )
        append_jsonl(log_file, record)

    started = time.perf_counter()
    try:
        result = agent.run(question)
    except DriverError as e:
        spinner.stop()
        write_log("error", time.perf_counter() - started, error_type=type(e).__name__)
        print_fn(f"エラー: {e}")
        return 1
    finally:
        spinner.stop()
    elapsed = time.perf_counter() - started

    final = result.final
    print_fn(f"\n[{LABELS[final.kind]}]")
    print_fn(strip_heading(final.message))
    if final.sources:
        print_fn(f"\n根拠: {', '.join(final.sources)}")
    print_fn(f"（所要時間: {elapsed:.1f}秒）")

    escalation_tool = next(t for t in tools if t.name == "create_escalation")
    follow_up_result = follow_up(question, final, escalation_tool, approver, input_fn, print_fn, marker)
    write_log(final.kind, elapsed, follow_up_result=follow_up_result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
