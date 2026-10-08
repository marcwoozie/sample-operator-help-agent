"""評価の実行エントリポイント。

uv run python -m evals --driver cli [--model opus] [--only A01 E01]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from evals.grader import load_cases
from evals.runner import AgentFactory, format_report, run_eval, summarize, to_json
from operator_help_agent.__main__ import DEFAULT_DATA_DIR, DRIVERS, create_driver
from operator_help_agent.agent import Agent
from operator_help_agent.tools import build_tools, make_ask_operator_tool

EVALS_DIR = Path(__file__).resolve().parent


def make_agent_factory(driver_name: str, model: str | None) -> tuple[AgentFactory, str]:
    """Agent を作る関数と、実際に使うモデル名を返す。"""
    driver = create_driver(driver_name, model)

    def make_agent(data_dir: Path, ask: Callable[[str], str]) -> Agent:
        # 評価中の承認（G2）はすべて許可する。起票先はケースごとのコピー
        tools = [*build_tools(data_dir), make_ask_operator_tool(ask)]
        return Agent(driver=driver, tools=tools, approver=lambda call: True)

    return make_agent, driver.model


def main(
    argv: list[str] | None = None,
    make_agent_factory: Callable[[str, str | None], tuple[AgentFactory, str]] = make_agent_factory,
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = EVALS_DIR / "results",
    print_fn: Callable[..., None] = print,
) -> int:
    parser = argparse.ArgumentParser(description="評価セットでエージェントを採点する")
    parser.add_argument("--driver", choices=DRIVERS, default="cli")
    parser.add_argument("--model", help="モデル名（省略時は各ドライバーの既定）")
    parser.add_argument("--cases", type=Path, default=EVALS_DIR / "cases.jsonl")
    parser.add_argument("--only", nargs="+", metavar="ID", help="実行するケースID")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.only:
        unknown = sorted(set(args.only) - {c.id for c in cases})
        if unknown:
            parser.error(f"未知のケースIDです: {', '.join(unknown)}")
        cases = [c for c in cases if c.id in args.only]

    make_agent, model = make_agent_factory(args.driver, args.model)
    print_fn(f"{len(cases)} ケースを実行します（driver={args.driver}, model={model}）")

    results = run_eval(cases, make_agent, data_dir, concurrency=args.concurrency)
    summary = summarize(results)
    print_fn(format_report(results, summary))

    results_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = results_dir / f"{started}_{args.driver}_{model}.json"
    meta = {"driver": args.driver, "model": model, "cases": str(args.cases), "started": started}
    path.write_text(
        json.dumps(to_json(results, summary, meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_fn(f"\n結果を保存しました: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
