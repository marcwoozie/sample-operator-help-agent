"""実際の LLM を呼ぶ疎通確認。`uv run pytest -m live` で実行する（課金・利用枠を消費）。"""

import os
import shutil
from pathlib import Path

import pytest

from operator_help_agent.__main__ import create_driver
from operator_help_agent.agent import Agent
from operator_help_agent.tools import build_tools

pytestmark = pytest.mark.live

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DRIVER_PARAMS = [
    pytest.param(
        "api",
        marks=pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY 未設定"),
    ),
    pytest.param(
        "cli",
        marks=pytest.mark.skipif(shutil.which("claude") is None, reason="claude コマンドがない"),
    ),
]


@pytest.fixture
def data_dir(tmp_path):
    # 起票ファイルが本物の data/ に書かれないよう、コピーを使う
    shutil.copytree(DATA_DIR, tmp_path / "data", ignore=shutil.ignore_patterns("escalations.jsonl"))
    return tmp_path / "data"


def make_agent(driver_name, data_dir):
    return Agent(
        driver=create_driver(driver_name),
        tools=build_tools(data_dir),
        approver=lambda call: False,
    )


@pytest.mark.parametrize("driver_name", DRIVER_PARAMS)
def test_answers_from_manual(driver_name, data_dir):
    result = make_agent(driver_name, data_dir).run("月の途中で解約したら日割りで返金されますか？")

    assert result.final.kind == "answer"
    assert any("cancellation_and_refund.md" in s or s.startswith("T-") for s in result.final.sources)


@pytest.mark.parametrize("driver_name", DRIVER_PARAMS)
def test_escalates_refund_request(driver_name, data_dir):
    result = make_agent(driver_name, data_dir).run("今月2回引き落とされているので返金してほしいと言われました")

    assert result.final.kind == "escalation"
