import json

import pytest

from operator_help_agent.__main__ import create_driver, main, make_approver
from operator_help_agent.drivers.api import AnthropicApiDriver
from operator_help_agent.drivers.cli import ClaudeCliDriver
from operator_help_agent.models import AssistantTurn, DriverError, DriverStep, FinalAnswer, ToolCall


class FakeDriver:
    def __init__(self, steps):
        self.steps = list(steps)

    def next_step(self, system, history, tools):
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step if isinstance(step, DriverStep) else DriverStep(step)


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "manuals").mkdir()
    (tmp_path / "manuals" / "refund.md").write_text("# 返金\n\n## 解約後\n日割り返金はしない。\n", encoding="utf-8")
    (tmp_path / "past_tickets.json").write_text("[]", encoding="utf-8")
    return tmp_path


def run_main(argv, driver, data_dir, answers=()):
    answers = list(answers)
    output = []
    code = main(
        argv,
        driver_factory=lambda name, model: driver,
        data_dir=data_dir,
        log_file=data_dir / "logs" / "runs.jsonl",
        # 用意した入力が尽きたら Enter（空文字）を返す
        input_fn=lambda prompt: (output.append(prompt), answers.pop(0) if answers else "")[1],
        print_fn=lambda *args: output.append(" ".join(str(a) for a in args)),
    )
    return code, "\n".join(output)


# --- ドライバーの切り替え ---


def test_create_driver_cli():
    assert isinstance(create_driver("cli"), ClaudeCliDriver)


def test_create_driver_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-test")
    assert isinstance(create_driver("api"), AnthropicApiDriver)


def test_create_driver_rejects_unknown_name():
    with pytest.raises(ValueError):
        create_driver("openai")


def test_create_driver_uses_given_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-test")
    assert create_driver("cli", model="opus").model == "opus"
    assert create_driver("api", model="claude-opus-5-5").model == "claude-opus-5-5"


def test_create_driver_keeps_driver_default_model_when_not_given():
    assert create_driver("cli").model == "sonnet"


def test_main_passes_driver_and_model_to_factory(data_dir):
    received = []

    def factory(name, model):
        received.append((name, model))
        return FakeDriver([FinalAnswer(kind="answer", message="回答", sources=["refund.md"])])

    main(
        ["--driver", "cli", "--model", "opus", "質問"],
        driver_factory=factory,
        data_dir=data_dir,
        log_file=data_dir / "logs" / "runs.jsonl",
        input_fn=lambda prompt: "",
        print_fn=lambda *args: None,
    )

    assert received == [("cli", "opus")]


# --- 承認 ---


@pytest.mark.parametrize(("answer", "expected"), [("y", True), ("Y", True), ("n", False), ("", False)])
def test_approver_accepts_only_yes(answer, expected):
    shown = []
    approver = make_approver(
        input_fn=lambda prompt: answer, print_fn=lambda *args: shown.append(" ".join(map(str, args)))
    )

    approved = approver(ToolCall("c1", "create_escalation", {"team": "請求チーム"}))

    assert approved is expected
    # 何を実行しようとしているかをオペレーターに見せる
    assert "create_escalation" in "\n".join(shown)
    assert "請求チーム" in "\n".join(shown)


# --- エンドツーエンド（偽ドライバー） ---


def test_main_prints_answer_and_sources(data_dir):
    driver = FakeDriver(
        [
            AssistantTurn([ToolCall("c1", "search_manuals", {"query": "日割り"})]),
            FinalAnswer(kind="answer", message="日割り返金はありません。", sources=["refund.md"]),
        ]
    )

    code, output = run_main(["--driver", "cli", "日割り返金ある？"], driver, data_dir)

    assert code == 0
    assert "日割り返金はありません。" in output
    assert "refund.md" in output
    # ツールの呼び出しをその場で表示し、最後に所要時間を出す
    assert '→ search_manuals {"query": "日割り"}' in output
    assert "[調査の経過]" not in output
    assert "所要時間" in output


def test_main_reads_question_from_input_when_not_given(data_dir):
    driver = FakeDriver([FinalAnswer(kind="answer", message="回答", sources=["refund.md"])])

    code, output = run_main(["--driver", "cli"], driver, data_dir, answers=["質問です"])

    assert code == 0
    assert "回答" in output


def test_main_asks_approval_before_escalation(data_dir):
    call = ToolCall("c1", "create_escalation", {"team": "請求チーム", "summary": "s", "reason": "r"})
    driver = FakeDriver(
        [
            AssistantTurn([call]),
            FinalAnswer(kind="escalation", message="請求チームへ", sources=[]),
        ]
    )

    code, output = run_main(["--driver", "cli", "二重請求された"], driver, data_dir, answers=["y"])

    assert code == 0
    assert "エスカレーション" in output
    record = json.loads((data_dir / "escalations.jsonl").read_text(encoding="utf-8"))
    assert record["team"] == "請求チーム"


def test_main_returns_error_code_on_driver_error(data_dir):
    driver = FakeDriver([DriverError("LLM が応答を拒否しました")])

    code, output = run_main(["--driver", "cli", "質問"], driver, data_dir)

    assert code == 1
    assert "LLM が応答を拒否しました" in output


# --- 実行ログ ---


def read_log(data_dir):
    path = data_dir / "logs" / "runs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_main_appends_usage_log_without_text(data_dir):
    from operator_help_agent.models import Usage

    usage = Usage(input_tokens=1000, output_tokens=100, cost_usd=0.01, model="claude-sonnet-5-5")
    driver = FakeDriver(
        [
            DriverStep(AssistantTurn([ToolCall("c1", "search_manuals", {"query": "日割り"})]), usage),
            DriverStep(
                FinalAnswer(kind="answer", message="日割り返金はありません。", sources=["refund.md"]), usage
            ),
        ]
    )

    run_main(["--driver", "cli", "日割り返金ある？"], driver, data_dir)
    run_main(["--driver", "cli", "日割り返金ある？"], FakeDriver([FinalAnswer("answer", "a", ["refund.md"])]), data_dir)

    records = read_log(data_dir)
    assert len(records) == 2
    first = records[0]
    assert first["driver"] == "cli"
    assert first["outcome"] == "answer"
    assert first["tool_calls"] == ["search_manuals"]
    assert first["usage"]["input_tokens"] == 2000
    assert first["usage"]["cost_usd"] == 0.02
    serialized = json.dumps(first, ensure_ascii=False)
    assert "日割り" not in serialized


def test_main_logs_driver_error(data_dir):
    driver = FakeDriver([DriverError("LLM の出力: 個人情報を含むかもしれない文")])

    code, _ = run_main(["--driver", "cli", "質問"], driver, data_dir)

    assert code == 1
    record = read_log(data_dir)[0]
    assert record["outcome"] == "error"
    assert record["error_type"] == "DriverError"
    assert "個人情報" not in json.dumps(record, ensure_ascii=False)


# --- 再質問 ---


def test_main_asks_operator_and_feeds_reply_back(data_dir):
    histories = []

    class RecordingDriver(FakeDriver):
        def next_step(self, system, history, tools):
            histories.append(list(history))
            return super().next_step(system, history, tools)

    call = ToolCall("c1", "ask_operator", {"question": "契約はWebとApp Storeのどちらからですか？"})
    driver = RecordingDriver(
        [
            AssistantTurn([call]),
            FinalAnswer(kind="answer", message="ストアへ案内してください", sources=["refund.md"]),
        ]
    )

    code, output = run_main(["--driver", "cli", "返金してほしいと言われた"], driver, data_dir, answers=["App Store"])

    assert code == 0
    assert "[確認] 契約はWebとApp Storeのどちらからですか？" in output
    assert "→ ask_operator" not in output
    tool_result = histories[1][-1].results[0]
    assert tool_result.is_error is False
    assert "App Store" in tool_result.content


# --- 解決ヒアリング ---


def test_main_logs_resolved_answer(data_dir):
    driver = FakeDriver([FinalAnswer(kind="answer", message="回答", sources=["refund.md"])])

    code, output = run_main(["--driver", "cli", "質問"], driver, data_dir, answers=["y"])

    assert code == 0
    assert "解決しましたか" in output
    record = read_log(data_dir)[0]
    assert record["resolved"] is True
    assert record["followup_escalated"] is False


def test_main_escalates_unresolved_answer(data_dir):
    driver = FakeDriver([FinalAnswer(kind="answer", message="回答", sources=["refund.md"])])

    # 解決しなかった → 内容 → 宛先（Enter=スーパーバイザー）→ 承認
    run_main(["--driver", "cli", "質問"], driver, data_dir, answers=["n", "まだ見られない", "", "y"])

    escalation = json.loads((data_dir / "escalations.jsonl").read_text(encoding="utf-8"))
    assert escalation["team"] == "スーパーバイザー"
    record = read_log(data_dir)[0]
    assert record["resolved"] is False
    assert record["followup_escalated"] is True
    assert "まだ見られない" not in json.dumps(record, ensure_ascii=False)


def test_main_does_not_ask_resolution_for_escalation(data_dir):
    driver = FakeDriver([FinalAnswer(kind="escalation", message="請求チームへ", sources=[])])

    _, output = run_main(["--driver", "cli", "質問"], driver, data_dir)

    assert "解決しましたか" not in output
    assert read_log(data_dir)[0]["resolved"] is None


# --- 見出しの重複 ---


@pytest.mark.parametrize(
    "message",
    [
        "【回答案】日割り返金はありません。",
        "[回答案]\n日割り返金はありません。",
        "  【 回答案 】\n\n日割り返金はありません。",
    ],
)
def test_main_removes_heading_duplicated_by_llm(data_dir, message):
    driver = FakeDriver([FinalAnswer(kind="answer", message=message, sources=["refund.md"])])

    _, output = run_main(["--driver", "cli", "質問"], driver, data_dir)

    assert output.count("回答案") == 1
    assert "[回答案]\n日割り返金はありません。" in output


def test_main_keeps_heading_like_text_in_the_middle(data_dir):
    message = "日割り返金はありません。\n【補足】180日間保持されます。"
    driver = FakeDriver([FinalAnswer(kind="answer", message=message, sources=["refund.md"])])

    _, output = run_main(["--driver", "cli", "質問"], driver, data_dir)

    assert message in output


def test_main_removes_duplicated_escalation_heading(data_dir):
    message = "【エスカレーションが必要】請求チームへ回してください。"
    driver = FakeDriver([FinalAnswer(kind="escalation", message=message, sources=[])])

    _, output = run_main(["--driver", "cli", "質問"], driver, data_dir)

    assert output.count("エスカレーションが必要") == 1


def test_main_prompts_question_with_simple_marker(data_dir):
    prompts = []
    driver = FakeDriver([FinalAnswer(kind="answer", message="回答", sources=["refund.md"])])

    main(
        ["--driver", "cli"],
        driver_factory=lambda name, model: driver,
        data_dir=data_dir,
        log_file=data_dir / "logs" / "runs.jsonl",
        input_fn=lambda prompt: (prompts.append(prompt), "質問です" if len(prompts) == 1 else "")[1],
        print_fn=lambda *args: None,
    )

    assert prompts[0] == "> "


def test_input_marker_blinks_only_on_terminal():
    from operator_help_agent.__main__ import input_marker

    assert input_marker(blink=False) == "> "
    assert input_marker(blink=True) == "\033[5m>\033[0m "


def test_main_uses_blinking_marker_when_enabled(data_dir):
    prompts = []
    driver = FakeDriver([FinalAnswer(kind="answer", message="回答", sources=["refund.md"])])

    main(
        ["--driver", "cli"],
        driver_factory=lambda name, model: driver,
        data_dir=data_dir,
        log_file=data_dir / "logs" / "runs.jsonl",
        input_fn=lambda prompt: (prompts.append(prompt), "質問です" if len(prompts) == 1 else "")[1],
        print_fn=lambda *args: None,
        blink=True,
    )

    assert prompts[0] == "\033[5m>\033[0m "
