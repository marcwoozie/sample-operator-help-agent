"""claude -p ドライバー。

ネイティブの Tool Use は使わない。ツール一覧と履歴をプロンプトに書き込み、
「次の一手」（ツール呼び出し or 最終回答）を structured output（--json-schema）で出させる。
claude -p は毎回使い捨てで起動し、会話の状態は Agent 側の履歴で持つ。
"""

import json
import subprocess
from dataclasses import dataclass
from typing import Callable

from operator_help_agent.models import (
    FINAL_ANSWER_SCHEMA,
    AssistantTurn,
    DriverError,
    DriverStep,
    FinalAnswer,
    ToolCall,
    ToolResultsTurn,
    Turn,
    Usage,
    UserTurn,
)
from operator_help_agent.tools import ToolSpec

MODEL = "sonnet"
TIMEOUT_SECONDS = 300

NEXT_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["call_tools", "final"],
            "description": "call_tools: ツールを呼ぶ / final: 最終回答を返す",
        },
        "tool_calls": {
            "type": "array",
            "description": "action が call_tools のときに指定する",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "input": {"type": "object"},
                },
                "required": ["name", "input"],
                "additionalProperties": False,
            },
        },
        "final": {**FINAL_ANSWER_SCHEMA, "description": "action が final のときに指定する"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], str], CommandResult]


def run_subprocess(args: list[str], input: str) -> CommandResult:
    completed = subprocess.run(
        args, input=input, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class ClaudeCliDriver:
    def __init__(self, run_command: CommandRunner = run_subprocess, model: str = MODEL):
        # run_command を差し替えられるようにして、テストでは偽の実行関数を渡す
        self.run_command = run_command
        self.model = model

    def next_step(
        self, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> DriverStep:
        args = [
            "claude",
            "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(NEXT_STEP_SCHEMA, ensure_ascii=False),
            "--system-prompt", system,
            "--model", self.model,
            "--no-session-persistence",
            # Claude Code 自身の機能を切り、このエージェントのツールだけを使わせる
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "",
        ]
        result = self.run_command(args, _build_prompt(history, tools))
        payload = _parse_output(result)
        return DriverStep(
            step=_parse_step(payload["structured_output"], history), usage=_parse_usage(payload)
        )


def _parse_step(step: dict, history: list[Turn]) -> AssistantTurn | FinalAnswer:
    # G7: 有効な次の一手を得られない場合は DriverError
    match step.get("action"):
        case "call_tools":
            raw_calls = step.get("tool_calls") or []
            if not raw_calls:
                raise DriverError("call_tools なのに tool_calls が空です")
            offset = _count_tool_calls(history)
            return AssistantTurn(
                tool_calls=[
                    ToolCall(id=f"cli-{offset + i + 1}", name=c["name"], input=c["input"])
                    for i, c in enumerate(raw_calls)
                ]
            )
        case "final":
            final = step.get("final")
            try:
                return FinalAnswer(
                    kind=final["kind"], message=final["message"], sources=list(final["sources"])
                )
            except (KeyError, TypeError) as e:
                raise DriverError(f"final の形式が不正です: {final!r}") from e
        case other:
            raise DriverError(f"想定外の action です: {other!r}")


def _parse_usage(payload: dict) -> Usage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    # claude -p が内部で複数モデルを使った場合は、すべての名前をつなげる
    models = sorted(payload.get("modelUsage") or {})
    return Usage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cost_usd=payload.get("total_cost_usd"),
        model=",".join(models) or None,
    )


def _parse_output(result: CommandResult) -> dict:
    if result.returncode != 0:
        raise DriverError(f"claude -p が失敗しました（exit {result.returncode}）: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise DriverError(f"claude -p の出力を JSON として解釈できません: {result.stdout[:200]!r}") from e
    if payload.get("is_error") or payload.get("subtype") != "success":
        raise DriverError(f"claude -p がエラーを返しました: {payload.get('subtype')}")
    if not isinstance(payload.get("structured_output"), dict):
        raise DriverError("claude -p の出力に structured_output がありません")
    return payload


def _count_tool_calls(history: list[Turn]) -> int:
    return sum(len(t.tool_calls) for t in history if isinstance(t, AssistantTurn))


def _build_prompt(history: list[Turn], tools: list[ToolSpec]) -> str:
    lines = [
        "# 利用できるツール",
        "",
        "以下のツールは、あなたが直接呼び出すことはできません。"
        'ツールを使いたいときは、action を "call_tools" にした JSON で指示してください。'
        "こちらで実行し、結果を次のやりとりに書き込みます。",
        "",
    ]
    for tool in tools:
        lines += [
            f"## {tool.name}",
            tool.description,
            f"入力スキーマ: {json.dumps(tool.input_schema, ensure_ascii=False)}",
            "",
        ]

    lines += ["# これまでのやりとり", ""]
    for turn in history:
        match turn:
            case UserTurn(text=text):
                lines += ["## オペレーターの質問", text, ""]
            case AssistantTurn(tool_calls=calls):
                lines += ["## あなたのツール呼び出し"]
                lines += [
                    f"- [{c.id}] {c.name} {json.dumps(c.input, ensure_ascii=False)}" for c in calls
                ]
                lines += [""]
            case ToolResultsTurn(results=results):
                lines += ["## ツールの実行結果"]
                for r in results:
                    label = "エラー" if r.is_error else "結果"
                    lines += [f"### [{r.call_id}] {label}", r.content]
                lines += [""]

    lines += [
        "# 指示",
        "次の一手を JSON で出力してください。",
        '- 情報を集める必要があれば action を "call_tools" にし、tool_calls に呼び出すツールを並べる',
        '- 回答をまとめられる、またはエスカレーションすべきと判断したら action を "final" にし、final に最終回答を入れる',
    ]
    return "\n".join(lines)
