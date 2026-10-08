"""エージェントが使うツールの定義と実装。

各ツールは ToolSpec にまとめる。
- name / description / input_schema: LLM に渡すツール定義
- handler: 実際に実行する関数。キーワード引数で入力を受け取り、文字列を返す
- requires_approval: True なら実行前にオペレーターの承認が必要（G2）
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

MAX_RESULTS = 5
SNIPPET_LENGTH = 200
NO_HIT = "該当なし"

TEAMS = ["請求チーム", "個人情報窓口", "セキュリティチーム", "技術サポート", "スーパーバイザー"]


class ToolError(Exception):
    """ツールの入力や実行に問題があったことを LLM に伝えるための例外。"""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., str]
    requires_approval: bool = False


@dataclass(frozen=True)
class _Section:
    file: str
    heading: str
    body: str


def build_tools(data_dir: Path, now: Callable[[], datetime] = datetime.now) -> list[ToolSpec]:
    data_dir = Path(data_dir)
    manuals_dir = data_dir / "manuals"
    tickets_path = data_dir / "past_tickets.json"
    escalations_path = data_dir / "escalations.jsonl"

    def manual_names() -> list[str]:
        return sorted(p.name for p in manuals_dir.glob("*.md"))

    def search_manuals(query: str) -> str:
        keywords = _parse_keywords(query)
        sections = [
            section
            for name in manual_names()
            for section in _split_sections(name, (manuals_dir / name).read_text(encoding="utf-8"))
        ]
        hits = _rank(sections, keywords, lambda s: s.heading + "\n" + s.body)
        if not hits:
            return NO_HIT
        return "\n\n".join(
            f"[{s.file}] {s.heading}\n{s.body[:SNIPPET_LENGTH]}" for s in hits
        )

    def read_manual(name: str) -> str:
        # G5: ユーザー入力をパスとして組み立てず、実在するファイル名の一覧と照合する
        names = manual_names()
        if name not in names:
            raise ToolError(f"マニュアル '{name}' はありません。読めるファイル: {', '.join(names)}")
        return (manuals_dir / name).read_text(encoding="utf-8")

    def search_past_tickets(query: str) -> str:
        keywords = _parse_keywords(query)
        tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
        hits = _rank(
            tickets,
            keywords,
            lambda t: "\n".join([t["category"], t["question"], t["answer"]]),
        )
        if not hits:
            return NO_HIT
        return "\n\n".join(
            f"{t['id']}（{t['date']} / {t['category']} / "
            f"エスカレーション: {'あり' if t['escalated'] else 'なし'}）\n"
            f"Q: {t['question']}\nA: {t['answer']}"
            for t in hits
        )

    def create_escalation(team: str, summary: str, reason: str) -> str:
        if team not in TEAMS:
            raise ToolError(f"宛先 '{team}' は不正です。指定できる宛先: {', '.join(TEAMS)}")
        existing = (
            escalations_path.read_text(encoding="utf-8").splitlines()
            if escalations_path.exists()
            else []
        )
        record = {
            "id": f"ESC-{len(existing) + 1:04d}",
            "created_at": now().isoformat(),
            "team": team,
            "summary": summary,
            "reason": reason,
        }
        with escalations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return f"起票しました: {record['id']}（宛先: {team}）"

    query_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索キーワード。複数ある場合はスペース区切り（例: '解約 返金'）",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    return [
        ToolSpec(
            name="search_manuals",
            description=(
                "業務マニュアルをキーワードで検索し、一致したセクション（ファイル名・見出し・抜粋）を"
                "一致数の多い順に最大5件返す。全文が必要なら read_manual を使う。"
            ),
            input_schema=query_schema,
            handler=search_manuals,
        ),
        ToolSpec(
            name="read_manual",
            description="業務マニュアルの全文を返す。name には search_manuals の結果に出たファイル名を指定する。",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "マニュアルのファイル名（例: 'cancellation_and_refund.md'）",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=read_manual,
        ),
        ToolSpec(
            name="search_past_tickets",
            description=(
                "過去の問い合わせと、そのときの回答をキーワードで検索し、一致数の多い順に最大5件返す。"
                "エスカレーションされた事例かどうかも含む。"
            ),
            input_schema=query_schema,
            handler=search_past_tickets,
        ),
        ToolSpec(
            name="create_escalation",
            description=(
                "担当チームへのエスカレーションを起票する。実行前にオペレーターの承認が必要。"
                "マニュアルのエスカレーション基準に当てはまる場合にだけ使う。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "team": {"type": "string", "enum": TEAMS, "description": "エスカレーション先"},
                    "summary": {"type": "string", "description": "問い合わせの要約"},
                    "reason": {"type": "string", "description": "エスカレーションする理由"},
                },
                "required": ["team", "summary", "reason"],
                "additionalProperties": False,
            },
            handler=create_escalation,
            requires_approval=True,
        ),
    ]


ASK_OPERATOR = "ask_operator"
MAX_QUESTIONS = 2


def make_ask_operator_tool(ask: Callable[[str], str]) -> ToolSpec:
    """オペレーターに再質問するツールを作る。呼び出し回数を数えるため、実行ごとに新しく作る。"""
    asked = 0

    def ask_operator(question: str) -> str:
        nonlocal asked
        if not question.strip():
            raise ToolError("question が空です。確認したい内容を指定してください。")
        # G8: 聞きすぎを防ぐ
        if asked >= MAX_QUESTIONS:
            raise ToolError(
                f"再質問は{MAX_QUESTIONS}回までです。手元の情報で回答するか、エスカレーションしてください。"
            )
        asked += 1
        reply = ask(question).strip()
        return f"オペレーターの回答: {reply}" if reply else "オペレーターの回答: （回答なし）"

    return ToolSpec(
        name=ASK_OPERATOR,
        description=(
            "オペレーターに確認したいことを質問し、その回答を得る。"
            "回答の内容が変わる情報（契約経路、エラーコードなど）が質問から読み取れないときにだけ使う。"
            "マニュアルや過去事例を調べればわかることは聞かない。確認したいことは1回にまとめる。"
            f"1回の対応で{MAX_QUESTIONS}回まで。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "オペレーターへの確認事項"},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=ask_operator,
    )


def _parse_keywords(query: str) -> list[str]:
    keywords = query.split()
    if not keywords:
        raise ToolError("query が空です。検索キーワードを指定してください。")
    return keywords


def _split_sections(file: str, text: str) -> list[_Section]:
    """Markdown を '## ' 見出し単位に分ける。最初の見出しより前は '# ' タイトルを見出しとする。"""
    sections: list[_Section] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append(_Section(file, heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        elif line.startswith("# ") and not heading:
            heading = line[2:].strip()
        else:
            body.append(line)
    sections.append(_Section(file, heading, "\n".join(body).strip()))
    return sections


def _rank[T](items: list[T], keywords: list[str], text_of: Callable[[T], str]) -> list[T]:
    """含まれるキーワードの数で並べ替え、1つ以上含むものを上位から返す（同点は元の順）。"""
    lowered = [k.lower() for k in keywords]
    scored = [(sum(k in text_of(item).lower() for k in lowered), item) for item in items]
    hits = [pair for pair in scored if pair[0] > 0]
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in hits[:MAX_RESULTS]]
