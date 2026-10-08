import json
from datetime import datetime

import pytest

from operator_help_agent.tools import ToolError, build_tools

FIXED_NOW = datetime(2026, 10, 7, 12, 0, 0)


@pytest.fixture
def data_dir(tmp_path):
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    (manuals / "billing.md").write_text(
        "# 請求\n"
        "\n"
        "## 請求タイミング\n"
        "前回の請求日から1か月後に請求する。\n"
        "\n"
        "## プラン変更\n"
        "アップグレードは即時反映。ダウングレードは次回請求日から。返金はしない。\n",
        encoding="utf-8",
    )
    (manuals / "refund.md").write_text(
        "# 解約と返金\n"
        "\n"
        "## 解約後の扱い\n"
        "日割り返金はしない。解約後も次回請求日の前日まで視聴できる。\n",
        encoding="utf-8",
    )
    (tmp_path / "past_tickets.json").write_text(
        json.dumps(
            [
                {
                    "id": "T-1",
                    "date": "2026-08-01",
                    "category": "解約",
                    "question": "解約したら日割り返金ある？",
                    "answer": "日割り返金はなしと案内。",
                    "escalated": False,
                },
                {
                    "id": "T-2",
                    "date": "2026-08-02",
                    "category": "請求",
                    "question": "二重請求された",
                    "answer": "請求チームへエスカレーション。",
                    "escalated": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tools(data_dir):
    return {spec.name: spec for spec in build_tools(data_dir, now=lambda: FIXED_NOW)}


# --- ツール定義 ---


def test_build_tools_returns_four_tools(tools):
    assert set(tools) == {
        "search_manuals",
        "read_manual",
        "search_past_tickets",
        "create_escalation",
    }


def test_every_tool_has_strict_object_schema(tools):
    for spec in tools.values():
        assert spec.description
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.input_schema["required"]


def test_only_create_escalation_requires_approval(tools):
    needs_approval = {name for name, spec in tools.items() if spec.requires_approval}
    assert needs_approval == {"create_escalation"}


# --- search_manuals ---


def test_search_manuals_returns_matching_section(tools):
    result = tools["search_manuals"].handler(query="日割り")

    assert "refund.md" in result
    assert "解約後の扱い" in result


def test_search_manuals_ranks_sections_by_keyword_hits(tools):
    # 「請求日」「返金」の両方を含む「解約後の扱い」「プラン変更」が、
    # 「請求日」しか含まない「請求タイミング」より上に来る
    result = tools["search_manuals"].handler(query="返金 請求日")

    assert result.index("解約後の扱い") < result.index("請求タイミング")
    assert result.index("プラン変更") < result.index("請求タイミング")


def test_search_manuals_returns_at_most_five_sections(data_dir):
    lines = ["# 多い\n"] + [f"## セクション{i}\nキーワード\n" for i in range(10)]
    (data_dir / "manuals" / "many.md").write_text("".join(lines), encoding="utf-8")
    tools = {spec.name: spec for spec in build_tools(data_dir)}

    result = tools["search_manuals"].handler(query="キーワード")

    assert result.count("[many.md]") == 5


def test_search_manuals_reports_no_hit(tools):
    assert tools["search_manuals"].handler(query="存在しない語") == "該当なし"


def test_search_manuals_rejects_blank_query(tools):
    with pytest.raises(ToolError):
        tools["search_manuals"].handler(query="   ")


# --- read_manual ---


def test_read_manual_returns_full_text(tools):
    result = tools["read_manual"].handler(name="refund.md")

    assert "# 解約と返金" in result
    assert "日割り返金はしない。" in result


@pytest.mark.parametrize(
    "name",
    ["missing.md", "../past_tickets.json", "/etc/passwd", "manuals/refund.md"],
)
def test_read_manual_rejects_unknown_or_outside_files(tools, name):
    # G5: data/manuals/ 直下の実在ファイル以外は読めない
    with pytest.raises(ToolError) as excinfo:
        tools["read_manual"].handler(name=name)

    # LLM が選び直せるように、読めるファイル名を伝える
    assert "billing.md" in str(excinfo.value)
    assert "refund.md" in str(excinfo.value)


# --- search_past_tickets ---


def test_search_past_tickets_returns_matching_ticket(tools):
    result = tools["search_past_tickets"].handler(query="日割り")

    assert "T-1" in result
    assert "日割り返金はなしと案内。" in result
    assert "T-2" not in result


def test_search_past_tickets_matches_category(tools):
    result = tools["search_past_tickets"].handler(query="請求")

    assert "T-2" in result


def test_search_past_tickets_reports_no_hit(tools):
    assert tools["search_past_tickets"].handler(query="存在しない語") == "該当なし"


# --- create_escalation ---


def test_create_escalation_appends_record(tools, data_dir):
    result = tools["create_escalation"].handler(
        team="請求チーム", summary="二重請求の申告", reason="返金判断が必要"
    )

    assert "ESC-0001" in result
    records = [
        json.loads(line)
        for line in (data_dir / "escalations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "id": "ESC-0001",
            "created_at": "2026-10-07T12:00:00",
            "team": "請求チーム",
            "summary": "二重請求の申告",
            "reason": "返金判断が必要",
        }
    ]


def test_create_escalation_numbers_sequentially(tools):
    tools["create_escalation"].handler(team="技術サポート", summary="a", reason="b")
    result = tools["create_escalation"].handler(team="技術サポート", summary="c", reason="d")

    assert "ESC-0002" in result


def test_create_escalation_rejects_unknown_team(tools, data_dir):
    with pytest.raises(ToolError):
        tools["create_escalation"].handler(team="営業部", summary="a", reason="b")

    assert not (data_dir / "escalations.jsonl").exists()


# --- ask_operator ---


def test_ask_operator_returns_operator_reply():
    from operator_help_agent.tools import make_ask_operator_tool

    asked = []
    tool = make_ask_operator_tool(lambda question: asked.append(question) or "App Storeです")

    result = tool.handler(question="契約はどこからですか？")

    assert asked == ["契約はどこからですか？"]
    assert "App Storeです" in result
    assert tool.name == "ask_operator"
    assert tool.requires_approval is False
    assert tool.input_schema["required"] == ["question"]
    assert tool.input_schema["additionalProperties"] is False


def test_ask_operator_treats_blank_reply_as_no_answer():
    from operator_help_agent.tools import make_ask_operator_tool

    tool = make_ask_operator_tool(lambda question: "   ")

    assert "回答なし" in tool.handler(question="契約はどこからですか？")


def test_ask_operator_can_be_used_at_most_twice():
    # G8: 聞きすぎを防ぐ
    from operator_help_agent.tools import make_ask_operator_tool

    asked = []
    tool = make_ask_operator_tool(lambda question: asked.append(question) or "答え")
    tool.handler(question="1つ目")
    tool.handler(question="2つ目")

    with pytest.raises(ToolError) as excinfo:
        tool.handler(question="3つ目")

    assert asked == ["1つ目", "2つ目"]
    assert "エスカレーション" in str(excinfo.value)


def test_ask_operator_rejects_blank_question():
    from operator_help_agent.tools import make_ask_operator_tool

    tool = make_ask_operator_tool(lambda question: "答え")

    with pytest.raises(ToolError):
        tool.handler(question="  ")
