from operator_help_agent.prompts import (
    DEFAULT_ANSWER_TEMPLATE_PATH,
    SYSTEM_PROMPT,
    build_system_prompt,
    load_answer_template,
)

TEMPLATE = "## 結論\n（1〜2文で）\n\n## お客様への案内文\n（案内文）\n"


def test_build_system_prompt_embeds_answer_template():
    prompt = build_system_prompt(TEMPLATE)

    assert TEMPLATE.strip() in prompt
    assert "テンプレート" in prompt


def test_load_answer_template_reads_file(tmp_path):
    path = tmp_path / "answer.md"
    path.write_text(TEMPLATE, encoding="utf-8")

    assert load_answer_template(path) == TEMPLATE


def test_bundled_template_has_expected_sections():
    template = load_answer_template(DEFAULT_ANSWER_TEMPLATE_PATH)

    for heading in ["## 回答", "## お客様への案内文", "## 補足・注意点"]:
        assert heading in template


def test_default_system_prompt_uses_bundled_template():
    assert load_answer_template(DEFAULT_ANSWER_TEMPLATE_PATH).strip() in SYSTEM_PROMPT
