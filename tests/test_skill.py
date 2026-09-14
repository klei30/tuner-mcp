from pathlib import Path

from tuner.mcp_server import CURATED_TOOLS
from tuner.recipes import CATALOG

ROOT = Path(__file__).parents[1]


def test_skill_documents_current_tools_and_recipes() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    tools = (ROOT / "references" / "tools.md").read_text(encoding="utf-8")
    recipes = (ROOT / "references" / "recipes.md").read_text(encoding="utf-8")

    assert skill.startswith("---\nname: tuner-mcp\n")
    assert "native Tuner MCP tools" in skill
    assert not [name for name in CURATED_TOOLS if f"`{name}`" not in tools]
    assert not [item.recipe for item in CATALOG if f"`{item.recipe}`" not in recipes]


def test_skill_references_exist() -> None:
    for name in ("tools", "recipes", "contracts", "workflows", "operations"):
        assert (ROOT / "references" / f"{name}.md").is_file()

    assert (ROOT / "agents" / "openai.yaml").is_file()
