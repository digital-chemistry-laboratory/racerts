"""The documentation names every setting and every option: a new one needs its row."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from racerts import cli
from racerts.config import PipelineConfig

DOCS = Path(__file__).resolve().parents[1] / "docs"
pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="no docs next to the tests")


def test_every_setting_has_its_row():
    # Sections as PipelineConfig has them; the restraints have a page of their own.
    config = PipelineConfig()
    settings = (DOCS / "settings.md").read_text()
    restraints = (DOCS / "restraints.md").read_text()
    sections = {"embed", "refine", "prune", "restraints"}
    missing = [
        f.name
        for f in fields(config)
        if f.name not in sections and f"`{f.name}`" not in settings
    ]
    for section in sorted(sections):
        page = restraints if section == "restraints" else settings
        missing += [
            f"{section}.{f.name}"
            for f in fields(getattr(config, section))
            if f"`{f.name}`" not in page
        ]
    assert missing == []


def test_every_option_of_the_command_line_is_described():
    described = "".join((DOCS / page).read_text() for page in ("cli.md", "swap.md"))
    parser = cli._subcommand_parser()
    commands = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    missing = sorted(
        {
            f"{name} {option}"
            for name, command in commands.choices.items()
            for action in command._actions
            for option in action.option_strings
            if option.startswith("--")
            and option != "--help"
            and option not in described
        }
    )
    assert missing == []


def test_every_extra_that_the_docs_name_exists():
    # pip install "racerts[xtb]" on a page needs the extra in pyproject.toml.
    root = DOCS.parent
    extras = (root / "pyproject.toml").read_text()
    extras = extras.split("[project.optional-dependencies]")[1].split("\n[")[0]
    defined = set(re.findall(r"^(\w+) = \[", extras, flags=re.MULTILINE))
    named = set()
    for page in [*DOCS.rglob("*.md"), root / "README.md"]:
        named |= set(re.findall(r"racerts\[(\w+)\]", page.read_text()))
    assert {"ase", "xtb", "yaml"} <= named <= defined
