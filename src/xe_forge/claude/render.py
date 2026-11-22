"""Render a template under ``templates/``. Jinja only: the skills that run under a framework's
interpreter render their scripts with it, so it imports nothing else of Xe-Forge."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(enabled_extensions=()),
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
    # A variable a template uses but the caller did not pass is an error, not a blank.
    undefined=StrictUndefined,
)


def render(template_name: str, **context: object) -> str:
    return _env.get_template(template_name).render(**context)
