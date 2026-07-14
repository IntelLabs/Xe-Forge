from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from xe_forge.prompts.device_prompts import PromptLibrary

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=select_autoescape(enabled_extensions=()),
)


def render_signature_instructions(template_name: str, **context) -> str:
    """Render a Signature instruction block from a Jinja2 template."""
    return _env.get_template(f"{template_name}.md.j2").render(**context)


__all__ = ["PromptLibrary", "render_signature_instructions"]
