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
    """Render a Signature instruction block from a Jinja2 template. From the registry, by
    ``dsl``: ``stages`` (the stage values it supports) and ``generic_python`` (a Python
    file that is not Triton's API, so none of Triton's text applies to it)."""
    from xe_forge.config import device_kind
    from xe_forge.dsl_registry import TRITON_FAMILY, get_stages_for_dsl
    from xe_forge.models import DSL

    dsl = DSL(str(context.get("dsl") or "triton"))
    if context.get("device_type"):  # the templates are per type: "xpu" of "xpu:1"
        context["device_type"] = device_kind(context["device_type"])
    context.setdefault("stages", [s.value for s in get_stages_for_dsl(dsl)])
    context.setdefault("generic_python", dsl.code_language == "python" and dsl not in TRITON_FAMILY)
    return _env.get_template(f"{template_name}.md.j2").render(**context)


__all__ = ["PromptLibrary", "render_signature_instructions"]
