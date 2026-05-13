from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from xe_forge.core.tile_search.templates.fa import generate_fa_source
from xe_forge.core.tile_search.templates.fa_v2 import generate_fa_v2_source
from xe_forge.core.tile_search.templates.gemm import generate_gemm_source
from xe_forge.core.tile_search.templates.grouped_gemm import generate_grouped_gemm_source
from xe_forge.core.tile_search.templates.moe_gemm import generate_moe_gemm_source

_TEMPLATES_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(enabled_extensions=()),
    keep_trailing_newline=True,
    variable_start_string="<%",
    variable_end_string="%>",
    block_start_string="<%%",
    block_end_string="%%>",
    comment_start_string="<#",
    comment_end_string="#>",
)


def render(template_name: str, **context: object) -> str:
    return _env.get_template(template_name).render(**context)


__all__ = [
    "generate_fa_source",
    "generate_fa_v2_source",
    "generate_gemm_source",
    "generate_grouped_gemm_source",
    "generate_moe_gemm_source",
    "render",
]
