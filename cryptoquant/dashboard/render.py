"""Genera el panel como un unico fichero HTML autocontenido.

Sin servidor, sin dependencias y sin conexion: los datos van incrustados como
JSON y los graficos se dibujan en SVG con JavaScript propio. Se abre con doble
clic y la tarea programada lo regenera en cada pasada.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config as cfgmod
from ..config import Config
from .state import clean, collect_state

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "__STATE__"

# Caracteres que no pueden aparecer tal cual dentro de un <script>.
_LINE_SEP = chr(0x2028)
_PARA_SEP = chr(0x2029)


def to_json(state: dict) -> str:
    """JSON listo para ir dentro de <script type="application/json">.

    `</` se escapa como `<\\/` (equivalente en JSON) para que ningun texto de los
    datos pueda cerrar la etiqueta script antes de tiempo. Los separadores de
    linea Unicode se escapan tambien por prudencia.
    """
    text = json.dumps(clean(state), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)
    text = text.replace("</", "<\\/")
    text = text.replace(_LINE_SEP, "\\u2028").replace(_PARA_SEP, "\\u2029")
    return text


def render_html(state: dict) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    if html.count(PLACEHOLDER) != 1:
        raise RuntimeError("la plantilla debe contener exactamente un marcador de estado")
    return html.replace(PLACEHOLDER, to_json(state), 1)


def build_dashboard(cfg: Config, out: Path | None = None) -> Path:
    out = out or cfgmod.REPORTS_DIR / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(collect_state(cfg))
    # Escritura atomica: si el navegador lo abre a mitad de generacion, ve la
    # version anterior completa en vez de un fichero cortado.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(out)
    return out
