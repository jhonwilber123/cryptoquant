"""Los cuadernos por tema de clase/cuadernos/.

Cada tema existe en Python (python/*.ipynb, generado desde fuentes/*.py) y en
R (r/*.Rmd). Lo que se comprueba aquí es lo que se rompe sin que nadie lo
note: un .ipynb editado a mano o sin regenerar, un tema que solo existe en un
lenguaje, una celda de preparación que se desvía, o periodos de estudio
distintos en Python y en R (las cifras dejarían de coincidir).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import nbformat
import pytest

CUADERNOS = Path(__file__).resolve().parents[1] / "clase" / "cuadernos"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generar_cuadernos_tema", CUADERNOS / "generar.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _temas(carpeta: Path, patron: str) -> list[str]:
    return sorted(p.stem for p in carpeta.glob(patron) if p.stem[:2].isdigit())


def test_cada_tema_en_python_y_en_r():
    python = _temas(CUADERNOS / "fuentes", "*.py")
    assert python == _temas(CUADERNOS / "r", "*.Rmd")
    assert python == _temas(CUADERNOS / "python", "*.ipynb")
    assert len(python) == 9


def test_los_ipynb_estan_al_dia_con_sus_fuentes(gen):
    for fuente in gen.fuentes([]):
        en_disco = nbformat.read(CUADERNOS / "python" / f"{fuente.stem}.ipynb", 4)
        esperado = gen.construir(fuente)
        assert [c.source for c in en_disco.cells] == [c.source for c in esperado.cells], fuente.name


def test_los_ipynb_se_publican_ejecutados_y_sin_errores():
    for nb_ruta in sorted((CUADERNOS / "python").glob("*.ipynb")):
        nb = nbformat.read(nb_ruta, 4)
        codigo = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.get("execution_count") for c in codigo), f"{nb_ruta.name}: hay celdas sin ejecutar"
        errores = [o for c in codigo for o in c.get("outputs", []) if o.output_type == "error"]
        assert not errores, f"{nb_ruta.name}: {errores[0].ename}"


def test_la_preparacion_es_comun_y_plegada(gen):
    preparacion = (CUADERNOS / "fuentes" / "_preparacion.py").read_text(encoding="utf-8").strip()
    for fuente in gen.fuentes([]):
        nb = gen.construir(fuente)
        prep = [c for c in nb.cells if c.cell_type == "code" and c.source == preparacion]
        if fuente.stem.startswith("01"):
            assert not prep                     # la Unidad 1 la construye paso a paso
            continue
        assert len(prep) == 1 and prep[0].metadata.get("cellView") == "form", fuente.name
    for rmd in sorted((CUADERNOS / "r").glob("0[2-9]*.Rmd")):
        assert 'source("comun.R")' in rmd.read_text(encoding="utf-8"), rmd.name


def test_mismo_periodo_de_estudio_en_python_y_en_r():
    def fechas(texto: str) -> tuple[str, str]:
        return (re.search(r'DESDE\s*(?:=|<-)\s*(?:as\.Date\()?"([\d-]+)"', texto).group(1),
                re.search(r'HASTA\s*(?:=|<-)\s*(?:as\.Date\()?"([\d-]+)"', texto).group(1))

    python = fechas((CUADERNOS / "fuentes" / "_preparacion.py").read_text(encoding="utf-8"))
    assert python == fechas((CUADERNOS / "r" / "comun.R").read_text(encoding="utf-8"))
    assert python == fechas((CUADERNOS / "r" / "01_datos.Rmd").read_text(encoding="utf-8"))
    assert python[1] in (CUADERNOS / "fuentes" / "01_datos.py").read_text(encoding="utf-8")


def test_formato_percent(gen):
    texto = "# %% [markdown]\n# # Título\n#\n# Texto\n\n# %%\nx = 1\n\n# %% preparacion\n"
    assert gen.celdas(texto) == [("md", "# Título\n\nTexto"), ("code", "x = 1"), ("prep", "")]
