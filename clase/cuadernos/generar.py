"""Genera los cuadernos de Python (python/*.ipynb) desde fuentes/*.py.

Cada fuente es un .py en formato «percent»: `# %%` abre una celda de código y
`# %% [markdown]` una de texto, con cada línea comentada. `# %% preparacion`
inserta la celda común (fuentes/_preparacion.py), que Colab muestra plegada.

    python clase/cuadernos/generar.py              # regenera los .ipynb (sin salidas)
    python clase/cuadernos/generar.py --ejecutar   # además los ejecuta y guarda las salidas
    python clase/cuadernos/generar.py --ejecutar 06 09   # solo los que empiezan por 06 y 09
    python clase/cuadernos/generar.py --r          # renderiza r/*.Rmd a HTML (R y pandoc)

Ejecutarlos necesita nbclient e ipykernel en el mismo Python; renderizar los
de R, Rscript y el pandoc de RStudio (variable RSTUDIO_PANDOC).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
FUENTES = AQUI / "fuentes"
PYTHON = AQUI / "python"
R = AQUI / "r"


def celdas(texto: str) -> list[tuple[str, str]]:
    """[(tipo, contenido)] de una fuente en formato percent."""
    partes: list[tuple[str, list[str]]] = []
    for linea in texto.splitlines():
        if linea.startswith("# %%"):
            marca = linea[4:].strip()
            partes.append(({"[markdown]": "md", "preparacion": "prep", "": "code"}[marca], []))
        elif partes:
            partes[-1][1].append(linea)
    salida = []
    for tipo, lineas in partes:
        if tipo == "md":
            lineas = [x[2:] if x.startswith("# ") else x.lstrip("#") for x in lineas]
        while lineas and not lineas[-1].strip():
            lineas.pop()
        while lineas and not lineas[0].strip():
            lineas.pop(0)
        salida.append((tipo, "\n".join(lineas)))
    return salida


def construir(fuente: Path) -> nbf.NotebookNode:
    preparacion = (FUENTES / "_preparacion.py").read_text(encoding="utf-8").strip()
    nb = nbf.v4.new_notebook()
    for tipo, contenido in celdas(fuente.read_text(encoding="utf-8")):
        if tipo == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(contenido))
        elif tipo == "prep":
            celda = nbf.v4.new_code_cell(preparacion)
            celda.metadata["cellView"] = "form"   # Colab la muestra plegada
            nb.cells.append(celda)
        else:
            nb.cells.append(nbf.v4.new_code_cell(contenido))
    for i, c in enumerate(nb.cells):
        c["id"] = f"c{i:02d}"                     # ids estables: el diff solo muestra cambios reales
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    return nb


def fuentes(prefijos: list[str]) -> list[Path]:
    todas = sorted(p for p in FUENTES.glob("[0-9]*.py"))
    return [p for p in todas if not prefijos or any(p.name.startswith(x) for x in prefijos)]


def ejecutar(nb: nbf.NotebookNode) -> None:
    from nbclient import NotebookClient

    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(PYTHON)}}).execute()


def renderizar_r(prefijos: list[str]) -> None:
    rscript = shutil.which("Rscript") or r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    entorno = dict(os.environ)
    entorno.setdefault("RSTUDIO_PANDOC", r"C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools")
    for rmd in sorted(R.glob("[0-9]*.Rmd")):
        if prefijos and not any(rmd.name.startswith(x) for x in prefijos):
            continue
        print("renderizando", rmd.name, flush=True)
        subprocess.run([rscript, "-e", f"rmarkdown::render('{rmd.name}', quiet = TRUE)"],
                       cwd=R, env=entorno, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefijos", nargs="*", help="solo los cuadernos cuyo nombre empieza así (01, 06...)")
    ap.add_argument("--ejecutar", action="store_true", help="ejecutar y guardar las salidas")
    ap.add_argument("--r", action="store_true", help="renderizar los cuadernos de R a HTML")
    args = ap.parse_args()
    if args.r:
        renderizar_r(args.prefijos)
        sys.exit(0)
    PYTHON.mkdir(exist_ok=True)
    for fuente in fuentes(args.prefijos):
        nb = construir(fuente)
        destino = PYTHON / (fuente.stem + ".ipynb")
        if args.ejecutar:
            print("ejecutando", fuente.stem, flush=True)
            ejecutar(nb)
        nbf.write(nb, destino)
        print("escrito", destino.relative_to(AQUI))
