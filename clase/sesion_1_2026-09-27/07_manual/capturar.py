"""Capturas del manual de usuario: la app, el panel, los informes de R y la clase.

Necesita:
- la app en Docker en http://127.0.0.1:8501 con la cartera INVENTADA de
  10_version_de_hoy/piloto_demo (copiada aparte: la app escribe en ella) y una
  contraseña de prueba, p. ej.:
      docker run -d --name piloto-prueba -p 127.0.0.1:8501:8501 \
        -v <copia de piloto_demo>:/datos -e PILOTO_CLAVE_HASH=<derivada> piloto-app
- Playwright (pip install playwright), que maneja el Edge del equipo sin ventana;
- R (para el grafico de clase01.R) y nbconvert (para el bloque del cuaderno). Si
  nbconvert no encuentra sus plantillas, JUPYTER_PATH debe apuntar a su share/jupyter.

Uso, desde la raiz del repositorio:
    python clase/sesion_1_2026-09-27/07_manual/capturar.py <contraseña de prueba>
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CLASE = AQUI.parent
SALIDA = AQUI / "capturas"
URL = "http://127.0.0.1:8501"
ESCALA = 1.5   # pixeles por punto: nitidas al ancho de una pagina
RSCRIPT = Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")


def esperar_app(pag: Page, ms: int = 1500) -> None:
    """Hasta que Streamlit termina de ejecutar la pagina y los graficos se dibujan."""
    pag.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"] [data-testid=\"stStatusWidgetRunningIcon\"]')"
        " && !document.querySelector('[data-testid=\"stSpinner\"]')", timeout=180_000)
    pag.wait_for_timeout(ms)


def panel_visible(pag: Page):
    return pag.locator('[data-testid="stTabPanel"]:visible').first


def pestana(pag: Page, nombre: str) -> None:
    pag.get_by_role("tab", name=nombre).click()
    esperar_app(pag)


def guardar(elemento, nombre: str) -> None:
    elemento.screenshot(path=str(SALIDA / nombre))
    print("captura:", nombre)


def app(nav, clave: str) -> None:
    ctx = nav.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=ESCALA,
                          color_scheme="light", locale="es-PE")
    pag = ctx.new_page()
    pag.goto(URL)

    campo = pag.get_by_label("Contraseña")
    campo.wait_for(timeout=60_000)
    pag.set_viewport_size({"width": 1000, "height": 520})
    esperar_app(pag, 800)
    # Solo el titulo y el formulario, sin el margen de la pagina.
    titulo = pag.get_by_role("heading", name="Piloto de riesgo").bounding_box()
    form = pag.locator('[data-testid="stForm"]').bounding_box()
    pag.screenshot(path=str(SALIDA / "01_acceso.png"), clip={
        "x": form["x"] - 16, "y": titulo["y"] - 12, "width": form["width"] + 32,
        "height": form["y"] + form["height"] + 16 - (titulo["y"] - 12)})
    print("captura: 01_acceso.png")

    pag.set_viewport_size({"width": 1440, "height": 900})
    campo.fill(clave)
    pag.get_by_role("button", name="Entrar").click()
    pag.get_by_text("Valor de la cartera").wait_for(timeout=180_000)
    esperar_app(pag, 2500)
    guardar(pag, "02_vista_general.png")

    # Streamlit desplaza un contenedor interno, no la pagina: una captura no pasa
    # del alto de la ventana. Para las pestañas, una ventana alta.
    pag.set_viewport_size({"width": 1440, "height": 2600})
    barra = pag.locator('[data-testid="stSidebarUserContent"]')
    pag.get_by_text("Más ajustes").click()
    esperar_app(pag, 1000)
    mas_ajustes = barra.locator('[data-testid="stExpander"]').screenshot()
    pag.get_by_text("Más ajustes").click()
    esperar_app(pag, 800)

    guardar(panel_visible(pag), "04_hoy.png")

    pestana(pag, "Tiempos duros")
    pag.get_by_role("button", name="Comprobar el mal día").click()
    # Las tablas se dibujan en un canvas: se espera a la nota que va debajo.
    pag.get_by_text("Se esperaba fallar el 5 %").wait_for(timeout=180_000)
    esperar_app(pag, 1500)
    guardar(panel_visible(pag), "05_tiempos_duros.png")

    pestana(pag, "La regla en el pasado")
    esperar_app(pag, 3000)
    guardar(panel_visible(pag), "06_regla_en_el_pasado.png")

    pestana(pag, "Mi diario")
    esperar_app(pag, 2500)
    guardar(panel_visible(pag), "07_mi_diario.png")

    # Cambiar el riesgo sin guardar: la barra lateral lo avisa.
    pestana(pag, "Hoy")
    pag.get_by_text("Moderado", exact=True).first.click()
    esperar_app(pag, 2000)
    # Del nivel de riesgo al aviso, y al lado Más ajustes: entera, la barra no cabe en una pagina.
    caja = barra.bounding_box()
    arriba = barra.get_by_role("heading", name="Riesgo").bounding_box()["y"] - 12
    aviso = barra.get_by_text("Hay cambios sin guardar").bounding_box()
    abajo = aviso["y"] + aviso["height"] + 24
    riesgo = pag.screenshot(clip={"x": caja["x"], "y": arriba, "width": caja["width"], "height": abajo - arriba})
    juntar([riesgo, mas_ajustes], "03_barra_lateral.png")
    ctx.close()


def juntar(imagenes: list[bytes], nombre: str, hueco: int = 48) -> None:
    """Varias capturas en una, una al lado de otra y alineadas arriba."""
    from PIL import Image

    partes = [Image.open(io.BytesIO(b)).convert("RGB") for b in imagenes]
    lienzo = Image.new("RGB", (sum(p.width for p in partes) + hueco * (len(partes) - 1),
                               max(p.height for p in partes)), "white")
    x = 0
    for p in partes:
        lienzo.paste(p, (x, 0))
        x += p.width + hueco
    lienzo.save(SALIDA / nombre)
    print("captura:", nombre)


def archivo(nav, ruta: Path, nombre: str, titulo: str | None = None, texto: str | None = None,
            alto: int = 900) -> None:
    """Una pagina HTML local, desde un encabezado (`titulo`, no el del indice) o un parrafo (`texto`)."""
    ctx = nav.new_context(viewport={"width": 1280, "height": alto}, device_scale_factor=ESCALA,
                          color_scheme="light")
    pag = ctx.new_page()
    pag.goto(ruta.as_uri())
    pag.wait_for_timeout(2500)
    destino = (pag.get_by_role("heading", name=titulo) if titulo else
               pag.get_by_text(texto) if texto else None)
    if destino is not None:
        destino.first.evaluate("e => e.scrollIntoView()")
        pag.evaluate("window.scrollBy(0, -20)")
        pag.wait_for_timeout(800)
    guardar(pag, nombre)
    ctx.close()


def grafico_clase01() -> None:
    """El grafico de capital de clase01.R (Posit Cloud), con el CSV de respaldo de la clase."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "BTCUSDT_1d.csv").write_bytes((CLASE / "04_datos_respaldo" / "BTCUSDT_1d.csv").read_bytes())
        guion = (RAIZ / "clase" / "01_primera_clase" / "posit" / "clase01.R").as_posix()
        subprocess.run([str(RSCRIPT), "-e",
                        "suppressMessages({png('g%d.png', width = 1800, height = 1000, res = 220); "
                        f"source('{guion}', print.eval = TRUE); invisible(dev.off())}})"],
                       cwd=tmp, check=True, capture_output=True)
        (SALIDA / "12_clase01_capital.png").write_bytes((Path(tmp) / "g1.png").read_bytes())
    print("captura: 12_clase01_capital.png")


def _celdas_bloque1() -> list:
    """El bloque 1 del cuaderno con la solucion del docente pegada y lo que responde el verificador."""
    import nbformat

    al = json.loads((CLASE / "02_cuadernos_colab" / "alumno.ipynb").read_text(encoding="utf-8"))
    do = json.loads((CLASE / "02_cuadernos_colab" / "docente.ipynb").read_text(encoding="utf-8"))
    src = lambda c: "".join(c["source"])  # noqa: E731
    ns: dict = {}

    def correr(codigo: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(codigo, "<celda>", "exec"), ns)  # noqa: S102 - el cuaderno de la clase
        return buf.getvalue()

    def celda(codigo: str, salida: str, n: int):
        return nbformat.v4.new_code_cell(codigo, execution_count=n, outputs=[
            nbformat.v4.new_output("stream", name="stdout", text=salida)])

    correr(src(al["cells"][1]))                                  # Preparacion: los verificadores
    pegado = src(al["cells"][3]).rstrip("\n") + "\n" + src(do["cells"][3])   # lo que pegaria el alumno
    salida_pegado = correr(src(do["cells"][3]))
    bien = correr(src(al["cells"][4]))
    mal = correr("valida_antes, valida_despues = True, True\n" + src(al["cells"][4]))
    return [nbformat.v4.new_markdown_cell(src(al["cells"][2])), celda(pegado, salida_pegado, 2),
            celda(src(al["cells"][4]), bien, 3), celda(src(al["cells"][4]), mal, 3)]


def cuaderno(nav) -> None:
    import nbformat
    from nbconvert import HTMLExporter

    nb = nbformat.v4.new_notebook()
    nb.cells = _celdas_bloque1()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    with tempfile.TemporaryDirectory() as tmp:
        pagina = Path(tmp) / "bloque1.html"
        pagina.write_text(html, encoding="utf-8")
        ctx = nav.new_context(viewport={"width": 1100, "height": 3000}, device_scale_factor=ESCALA)
        pag = ctx.new_page()
        pag.goto(pagina.as_uri())
        pag.wait_for_timeout(1500)
        celdas = pag.locator(".jp-Cell")
        guardar(celdas.nth(0), "13_cuaderno_bloque.png")
        guardar(celdas.nth(2), "14_cuaderno_verificar.png")
        guardar(celdas.nth(3), "15_cuaderno_revisar.png")
        ctx.close()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    SALIDA.mkdir(parents=True, exist_ok=True)
    grafico_clase01()
    with sync_playwright() as p:
        nav = p.chromium.launch(channel="msedge", headless=True)
        cuaderno(nav)
        app(nav, sys.argv[1])
        archivo(nav, RAIZ / "reports" / "dashboard.html", "09_panel.png")
        archivo(nav, RAIZ / "laboratorio" / "r" / "riesgo.html", "10_r_riesgo.png",
                titulo="¿Coincide con Python?", alto=300)
        archivo(nav, RAIZ / "laboratorio" / "r" / "laboratorio.html", "11_r_laboratorio.png",
                texto="¿Coinciden con las de Python?", alto=360)
        nav.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
