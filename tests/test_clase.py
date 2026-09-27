"""Los verificadores de la primera clase.

Un alumno que nunca programo no puede juzgar el codigo que le da la IA: el
verificador lo hace por el. Por eso lo que importa no es que acepte lo
correcto, sino que detecte los errores tipicos de la IA y diga que revisar:
la fecha tomada del cierre, la unidad de tiempo equivocada, la vela en curso,
porcentajes en lugar de fracciones, 252 dias en lugar de 365, o un modelo que
"acierta" demasiado porque mezclo el futuro.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

CLASE = Path(__file__).resolve().parents[1] / "clase" / "01_primera_clase"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generar_cuadernos", CLASE / "generar_cuadernos.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def verificar(gen, capsys):
    ns: dict = {}
    exec(gen.PREPARACION, ns)
    capsys.readouterr()
    return ns["verificar"]


@pytest.fixture
def df():
    """Velas diarias desde 2023-01-01 con el cierre de control real en su dia."""
    fechas = pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = 16616.75 * np.exp(np.concatenate([[0], np.cumsum(rng.standard_t(4, 399) * 0.03)]))
    openp = np.concatenate([[16541.77], close[:-1]])
    return pd.DataFrame({"fecha": fechas, "open": openp, "high": np.maximum(openp, close) * 1.01,
                         "low": np.minimum(openp, close) * 0.99, "close": close, "volume": 1.0})


def _salida(capsys):
    return capsys.readouterr().out


def test_velas_correctas(verificar, df, capsys):
    verificar.velas(df)
    assert _salida(capsys).startswith("[OK]")


def test_fecha_tomada_del_cierre(verificar, df, capsys):
    # Binance da la hora de cierre como 23:59:59.999 del MISMO dia: el precio de
    # control no lo delata, la hora si.
    malo = df.assign(fecha=df["fecha"] + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))
    verificar.velas(malo)
    assert "hora de CIERRE" in _salida(capsys)


def test_columna_equivocada_o_dia_desplazado(verificar, df, capsys):
    verificar.velas(df.assign(close=df["open"]))           # la apertura tomada como cierre
    assert "columna equivocada" in _salida(capsys)
    verificar.velas(df.assign(fecha=df["fecha"] + pd.Timedelta(days=1)))
    assert "[REVISAR]" in _salida(capsys)


def test_unidad_de_tiempo_equivocada(verificar, df, capsys):
    ms = df["fecha"].astype("int64") // 10**6
    malo = df.assign(fecha=pd.to_datetime(ms, utc=True))  # sin unit="ms": pandas lee nanosegundos, 1970
    verificar.velas(malo)
    assert "unidad" in _salida(capsys)


def test_vela_en_curso(verificar, df, capsys):
    hoy = pd.Timestamp.now(tz="UTC").normalize()
    extra = df.iloc[[-1]].assign(fecha=hoy)
    verificar.velas(pd.concat([df[df["fecha"] < hoy], extra], ignore_index=True))
    assert "no ha cerrado" in _salida(capsys)


def test_precios_como_texto(verificar, df, capsys):
    verificar.velas(df.astype({"close": str}))
    assert "números" in _salida(capsys)


def test_fecha_como_indice_tambien_vale(verificar, df, capsys):
    verificar.velas(df.set_index("fecha"))
    assert _salida(capsys).startswith("[OK]")


def _medidas(df):
    r = np.log(df["close"]).diff().dropna()
    return (r.std() * np.sqrt(365), df["close"].pct_change().min(),
            int((((r - r.mean()) / r.std()).abs() > 4).sum()))


def test_riesgo_correcto(verificar, df, capsys):
    verificar.riesgo(df, *_medidas(df))
    assert _salida(capsys).startswith("[OK]")


def test_riesgo_errores_tipicos(verificar, df, capsys):
    vol, peor, n4 = _medidas(df)
    verificar.riesgo(df, vol * 100, peor, n4)
    assert "porcentaje" in _salida(capsys)
    verificar.riesgo(df, vol / np.sqrt(365), peor, n4)
    assert "DIARIA" in _salida(capsys)
    verificar.riesgo(df, vol / np.sqrt(365) * np.sqrt(252), peor, n4)
    assert "252" in _salida(capsys)


def test_prediccion(verificar, capsys):
    verificar.prediccion(0.50, 0.49)
    assert "es la lección" in _salida(capsys)
    verificar.prediccion(0.71, 0.49)
    assert "sospechoso" in _salida(capsys)
    verificar.prediccion(50, 49)  # en porcentaje: se normaliza
    assert "es la lección" in _salida(capsys)


def test_cadena(verificar, capsys):
    verificar.cadena(True, False)
    assert _salida(capsys).startswith("[OK]")
    verificar.cadena(True, True)
    assert "SIN recalcular" in _salida(capsys)


def test_nombre_desconocido_explicado_sin_traza(gen, capsys):
    """Verificador ejecutado antes que el codigo, o Claude uso otro nombre."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        shell.run_cell(gen.PREPARACION)
        capsys.readouterr()
        r = shell.run_cell("verificar.velas(df)")
        salida = _salida(capsys)
        assert isinstance(r.error_in_exec, NameError)
        assert "Python no conoce «df»" in salida and "variable llamada df" in salida
        assert "Traceback" not in salida
        shell.run_cell("x = json.dumps({})")  # nombre ajeno a los verificadores: falta un import
        assert "Suele faltar un import" in _salida(capsys)
    finally:
        InteractiveShell.clear_instance()


def test_cuadernos_al_dia_con_el_generador(gen):
    """alumno.ipynb y docente.ipynb deben ser lo que produce el generador."""
    for nombre, docente in (("alumno.ipynb", False), ("docente.ipynb", True)):
        en_disco = nbformat.read(CLASE / nombre, 4)
        fuentes = [c.source for c in en_disco.cells]
        assert fuentes == [c.source for c in gen.construir(docente).cells], nombre


def test_el_alumno_no_trae_soluciones(gen):
    alumno = gen.construir(False)
    codigo = "\n".join(c.source for c in alumno.cells if c.cell_type == "code")
    assert "RandomForestClassifier" not in codigo and "hashlib" not in codigo
    assert alumno.cells[1].metadata.get("cellView") == "form"
