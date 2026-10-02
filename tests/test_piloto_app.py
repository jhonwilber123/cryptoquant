"""La app del piloto, pulsando sus controles como lo haria el usuario.

`streamlit.testing.v1.AppTest` ejecuta la app sin navegador. Los precios son
sinteticos (sin red) y el diario vive en una carpeta temporal. La tabla de la
cartera es un editor que AppTest no sabe rellenar: su contenido se prepara
guardando antes la cartera, que es lo que la app lee al abrirse.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cryptoquant.piloto import diario, mercado, motor  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "cryptoquant" / "piloto" / "app.py"
CARTERA = {"BTC": 1.0, "ETH": 5.0, "USDT": 1000.0}


def _cierres(n: int = 500, desplome: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1),
                        periods=n, freq="D", tz="UTC")
    out = {}
    for i, (a, v) in enumerate({"BTC": 0.03, "ETH": 0.04, "SOL": 0.06}.items()):
        r = v * rng.standard_normal(n)
        if desplome:
            r[-30:] = -0.02
        out[a] = 100.0 * (i + 1) * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=idx)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """abrir(tenencias, ajustes, mezcla) -> AppTest ya ejecutado."""
    monkeypatch.setenv("CRYPTOQUANT_PILOTO", str(tmp_path))
    estado = {"cierres": _cierres(), "precios": None, "fallo": None}

    def cierres(activos):
        if estado["fallo"] is not None:
            raise estado["fallo"]
        c = estado["cierres"]
        for a in activos:
            if a not in c:
                raise ValueError(f"{a}: no hay par {a}/USDT en Binance o no hay conexion (BadSymbol)")
        return c[list(activos)], []

    def precios(activos):
        if estado["precios"] is not None:
            return estado["precios"]
        return {a: float(estado["cierres"][a].iloc[-1]) for a in activos if a in estado["cierres"]}

    monkeypatch.setattr(mercado, "cierres", cierres)
    monkeypatch.setattr(mercado, "precios_ahora", precios)
    st.cache_data.clear()

    def abrir(tenencias=None, ajustes=None, mezcla=None):
        if tenencias is not None:
            diario.guardar_cartera(tenencias, mezcla,
                                   {**asdict(motor.Ajustes()), "reparto": "actual", **(ajustes or {})})
        return AppTest.from_file(str(APP), default_timeout=120).run()

    abrir.estado, abrir.dir = estado, tmp_path
    return abrir


def _textos(elementos) -> str:
    return " | ".join(str(e.value) for e in elementos)


def _cifra(texto: str) -> float:
    """'1 318 USDT' -> 1318.0; '−27 %' -> -27.0; '63 % al año' -> 63.0"""
    return float(re.sub(r"[^0-9,−-]", "", texto.split("%")[0]).replace(",", ".").replace("−", "-"))


def _metrica(at, etiqueta: str) -> str:
    return next(m.value for m in at.metric if m.label == etiqueta)


def _boton(at, empieza: str):
    return next(b for b in at.button if b.label.startswith(empieza))


def _sin_fallos(at):
    assert not at.exception, [e.value for e in at.exception]


# --------------------------------------------------------------------------
def test_primera_vez(app):
    at = app()
    _sin_fallos(at)
    assert "Anote en la barra lateral" in _textos(at.info)
    assert not at.error


def test_cartera_guardada(app):
    at = app(CARTERA)
    _sin_fallos(at)
    assert not at.error
    assert [t.label for t in at.tabs] == ["Hoy", "Tiempos duros", "La regla en el pasado", "Mi diario",
                                          "Cómo funciona"]
    etiquetas = {m.label for m in at.metric}
    assert {"Valor de la cartera", "En cripto ahora", "Recomendado en cripto",
            "Movimiento previsto de su mezcla", "Caída máxima"} <= etiquetas
    assert "cambios sin guardar" not in _textos(at.sidebar.warning)
    tabla = at.sidebar.dataframe[0].value
    assert dict(zip(tabla["Activo"], tabla["Cantidad"])) == CARTERA
    total = sum(CARTERA[a] * float(app.estado["cierres"][a].iloc[-1]) for a in ("BTC", "ETH")) + 1000
    assert _cifra(_metrica(at, "Valor de la cartera")) == pytest.approx(total, abs=1)


def test_niveles_de_riesgo(app):
    at = app(CARTERA)
    prudente = _cifra(_metrica(at, "Recomendado en cripto"))
    at.radio[0].set_value("Decidido").run()
    _sin_fallos(at)
    assert _cifra(_metrica(at, "Recomendado en cripto")) > prudente
    assert "cambios sin guardar" in _textos(at.sidebar.warning)
    at.radio[0].set_value("A medida").run()
    objetivo = next(s for s in at.slider if s.label == "Volatilidad anual objetivo")
    objetivo.set_value(5).run()
    _sin_fallos(at)
    assert _cifra(_metrica(at, "Recomendado en cripto")) < prudente


def test_maximo_en_cripto(app):
    at = app(CARTERA)
    at.radio[0].set_value("Decidido").run()
    next(s for s in at.slider if s.label == "Máximo en cripto").set_value(20).run()
    _sin_fallos(at)
    assert _cifra(_metrica(at, "Recomendado en cripto")) <= 20


def test_repartos(app):
    at = app(CARTERA)
    at.radio[1].set_value("igual_riesgo").run()
    _sin_fallos(at)
    assert not at.error
    at.radio[1].set_value("medida").run()
    _sin_fallos(at)
    assert "Si no suma 100" in _textos(at.sidebar.caption)
    assert len(at.sidebar.dataframe) == 2
    mezcla = at.sidebar.dataframe[1].value
    assert set(mezcla["Activo"]) == {"BTC", "ETH"}


def test_reparto_a_medida_guardado(app):
    at = app(CARTERA, {"reparto": "medida"}, mezcla={"SOL": 1.0})
    _sin_fallos(at)
    ordenes = at.dataframe[0].value
    assert "SOL" in set(ordenes["Moneda"]), "debe proponer comprar la moneda del reparto"


def test_freno(app):
    at = app(CARTERA)
    at.toggle[0].set_value(False).run()
    _sin_fallos(at)
    assert "Freno: desactivado." in _textos(at.markdown)
    assert not any(s.label.startswith("Empieza a frenar") for s in at.slider)
    at.toggle[0].set_value(True).run()
    next(s for s in at.slider if s.label.startswith("Empieza a frenar")).set_value((20, 20)).run()
    _sin_fallos(at)
    assert not at.error, "inicio y tope iguales deben corregirse solos"


def test_freno_activo_tras_un_desplome(app):
    app.estado["cierres"] = _cierres(desplome=True)
    c = app.estado["cierres"]
    diario.anotar_foto({"BTC": 1.0}, {"BTC": float(c["BTC"].iloc[-60])})
    fotos = diario.carpeta() / diario.FOTOS
    foto = json.loads(fotos.read_text(encoding="utf-8"))
    foto["fecha_utc"] = f"{c.index[-60].date()}T12:00:00+00:00"   # hace 60 dias
    fotos.write_text(json.dumps(foto) + "\n", encoding="utf-8")
    at = app({"BTC": 1.0})
    _sin_fallos(at)
    assert "Freno activo" in _textos(at.error)
    assert "activo" in _textos(at.markdown)


def test_guardar(app):
    at = app(CARTERA)
    at.radio[0].set_value("Moderado").run()
    _boton(at, "Guardar").click().run()
    _sin_fallos(at)
    guardada = json.loads((app.dir / "cartera.json").read_text(encoding="utf-8"))
    assert guardada["ajustes"]["objetivo"] == 0.25 and guardada["tenencias"] == CARTERA
    assert "cambios sin guardar" not in _textos(at.sidebar.warning)
    lineas = (app.dir / "fotos.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1 and json.loads(lineas[0])["tenencias"] == CARTERA
    # Guardar sin cambiar las tenencias no anade otra foto.
    _boton(at, "Guardar").click().run()
    assert len((app.dir / "fotos.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len(at.tabs) == 5 and at.dataframe  # el diario ya muestra su foto


def test_guardar_aunque_el_plan_no_se_pueda_calcular(app):
    """Sin red (o con una moneda sin historia) el plan falla, pero lo anotado se guarda.

    Antes el guardado iba despues del calculo: si este fallaba, la app cortaba
    antes de llegar y el boton se perdia sin aviso.
    """
    at = app(CARTERA)
    app.estado["fallo"] = ValueError("sin conexion")
    st.cache_data.clear()
    at.radio[0].set_value("Moderado")
    _boton(at, "Guardar").click().run()
    _sin_fallos(at)
    guardada = json.loads((app.dir / "cartera.json").read_text(encoding="utf-8"))
    assert guardada["ajustes"]["objetivo"] == 0.25
    assert "guardados" in _textos(at.toast)
    assert "sin conexion" in _textos(at.error)
    lineas = (app.dir / "fotos.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1 and json.loads(lineas[0])["tenencias"] == CARTERA


def test_validar_con_los_4_metodos(app):
    at = app(CARTERA)
    _boton(at, "Comprobar el mal día").click().run()
    _sin_fallos(at)
    tabla = next(d.value for d in at.dataframe if "Método" in d.value.columns)
    assert list(tabla["Método"]) == ["Histórico", "Normal", "t de Student", "GARCH-t"]


def test_actualizar_precios(app):
    at = app(CARTERA)
    _boton(at, "Actualizar precios").click().run()
    _sin_fallos(at)
    assert not at.error


def test_moneda_desconocida(app):
    at = app({"XYZ": 1.0, "USDT": 10.0})
    _sin_fallos(at)
    assert "XYZ" in _textos(at.error)


def test_historia_corta(app):
    app.estado["cierres"] = _cierres(40)
    at = app(CARTERA)
    _sin_fallos(at)
    assert "60 dias" in _textos(at.error)


def test_todo_en_estables(app):
    at = app({"USDT": 1000.0})
    _sin_fallos(at)
    assert "Todo está en estables" in _textos(at.info)
    assert "Puede pasar" in _textos(at.info)


def test_sin_precio_de_ahora(app):
    app.estado["precios"] = {}
    at = app(CARTERA)
    _sin_fallos(at)
    assert "Sin precio de este momento" in _textos(at.warning)


def test_fallo_inesperado_no_rompe_la_app(app):
    at = app(CARTERA)
    antes = (app.dir / "cartera.json").read_bytes()
    app.estado["fallo"] = RuntimeError("fallo raro")
    st.cache_data.clear()
    at.run()
    assert "Algo ha fallado" in _textos(at.error)
    assert (app.dir / "cartera.json").read_bytes() == antes


def test_cartera_danada(app):
    app.dir.mkdir(parents=True, exist_ok=True)
    (app.dir / "cartera.json").write_text("{roto", encoding="utf-8")
    at = app()
    _sin_fallos(at)
    assert "estaba danado" in _textos(at.warning)
    assert "Anote en la barra lateral" in _textos(at.info)


def test_ajustes_guardados_raros(app):
    at = app(CARTERA, {"objetivo": 7})
    _sin_fallos(at)
    assert "Ajustes guardados no válidos" in _textos(at.warning)
    # Valores validos para el motor pero fuera del rango de los controles: se acotan.
    at = app(CARTERA, {"freno_ventana": 3000, "banda": 0.5, "minimo_orden": 50_000})
    _sin_fallos(at)
    assert not at.error


# --------------------------------------------------------------------------
# Publicado en un servidor: nada de la cartera sin la contraseña
# --------------------------------------------------------------------------
def test_en_un_servidor_nada_se_ve_sin_la_contrasena(app, monkeypatch):
    from cryptoquant.piloto import acceso

    monkeypatch.setattr(acceso, "ESPERA_TRAS_FALLO", 0.0)
    monkeypatch.setenv("PILOTO_EXIGIR_CLAVE", "1")
    monkeypatch.setenv("PILOTO_CLAVE_HASH", acceso.crear("la contraseña del servidor", iteraciones=1000))
    at = app(CARTERA)
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Contraseña"]
    assert not at.metric and not at.tabs and not at.dataframe
    assert not at.sidebar.header   # ni siquiera la barra lateral con la cartera

    at.text_input[0].input("una contraseña cualquiera")
    at.button[0].click().run()
    assert "Contraseña incorrecta" in _textos(at.error)
    assert not at.metric

    at.text_input[0].input("la contraseña del servidor")
    at.button[0].click().run()
    _sin_fallos(at)
    assert any(m.label == "Valor de la cartera" for m in at.metric)
    assert "Su cartera" in [h.value for h in at.sidebar.header]
    at.run()   # la sesion queda abierta: no vuelve a pedirla
    assert not at.text_input and any(m.label == "Valor de la cartera" for m in at.metric)


def test_un_servidor_sin_contrasena_no_abre_la_app(app, monkeypatch):
    monkeypatch.setenv("PILOTO_EXIGIR_CLAVE", "1")
    monkeypatch.delenv("PILOTO_CLAVE_HASH", raising=False)
    at = app(CARTERA)
    assert not at.exception
    assert "no tiene contraseña" in _textos(at.error)
    assert not at.metric and not at.text_input
