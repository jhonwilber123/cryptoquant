"""Casos borde y entradas hostiles del modulo de riesgo.

Cada test corresponde a un fallo real encontrado al explorar entradas
degeneradas: un VaR "aceptado" con cero dias de datos, cantidades infinitas
en la cartera, un JSON con NaN para el panel, una senal que solo hacia
trampa en el segundo activo, y un registro de ideas que se podia recortar a
mano para ablandar el umbral sin que nadie lo notara.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptoquant import config as cfgmod
from cryptoquant.config import Config
from cryptoquant.riesgo import cartera, evidencia, informe, pruebas, var
from tests.test_riesgo import _momentum, _retornos, _velas_con_memoria


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(cfgmod, "DATA_DIR", tmp_path / "data")
    return tmp_path


# --------------------------------------------------------------------------
# VaR y pruebas
# --------------------------------------------------------------------------
def test_var_rechaza_metodo_alpha_y_muestra_minima():
    x = _retornos(300)["A"].to_numpy()
    for kw in ({"metodo": "magia"}, {"metodo": "normal", "alpha": 1.0},
               {"metodo": "normal", "alpha": 0.3}):
        with pytest.raises(ValueError):
            var.var_es(x, **kw)
    with pytest.raises(ValueError):
        var.var_es(x[:1], "normal")


def test_sin_exposicion_no_hay_riesgo():
    v = var.var_siguiente(_retornos(300), pd.Series({"A": 0.0, "B": 0.0}))
    assert (v.to_numpy() == 0).all()


def test_pesos_nan_cuentan_como_cero():
    R = _retornos(400)
    t = var.pronosticar(R, pd.Series({"A": np.nan, "B": 1.0}), metodos=("normal",))
    ref = var.pronosticar(R, pd.Series({"A": 0.0, "B": 1.0}), metodos=("normal",))
    pd.testing.assert_frame_equal(t, ref)


def test_sin_datos_suficientes_no_se_acepta_ningun_metodo():
    vacia = pd.DataFrame(columns=["exposicion", "retorno", "var_normal", "es_normal"], dtype=float)
    ev = pruebas.evaluar(vacia, 0.95, metodos=("normal",))
    assert ev.loc["normal", "n"] == 0 and not ev.loc["normal", "aceptado"]
    t = var.pronosticar(_retornos(300), pd.Series({"A": 1.0, "B": 0.0}), metodos=("normal",))
    ev = pruebas.evaluar(t, 0.95, metodos=("normal",))
    assert not ev.loc["normal", "suficiente"] and not ev.loc["normal", "aceptado"]


# --------------------------------------------------------------------------
# Cartera
# --------------------------------------------------------------------------
@pytest.mark.parametrize("malo", ["BTC=nan", "BTC=inf", "BTC=1e400", "=5", "../x=1", "B TC=1",
                                  "BTC=", "BTC=0x10", "", ",,,", "BTC=1;DROP TABLE"])
def test_parsear_rechaza_entradas_hostiles(malo):
    with pytest.raises(ValueError):
        cartera.parsear(malo)


def test_cartera_casos_degenerados():
    closes = 100 * np.exp(_retornos(700).cumsum())
    with pytest.raises(ValueError, match="solo tiene efectivo"):
        cartera.analizar({"USDT": 100}, closes)
    with pytest.raises(ValueError, match="solo tiene efectivo"):
        cartera.analizar({"A": 0.0, "USDT": 5}, closes)
    with pytest.raises(ValueError, match="sin precios"):
        cartera.analizar({"ZZZ": 1.0}, closes)
    with pytest.raises(ValueError, match="dias de historia"):
        cartera.analizar({"A": 1.0}, closes.iloc[-40:])
    for kw in ({"alpha": 1.5}, {"alpha": 0.2}, {"horizonte": 0}):
        with pytest.raises(ValueError):
            cartera.analizar({"A": 1.0}, closes, n_sim=10, **kw)


def test_cartera_con_historia_breve_no_se_declara_fiable():
    closes = 100 * np.exp(_retornos(200).cumsum())
    r = cartera.analizar({"A": 1.0}, closes, n_sim=200)
    assert not r.validacion["aceptado"].any()
    assert not r.validacion["suficiente"].any()


def test_cli_cartera_errores_sin_traceback(capsys):
    from cryptoquant.cli import main

    assert main(["cartera", "--tengo", "BTC=nan"]) == 2
    assert "finito" in capsys.readouterr().out
    assert main(["cartera", "--tengo", "USDT=500"]) == 2
    for malo in (["--confianza", "1.5"], ["--dias", "0"], ["--riesgo", "0"]):
        with pytest.raises(SystemExit):
            main(["cartera", "--tengo", "BTC=1", *malo])


# --------------------------------------------------------------------------
# Estado para el panel
# --------------------------------------------------------------------------
def test_estado_json_estricto_y_atomico(sandbox):
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    t = pd.DataFrame({"exposicion": 1.0, "retorno": [0.01, np.nan, -0.02, 0.0, 0.01],
                      "var_garch_t": 0.02, "var_historico": [np.nan, 0.02, 0.02, np.inf, 0.02]},
                     index=idx)
    inf = informe.InformeRiesgo(idx[-1], {"modelo": t},
                                pd.DataFrame({"cola_ratio": [np.nan], "alpha": [0.95]}),
                                pd.DataFrame({"x": [np.inf]}), pd.DataFrame({"var": [np.nan]}), 0.0)
    informe._exportar(inf, Config(), 0.95)
    texto = (sandbox / "reports" / "riesgo" / "estado.json").read_text(encoding="utf-8")

    def rechaza(c):
        raise ValueError(c)
    json.loads(texto, parse_constant=rechaza)  # ni NaN ni Infinity: JSON estricto
    assert informe.leer_estado()["alpha"] == 0.95
    assert not list((sandbox / "reports" / "riesgo").glob("*.tmp"))


@pytest.mark.parametrize("contenido", ["{roto", "[1, 2]", "", "null"])
def test_estado_corrupto_no_rompe_el_panel(sandbox, contenido):
    (informe.carpeta() / "estado.json").write_text(contenido, encoding="utf-8")
    assert informe.leer_estado() is None


def test_estado_binario_no_rompe_el_panel(sandbox):
    (informe.carpeta() / "estado.json").write_bytes(b"\xff\xfe\x00basura")
    assert informe.leer_estado() is None


def test_panel_escapa_texto_del_estado_de_riesgo():
    from cryptoquant.dashboard.render import render_html

    ataque = "</script><script>alert(1)</script>"
    html = render_html({"risk": {"validacion": [{"cartera": ataque}]}})
    assert ataque not in html


# --------------------------------------------------------------------------
# Filtro de evidencia
# --------------------------------------------------------------------------
def test_evidencia_admite_booleanos_nan_y_otro_indice(sandbox):
    velas = {k: _velas_con_memoria(seed=s) for k, s in (("X", 1), ("Y", 2))}
    base = evidencia.evaluar_senal("m", _momentum, velas, registrar=False)
    como_bool = evidencia.evaluar_senal("b", lambda df: _momentum(df) > 0, velas, registrar=False)
    salteada = evidencia.evaluar_senal("s", lambda df: _momentum(df).iloc[::2], velas,
                                       registrar=False)
    escalada = evidencia.evaluar_senal("e", lambda df: _momentum(df) * 7, velas, registrar=False)
    assert escalada.eventos == base.eventos
    assert 0 < salteada.eventos < base.eventos
    assert 0 < como_bool.eventos < base.eventos


def test_evidencia_trampa_en_un_solo_activo(sandbox):
    velas = {k: _velas_con_memoria(seed=s) for k, s in (("X", 1), ("Y", 2))}
    marca = velas["Y"]["close"].iloc[0]

    def tramposa(df):
        if df["close"].iloc[0] == marca:
            return np.sign(df["close"].shift(-1) - df["close"]).fillna(0.0)
        return _momentum(df)
    v = evidencia.evaluar_senal("tramposa", tramposa, velas, registrar=False)
    assert not v.causal and "Y" in v.motivos[0] and not v.pasa


@pytest.mark.parametrize("kw", [{"horizonte": 0}, {"horizonte": -3}, {"horizonte": 2.5},
                                {"horizonte": True}, {"frac_prueba": 0.0},
                                {"frac_prueba": 1.5}])
def test_evidencia_rechaza_parametros_imposibles(sandbox, kw):
    with pytest.raises(ValueError):
        evidencia.evaluar_senal("x", _momentum, {"X": _velas_con_memoria()}, **kw)


def test_evidencia_generador_roto_o_sin_velas(sandbox):
    with pytest.raises(ValueError, match="el generador"):
        evidencia.evaluar_senal("x", lambda df: 1 / 0, {"X": _velas_con_memoria()})
    with pytest.raises(ValueError, match="Series"):
        evidencia.evaluar_senal("x", lambda df: [1, 2], {"X": _velas_con_memoria()})
    with pytest.raises(ValueError, match="no hay velas"):
        evidencia.evaluar_senal("x", _momentum, {})


def _tres_ideas():
    velas = {"X": _velas_con_memoria(n=600)}
    for nombre in ("a", "b", "c"):
        evidencia.evaluar_senal(nombre, _momentum, velas)
    return evidencia.ruta_registro()


@pytest.mark.parametrize("alteracion", ["borrar", "editar", "reordenar", "basura", "resellar",
                                        "truncar_al_principio"])
def test_registro_detecta_alteraciones(sandbox, alteracion):
    ruta = _tres_ideas()
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    if alteracion == "borrar":            # quitar una idea para ablandar el umbral
        del lineas[1]
    elif alteracion == "editar":          # maquillar un veredicto
        lineas[0] = lineas[0].replace('"pasa": false', '"pasa": true')
    elif alteracion == "reordenar":
        lineas[0], lineas[1] = lineas[1], lineas[0]
    elif alteracion == "basura":
        lineas.insert(1, "esto no es json")
    elif alteracion == "resellar":        # editar y recalcular su propio hash
        e = json.loads(lineas[1])
        e["p_valor"] = 0.0001
        e["hash"] = evidencia._sello(e)
        lineas[1] = json.dumps(e, ensure_ascii=False)
    elif alteracion == "truncar_al_principio":
        del lineas[0]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    with pytest.raises(evidencia.RegistroCorrupto):
        evidencia.ideas_evaluadas()
    with pytest.raises(evidencia.RegistroCorrupto):
        evidencia.evaluar_senal("d", _momentum, {"X": _velas_con_memoria(n=600)})


def test_registro_intacto_se_verifica(sandbox):
    _tres_ideas()
    assert evidencia.ideas_evaluadas() == 3


def test_cli_evidencia_registro_roto_sin_traceback(sandbox, capsys):
    from cryptoquant.cli import main

    ruta = _tres_ideas()
    ruta.write_text(ruta.read_text(encoding="utf-8").splitlines()[1] + "\n", encoding="utf-8")
    assert main(["evidencia", "--registro"]) == 2
    assert "cadena de hashes" in capsys.readouterr().out
