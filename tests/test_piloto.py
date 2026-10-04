"""Piloto de riesgo: la cartera real del usuario.

Aqui se decide dinero de verdad, asi que los tests buscan sobre todo que el
piloto nunca recomiende algo imposible o peligroso (vender lo que no se tiene,
comprar con dinero que no hay, pasar del maximo de exposicion), que los datos
raros o danados den un mensaje claro y no un plan absurdo, y que nada mire al
futuro. Ninguno usa la red.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptoquant.laboratorio import control, datos
from cryptoquant.piloto import acceso, diario, mercado, motor, sesion, vista

RAIZ = Path(__file__).resolve().parents[1]
VOLS = {"BTC": 0.03, "ETH": 0.04, "SOL": 0.06, "ADA": 0.05}


def _cierres(n: int = 500, activos=("BTC", "ETH", "SOL"), semilla: int = 0,
             inicio: str = "2024-01-01") -> pd.DataFrame:
    """Precios diarios correlacionados, con colas t, como las cripto."""
    rng = np.random.default_rng(semilla)
    idx = pd.date_range(inicio, periods=n, freq="D", tz="UTC")
    comun = rng.standard_t(4, n)
    out = {}
    for i, a in enumerate(activos):
        e = 0.6 * comun + 0.8 * rng.standard_t(4, n)
        out[a] = 100.0 * (i + 1) * np.exp(np.cumsum(VOLS.get(a, 0.05) * e / 1.4))
    return pd.DataFrame(out, index=idx)


def _precios(c: pd.DataFrame) -> dict[str, float]:
    return {a: float(c[a].iloc[-1]) for a in c}


@pytest.fixture(autouse=True)
def aislado(tmp_path, monkeypatch):
    """Diario en una carpeta temporal y sin red: si algo intenta salir, falla."""
    monkeypatch.setenv("CRYPTOQUANT_PILOTO", str(tmp_path / "piloto"))

    def sin_red(*a, **k):
        raise AssertionError("un test ha intentado usar la red")
    monkeypatch.setattr(datos, "descargar_velas", sin_red)
    monkeypatch.setattr(mercado.requests, "get", sin_red)
    return tmp_path / "piloto"


# --------------------------------------------------------------------------
# Ajustes
# --------------------------------------------------------------------------
def test_ajustes_de_fabrica_validos():
    aj = motor.Ajustes().validar()
    assert (aj.objetivo, aj.banda, aj.freno_ventana) == (0.15, 0.05, 180)


@pytest.mark.parametrize("campo, valor", [
    ("objetivo", 0), ("objetivo", -0.1), ("objetivo", 5), ("objetivo", float("nan")),
    ("objetivo", float("inf")), ("objetivo", "0.15"), ("objetivo", True),
    ("max_exposicion", 0), ("max_exposicion", 1.5), ("banda", -0.01), ("banda", 1),
    ("freno_minimo", 1.5), ("freno_ventana", 1), ("freno_ventana", 2.5), ("freno_ventana", 5000),
    ("minimo_orden", -1), ("coste", 0.2), ("lam", 1), ("lam", 0), ("freno", "si"), ("freno", 1),
])
def test_ajustes_fuera_de_rango(campo, valor):
    with pytest.raises(ValueError):
        motor.Ajustes(**{campo: valor}).validar()


def test_freno_que_empieza_despues_de_su_tope():
    with pytest.raises(ValueError, match="antes de su tope"):
        motor.Ajustes(freno_inicio=0.3, freno_tope=0.2).validar()


def test_ajustes_guardados():
    assert motor.Ajustes.desde(None) == motor.Ajustes()
    assert motor.Ajustes.desde({"objetivo": 0.25, "de_una_version_futura": 1}).objetivo == 0.25
    with pytest.raises(ValueError):
        motor.Ajustes.desde([1, 2])
    with pytest.raises(ValueError):
        motor.Ajustes.desde({"banda": "mucha"})


@pytest.mark.parametrize("caida, esperado", [
    (0.0, 1.0), (-0.05, 1.0), (-0.10, 1.0), (-0.175, 0.5), (-0.20, 1 / 3), (-0.25, 0.25), (-0.9, 0.25)])
def test_factor_freno(caida, esperado):
    assert motor.factor_freno(caida, motor.Ajustes()) == pytest.approx(esperado)
    assert motor.factor_freno(caida, motor.Ajustes(freno=False)) == 1.0


# --------------------------------------------------------------------------
# Cantidades y mezclas raras
# --------------------------------------------------------------------------
@pytest.mark.parametrize("malo", [-1, float("nan"), float("inf"), -float("inf"), True, "3", None])
def test_cantidades_raras(malo):
    with pytest.raises(ValueError, match="BTC"):
        motor.separar({"BTC": malo, "USDT": 10})


def test_cantidades_validas():
    cripto, efectivo = motor.separar({"BTC": np.float64(0.5), "ETH": 0, "SOL": np.int64(3),
                                      "USDT": 100, "USDC": 50.5})
    assert cripto == {"BTC": 0.5, "SOL": 3.0} and efectivo == 150.5


def test_cantidades_enormes():
    c = _cierres()
    with pytest.raises(ValueError, match="demasiado grandes"):
        motor.decidir({"BTC": 1e308, "ETH": 1e308}, _precios(c), c)


@pytest.mark.parametrize("mezcla, mensaje", [
    ({"USDT": 1}, "sin estables"), ({"BTC": -1}, "finitos"), ({}, "vacia"), ({"BTC": 0}, "vacia"),
    ({"../X": 1}, "no valido"), ({"BTC": float("nan")}, "finitos"), ({"BTC": 1e308, "ETH": 1e308}, "grandes"),
])
def test_mezclas_invalidas(mezcla, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        motor.normalizar_mezcla(mezcla)


def test_mezcla_normalizada():
    w = motor.normalizar_mezcla({"btc": 1, "BTC": 1, "eth": 2})
    assert w.to_dict() == {"BTC": 0.5, "ETH": 0.5}


def test_estable_en_la_mezcla_da_el_error_correcto():
    """No 'sin precio para USDT': el problema es la mezcla."""
    c = _cierres()
    with pytest.raises(ValueError, match="sin estables"):
        motor.decidir({"BTC": 1, "USDT": 100}, _precios(c), c, mezcla={"BTC": 1, "USDT": 1})


def test_historia_corta_dice_que_moneda():
    c = _cierres()
    c.loc[c.index[:-40], "SOL"] = np.nan
    with pytest.raises(ValueError, match=r"60 dias.*SOL \(40 dias\)"):
        motor.decidir({"BTC": 1, "SOL": 1}, _precios(c), c)


def test_moneda_sin_historico():
    c = _cierres()
    with pytest.raises(ValueError, match="sin precios para: ADA"):
        motor.decidir({"BTC": 1, "ADA": 5}, {**_precios(c), "ADA": 1.0}, c)


@pytest.mark.parametrize("p", [None, 0, -5, float("nan"), float("inf")])
def test_precio_actual_invalido(p):
    c = _cierres()
    precios = _precios(c)
    precios["BTC"] = p
    with pytest.raises(ValueError, match="BTC: sin precio actual"):
        motor.decidir({"BTC": 1, "USDT": 100}, precios, c)


def test_precios_no_positivos_en_el_historico():
    c = _cierres()
    c.iloc[100, 0] = 0
    with pytest.raises(ValueError, match="nulos o negativos"):
        motor.decidir({"BTC": 1}, _precios(c), c)


# --------------------------------------------------------------------------
# La decision
# --------------------------------------------------------------------------
def test_invariantes_de_dinero():
    """Nunca vender lo que no se tiene, comprar sin dinero o pasar del maximo."""
    c = _cierres()
    precios = _precios(c)
    rng = np.random.default_rng(7)
    probados = 0
    for _ in range(300):
        ten = {a: float(rng.choice([0.0, rng.uniform(0, 20)])) for a in c}
        ten["USDT"] = float(rng.choice([0.0, rng.uniform(0, 5000)]))
        if sum(ten.values()) == 0:
            continue
        aj = motor.Ajustes(objetivo=float(rng.uniform(0.02, 2)), max_exposicion=float(rng.uniform(0.1, 1)),
                           banda=float(rng.uniform(0, 0.2)), freno=bool(rng.integers(2)),
                           minimo_orden=float(rng.choice([0, 10, 1000])))
        cripto = {a: u for a, u in ten.items() if a != "USDT" and u > 0}
        mezcla = None if cripto else {"BTC": 1.0, "SOL": 2.0}
        p = motor.decidir(ten, precios, c, mezcla, -float(rng.uniform(0, 0.6)), aj)
        o = p.ordenes
        assert 0 <= p.exposicion_objetivo <= aj.max_exposicion + 1e-12
        assert (o["unidades_objetivo"] >= 0).all()
        assert (o["diferencia"] >= -o["unidades"] - 1e-12).all(), "vende mas de lo que tiene"
        assert p.mover <= p.efectivo + 1e-6 * p.total, "compra sin dinero"
        assert p.cripto_objetivo == pytest.approx(p.exposicion_objetivo * p.total)
        assert o["importe"].sum() == pytest.approx(p.mover, abs=1e-6 * p.total)
        assert not o.loc[o["ejecutar"], "importe"].abs().lt(aj.minimo_orden).any()
        if not p.actuar:
            assert not o["ejecutar"].any()
        # Lo que se ensena para ejecutar, no solo el plan: con las ventas que de
        # verdad se hacen y el efectivo que hay, sin vender de mas.
        ej = o[o["ejecutar"]]
        assert ej["importe_orden"].sum() <= p.efectivo + 1e-6 * p.total, "las ordenes piden dinero que no hay"
        assert (-ej["cantidad_orden"] <= ej["unidades"] + 1e-12).all(), "la orden vende mas de lo que tiene"
        assert not ej["importe_orden"].abs().lt(aj.minimo_orden).any()
        assert (o.loc[~o["ejecutar"], "importe_orden"] == 0).all()
        assert (o["importe_orden"].abs() <= o["importe"].abs() + 1e-9).all()
        probados += 1
    assert probados > 250


def test_la_decision_solo_usa_cierres():
    """El precio de ahora valora; la volatilidad sale de los cierres."""
    c = _cierres()
    ten = {"BTC": 1.0, "ETH": 2.0, "USDT": 500.0}
    p1 = motor.decidir(ten, _precios(c), c)
    p2 = motor.decidir(ten, {a: v * 1.3 for a, v in _precios(c).items()}, c)
    assert p1.sigma_prevista == pytest.approx(p2.sigma_prevista)
    assert p1.escala_vol == pytest.approx(p2.escala_vol)
    assert p2.total > p1.total
    w = motor.normalizar_mezcla({a: ten[a] * _precios(c)[a] for a in ("BTC", "ETH")})
    lr = motor.retornos_log(c, ["BTC", "ETH"])
    assert p1.sigma_prevista == pytest.approx(motor.sigma_mezcla(lr, w).iloc[-1])


def test_un_activo_coincide_con_el_laboratorio():
    c = _cierres()
    lr = np.log(c["BTC"]).diff().dropna()
    esperado = control.sigma_ewma(lr).iloc[-1] * math.sqrt(365)
    assert motor.decidir({"BTC": 1}, _precios(c), c).sigma_prevista == pytest.approx(esperado)


def test_objetivo_alto_topa_en_el_maximo():
    c = _cierres()
    p = motor.decidir({"BTC": 1, "USDT": 1000}, _precios(c), c,
                      ajustes=motor.Ajustes(objetivo=4.9, max_exposicion=0.8))
    assert p.exposicion_objetivo == pytest.approx(0.8)


def test_sin_movimiento_se_admite_el_maximo():
    """Una moneda que no se mueve no aporta riesgo: no hay que venderla."""
    c = _cierres()
    c["PLANA"] = 1.0
    p = motor.decidir({"PLANA": 100, "USDT": 100}, {"PLANA": 1.0}, c)
    assert p.sigma_prevista == 0 and p.exposicion_objetivo == 1.0 and p.motivo != "reducir"


def _en_objetivo(c, aj=None):
    """Una cartera que ya esta en lo recomendado."""
    p = motor.decidir({"BTC": 1.0, "USDT": 1000.0}, _precios(c), c, ajustes=aj)
    px = _precios(c)["BTC"]
    return {"BTC": p.cripto_objetivo / px, "USDT": p.total - p.cripto_objetivo}, p


def test_banda():
    c = _cierres()
    ten, p0 = _en_objetivo(c)
    p = motor.decidir(ten, _precios(c), c)
    assert not p.actuar and p.motivo == "en_banda" and abs(p.mover) < 1e-6 * p.total
    # Un 3 % de la cartera de mas en cripto: dentro de la banda del 5 %.
    extra = 0.03 * p0.total / _precios(c)["BTC"]
    p = motor.decidir({**ten, "BTC": ten["BTC"] + extra, "USDT": ten["USDT"] - 0.03 * p0.total},
                      _precios(c), c)
    assert p.motivo == "en_banda"
    # Un 20 %: hay que reducir.
    extra = 0.2 * p0.total / _precios(c)["BTC"]
    p = motor.decidir({"BTC": ten["BTC"] + extra, "USDT": max(0.0, ten["USDT"] - 0.2 * p0.total)},
                      _precios(c), c)
    assert p.actuar and p.motivo == "reducir" and p.mover < 0


def test_aumentar_y_orden_minima():
    c = _cierres()
    p = motor.decidir({"BTC": 0.0, "USDT": 1000.0}, _precios(c), c, mezcla={"BTC": 1})
    assert p.motivo == "aumentar" and p.ordenes.loc["BTC", "ejecutar"]
    assert p.ordenes.loc["BTC", "unidades_objetivo"] == pytest.approx(p.cripto_objetivo / _precios(c)["BTC"])
    # Con 15 USDT, la compra no llega al minimo de Binance.
    p = motor.decidir({"USDT": 15.0}, _precios(c), c, mezcla={"BTC": 1})
    assert not p.ordenes["ejecutar"].any()


def test_freno_activo():
    c = _cierres()
    ten, _ = _en_objetivo(c)
    p = motor.decidir(ten, _precios(c), c, caida=-0.20)
    assert p.factor_freno == pytest.approx(1 / 3) and p.motivo == "freno" and p.mover < 0
    with pytest.raises(ValueError):
        motor.decidir(ten, _precios(c), c, caida=0.1)


def test_todo_en_estables():
    c = _cierres()
    with pytest.raises(ValueError, match="todo esta en estables"):
        motor.decidir({"USDT": 1000}, _precios(c), c)
    with pytest.raises(ValueError, match="no tiene valor"):
        motor.decidir({"USDT": 0, "BTC": 0}, _precios(c), c, mezcla={"BTC": 1})


def test_reequilibrar_dentro_de_la_cripto():
    c = _cierres()
    ten, _ = _en_objetivo(c)
    p = motor.decidir(ten, _precios(c), c, mezcla={"ETH": 1})
    assert p.ordenes.loc["BTC", "unidades_objetivo"] == 0
    assert p.ordenes.loc["ETH", "importe"] > 0 and p.ordenes.loc["BTC", "importe"] < 0


def test_ventas_omitidas_no_pagan_compras():
    """Una venta por debajo del minimo no se hace: la compra que pagaba se recorta al efectivo."""
    c = _cierres(activos=("BTC", "ETH", "SOL", "ADA"))
    px = _precios(c)
    polvo = {"BTC": 9 / px["BTC"], "ETH": 9 / px["ETH"], "SOL": 9 / px["SOL"], "ADA": 200 / px["ADA"]}
    aj = motor.Ajustes(objetivo=4.9, banda=0.0)
    p = motor.decidir({**polvo, "USDT": 0.0}, px, c, {"ADA": 1}, ajustes=aj)
    assert p.ordenes.loc["ADA", "importe"] == pytest.approx(27)      # el plan ideal no cambia
    assert not p.ordenes["ejecutar"].any() and p.ordenes.loc["ADA", "recortada"]
    # Con 20 USDT se compra lo que alcanza, y no mas.
    p = motor.decidir({**polvo, "USDT": 20.0}, px, c, {"ADA": 1}, ajustes=aj)
    o = p.ordenes.loc["ADA"]
    assert o["ejecutar"] and o["recortada"] and o["importe_orden"] == pytest.approx(20)
    assert o["cantidad_orden"] == pytest.approx(20 / px["ADA"])
    # Si queda por debajo del minimo, no se ensena.
    p = motor.decidir({**polvo, "USDT": 6.0}, px, c, {"ADA": 1}, ajustes=aj)
    assert not p.ordenes["ejecutar"].any()
    # Con efectivo de sobra no se recorta nada.
    p = motor.decidir({**polvo, "USDT": 1000.0}, px, c, {"ADA": 1},
                      ajustes=motor.Ajustes(objetivo=4.9, banda=0.0, max_exposicion=0.5))
    assert not p.ordenes["recortada"].any()
    assert p.ordenes.loc["ADA", "importe_orden"] == pytest.approx(p.ordenes.loc["ADA", "importe"])


def test_al_descartar_una_compra_pequena_las_demas_recuperan_su_efectivo():
    """Descartar una compra por debajo del minimo devuelve su parte del efectivo a las demas."""
    importe = pd.Series({"A": 100.0, "B": 11.0})
    orden = motor._al_efectivo(importe, pd.Series(True, index=importe.index),
                               efectivo=100.0, minimo=10.0)
    assert orden["B"] == 0 and orden["A"] == pytest.approx(100.0)


def test_las_ventas_pagan_compras_descontado_su_coste():
    """Vender 100 USDT de una moneda no deja 100 para comprar otra."""
    importe = pd.Series({"A": -100.0, "B": 100.0})
    orden = motor._al_efectivo(importe, pd.Series(True, index=importe.index),
                               efectivo=0.0, minimo=10.0, coste=0.0015)
    assert orden["A"] == -100.0 and orden["B"] == pytest.approx(99.85)


def test_contribucion_al_riesgo():
    c = _cierres()
    p = motor.decidir({"BTC": 1, "SOL": 1, "USDT": 100}, _precios(c), c)
    assert p.riesgo["peso_riesgo"].sum() == pytest.approx(1)
    assert p.riesgo.loc["SOL", "vol_anual"] > p.riesgo.loc["BTC", "vol_anual"]


def test_igual_riesgo():
    w = motor.pesos_igual_riesgo(_cierres(), ["BTC", "SOL"])
    assert sum(w.values()) == pytest.approx(1) and w["BTC"] > w["SOL"]


def test_repartos():
    c = _cierres()
    assert motor.candidatas_mezcla("actual", {"ETH": 2}, None) == {"ETH": 2}
    assert motor.candidatas_mezcla("actual", {}, None) == {"BTC": 1.0}
    assert motor.candidatas_mezcla("actual", {}, {"SOL": 3, "ETH": 1}) == {"ETH": 0.25, "SOL": 0.75}
    assert motor.candidatas_mezcla("medida", {"BTC": 1}, {"ETH": 1}) == {"ETH": 1.0}
    with pytest.raises(ValueError):
        motor.candidatas_mezcla("medida", {"BTC": 1}, None)
    with pytest.raises(ValueError):
        motor.candidatas_mezcla("medida", {"BTC": 1}, {"USDC": 1})
    with pytest.raises(ValueError):
        motor.candidatas_mezcla("otro", {"BTC": 1}, None)
    assert motor.resolver_mezcla("actual", {"BTC": 1}, {"BTC": 1}, c) is None
    assert motor.resolver_mezcla("actual", {}, {"BTC": 1.0}, c) == {"BTC": 1.0}
    assert set(motor.resolver_mezcla("igual_riesgo", {"BTC": 1, "ETH": 1}, {"BTC": 1, "ETH": 1}, c)) == {"BTC", "ETH"}


# --------------------------------------------------------------------------
# La regla en el pasado
# --------------------------------------------------------------------------
def test_simular_coincide_con_el_laboratorio():
    c = _cierres(800)
    t = motor.simular(c, {"BTC": 1}, motor.Ajustes(freno=False, banda=0, coste=0))
    ref = control.control_volatilidad(np.log(c["BTC"]).diff().dropna(), coste=0)
    assert len(t) == len(ref)
    np.testing.assert_allclose(t["estrategia"], ref["estrategia"].reindex(t.index), rtol=1e-12)
    np.testing.assert_allclose(t["exposicion"], ref["exposicion"].reindex(t.index), rtol=1e-12)


def test_simular_es_causal():
    c = _cierres(700)
    aj = motor.Ajustes()
    completa = motor.simular(c, {"BTC": 1, "ETH": 1}, aj)
    corta = motor.simular(c.iloc[:500], {"BTC": 1, "ETH": 1}, aj)
    pd.testing.assert_frame_equal(completa.loc[corta.index].drop(columns="pasiva_igual_vol"),
                                  corta.drop(columns="pasiva_igual_vol"))


def test_simular_limites_banda_y_coste():
    c = _cierres(800)
    sin_banda = motor.simular(c, {"BTC": 1, "SOL": 1}, motor.Ajustes(banda=0))
    con_banda = motor.simular(c, {"BTC": 1, "SOL": 1}, motor.Ajustes(banda=0.05, max_exposicion=0.7))
    assert con_banda.attrs["operaciones"] < sin_banda.attrs["operaciones"] / 3
    assert con_banda["exposicion"].between(0, 0.7 + 1e-12).all()
    gratis = motor.simular(c, {"BTC": 1}, motor.Ajustes(coste=0))
    cara = motor.simular(c, {"BTC": 1}, motor.Ajustes(coste=0.01))
    assert (1 + cara["estrategia"]).prod() < (1 + gratis["estrategia"]).prod()


def _desplome() -> pd.DataFrame:
    """Subida tranquila, desplome del 60 % y un ano plano."""
    rng = np.random.default_rng(3)
    r = np.concatenate([rng.normal(0.001, 0.01, 200), np.full(30, -0.03), rng.normal(0, 0.004, 400)])
    idx = pd.date_range("2022-01-01", periods=len(r), freq="D", tz="UTC")
    return pd.DataFrame({"BTC": 100 * np.exp(np.cumsum(r))}, index=idx)


def test_el_freno_se_suelta():
    c = _desplome()
    t = motor.simular(c, {"BTC": 1}, motor.Ajustes(freno_inicio=0.02, freno_tope=0.05,
                                                   freno_minimo=0.25, freno_ventana=90))
    assert t["freno"].min() < 1
    assert t["freno"].iloc[-1] == 1.0, "tras la ventana, el freno debe soltarse"


def test_sin_minimo_ni_ventana_el_freno_se_atasca():
    """Por eso el piloto no usa la regla del sistema tal cual."""
    c = _desplome()
    t = motor.simular(c, {"BTC": 1}, motor.Ajustes(freno_inicio=0.02, freno_tope=0.05,
                                                   freno_minimo=0.0, freno_ventana=3650))
    assert t["freno"].iloc[-1] == 0.0 and t["exposicion"].iloc[-1] == 0.0


# --------------------------------------------------------------------------
# La caida de la cuenta
# --------------------------------------------------------------------------
def _foto(dia: pd.Timestamp, ten: dict, hora: str = "12:00:00") -> dict:
    return {"fecha_utc": f"{dia.date()}T{hora}+00:00", "tenencias": ten}


def test_valor_cuenta_sin_ingresos_ni_retiros():
    c = _cierres(120)
    d, P = c.index, c["BTC"]
    # Todo en BTC; a mitad se retira la mitad: el valor sigue al precio.
    v = motor.valor_cuenta([_foto(d[10], {"BTC": 2.0}), _foto(d[50], {"BTC": 1.0})], c[["BTC"]])
    np.testing.assert_allclose(v.to_numpy(), (P[d[10]:] / P[d[10]]).to_numpy(), rtol=1e-12)
    # Mitad y mitad, y luego un ingreso de 10 000 USDT: no es una ganancia.
    p0 = float(P[d[10]])
    fotos = [_foto(d[10], {"BTC": 1.0, "USDT": p0}), _foto(d[40], {"BTC": 1.0, "USDT": p0 + 10_000})]
    v = motor.valor_cuenta(fotos, c[["BTC"]])
    esperado_40 = (P[d[40]] + p0) / (2 * p0)
    assert v[d[40]] == pytest.approx(esperado_40)
    v41 = esperado_40 * (P[d[41]] + p0 + 10_000) / (P[d[40]] + p0 + 10_000)
    assert v[d[41]] == pytest.approx(v41)


def test_valor_cuenta_solo_efectivo_e_ingresos():
    c = _cierres(60)
    v = motor.valor_cuenta([_foto(c.index[5], {"USDT": 1000}), _foto(c.index[20], {"USDT": 5000})], c)
    assert (v == 1.0).all()


def test_valor_cuenta_precio_de_ahora_y_huecos():
    c = _cierres(60)[["BTC"]]
    fotos = [_foto(c.index[5], {"BTC": 1.0})]
    ultimo = float(c["BTC"].iloc[-1])
    v = motor.valor_cuenta(fotos, c, ahora={"BTC": ultimo * 0.8})
    assert v.iloc[-1] == pytest.approx(v.iloc[-2] * 0.8)
    # Un dia sin precio no inventa ni una ganancia ni una perdida.
    c2 = c.copy()
    c2.iloc[30, 0] = np.nan
    v2 = motor.valor_cuenta(fotos, c2)
    assert np.isfinite(v2).all()
    assert motor.valor_cuenta([], c).empty and motor.caida_vigente(pd.Series(dtype=float), 30) == 0.0


def test_foto_sin_zona_horaria_se_toma_como_utc():
    c = _cierres(40)[["BTC"]]
    v = motor.valor_cuenta([{"fecha_utc": f"{c.index[3].date()}T08:00:00", "tenencias": {"BTC": 1}}], c)
    assert v.index[0] == c.index[3]


def test_caida_vigente_con_ventana():
    idx = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    v = pd.Series(np.r_[np.linspace(1, 2, 50), np.full(150, 1.5)], index=idx)
    assert motor.caida_vigente(v, 180) == pytest.approx(-0.25)
    assert motor.caida_vigente(v, 60) == 0.0   # el maximo ya salio de la ventana


# --------------------------------------------------------------------------
# Tiempos duros
# --------------------------------------------------------------------------
def _semana_negra(n: int = 400) -> pd.DataFrame:
    precio = np.full(n, 100.0)
    precio[200:207] = 100 * 0.5 ** (np.arange(1, 8) / 7)
    precio[207:] = 50.0
    return pd.DataFrame({"BTC": precio}, index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


def test_escenarios():
    c = _semana_negra()
    e = motor.escenarios(c, {"BTC": 1000.0}, 1000.0)
    assert e.loc["Peor semana", "perdida"] == pytest.approx(-500)
    assert e.loc["Peor semana", "perdida_pct"] == pytest.approx(-0.25)
    assert e.loc["Peor día", "perdida"] == pytest.approx(-1000 * (1 - 0.5 ** (1 / 7)))
    assert e.loc["Peor año", "perdida"] == pytest.approx(-500)
    corta = motor.escenarios(c.iloc[-100:], {"BTC": 1000.0}, 0)
    assert np.isnan(corta.loc["Peor año", "perdida"]) and corta.loc["Peor semana", "perdida"] == 0
    vacia = motor.escenarios(c, {}, 1000.0)
    assert (vacia["perdida"] == 0).all()


def test_mal_dia_y_proximo_mes():
    c = _cierres()
    todo = motor.mal_dia(c, {"BTC": 1000.0}, 0.0)
    mitad = motor.mal_dia(c, {"BTC": 500.0}, 500.0)
    assert mitad["var"] == pytest.approx(todo["var"] / 2) and todo["cvar"] >= todo["var"] > 0
    assert motor.mal_dia(c, {}, 1000.0)["var"] == 0
    m_todo, m_poco = (motor.proximo_mes(c, {"SOL": 1000.0}, 0.0),
                      motor.proximo_mes(c, {"SOL": 200.0}, 800.0))
    assert m_poco["prob_caida_20"] <= m_todo["prob_caida_20"]
    assert m_poco["retorno_p5"] > m_todo["retorno_p5"]
    assert motor.proximo_mes(c, {}, 100.0)["prob_perdida"] == 0


# --------------------------------------------------------------------------
# Diario: datos personales en disco
# --------------------------------------------------------------------------
def test_diario_ida_y_vuelta(aislado):
    diario.guardar_cartera({"btc": 0.5, "USDT": 100}, {"ETH": 1}, {"objetivo": 0.25})
    d, aviso = diario.leer_cartera()
    assert aviso is None and d["tenencias"] == {"BTC": 0.5, "USDT": 100.0} and d["mezcla"] == {"ETH": 1.0}
    assert not list(aislado.glob("*.tmp")), "la escritura atomica no debe dejar temporales"


@pytest.mark.parametrize("contenido", ["{no es json", "[1, 2]", '{"tenencias": {"BTC": -1}}',
                                       '{"tenencias": {"../../x": 1}}', '{"tenencias": {"BTC": 1}, "ajustes": 3}'])
def test_cartera_danada_se_aparta(aislado, contenido):
    aislado.mkdir(parents=True, exist_ok=True)
    (aislado / "cartera.json").write_text(contenido, encoding="utf-8")
    d, aviso = diario.leer_cartera()
    assert d == {} and "se aparto" in aviso
    assert not (aislado / "cartera.json").exists()
    assert len(list(aislado.glob("cartera.danada-*.json"))) == 1


@pytest.mark.parametrize("ten", [{"../../etc/x": 1}, {"BTC": -1}, {"BTC": float("inf")}, {"BTC": True},
                                 {"BTC": "mucho"}, "BTC=1"])
def test_diario_rechaza_tenencias_raras(aislado, ten):
    with pytest.raises(ValueError):
        diario.guardar_cartera(ten, None, {})
    assert not (aislado / "cartera.json").exists()


def test_fotos_danadas_no_bloquean(aislado):
    diario.anotar_foto({"BTC": 1}, {"BTC": 50_000})
    ruta = aislado / "fotos.jsonl"
    with ruta.open("a", encoding="utf-8") as f:
        f.write('{"fecha_utc": "2026-01-01T00:00:00+00:00", "tenencias": {"BTC": -1}}\n')
        f.write('{"fecha_utc": "2026-01-01T00:00:00", "tenencias": {"BTC": 1}}\n')   # sin zona horaria
        f.write('{"fecha_utc": "2026-01-02T00:00:00+00:00", "tenen')                  # corte a media linea
    diario.anotar_foto({"BTC": 2}, {"BTC": 51_000})
    fotos, avisos = diario.leer_fotos()
    assert [f["tenencias"]["BTC"] for f in fotos] == [1.0, 2.0]
    assert avisos and "3 linea" in avisos[0]


def test_foto_con_precio_invalido(aislado):
    with pytest.raises(ValueError):
        diario.anotar_foto({"BTC": 1}, {"BTC": 0})
    with pytest.raises(ValueError):
        diario.anotar_foto({"BTC": 1}, {"../x": 5})
    assert diario.leer_fotos() == ([], [])


def test_carpeta_por_variable_de_entorno(aislado):
    assert diario.carpeta() == aislado and aislado.is_dir()


# --------------------------------------------------------------------------
# Mercado, sin red
# --------------------------------------------------------------------------
def _velas_falsas(hasta: pd.Timestamp, n: int = 100) -> pd.DataFrame:
    idx = pd.date_range(end=hasta, periods=n, freq="D", tz="UTC", name="date")
    close = np.linspace(100, 120, n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1.0}, index=idx)


AHORA = pd.Timestamp("2026-09-27 03:00", tz="UTC")


def test_ultimo_cierre():
    assert mercado.ultimo_cierre(AHORA) == pd.Timestamp("2026-09-26", tz="UTC")
    assert mercado.ultimo_cierre(pd.Timestamp("2026-09-27 00:00:01", tz="UTC")).day == 26


def test_velas_de_cache_si_estan_al_dia(aislado, monkeypatch):
    (aislado / "velas").mkdir(parents=True)
    datos.exportar_csv(_velas_falsas(mercado.ultimo_cierre(AHORA)), aislado / "velas" / "BTCUSDT_1d.csv")
    v = mercado.velas("btc", AHORA)                          # sin red: el fixture falla si baja
    assert v.index[-1] == mercado.ultimo_cierre(AHORA) and not v.attrs.get("aviso")


def test_velas_viejas_se_renuevan(aislado, monkeypatch):
    (aislado / "velas").mkdir(parents=True)
    datos.exportar_csv(_velas_falsas(AHORA.normalize() - pd.Timedelta(days=5)),
                       aislado / "velas" / "BTCUSDT_1d.csv")
    llamadas = []
    monkeypatch.setattr(datos, "descargar_velas",
                        lambda s, tf, desde: llamadas.append(s) or _velas_falsas(mercado.ultimo_cierre(AHORA)))
    v = mercado.velas("BTC", AHORA)
    assert llamadas == ["BTC/USDT"] and v.index[-1] == mercado.ultimo_cierre(AHORA)
    assert datos.leer_csv(aislado / "velas" / "BTCUSDT_1d.csv").index[-1] == v.index[-1]


def test_sin_red_usa_las_guardadas_y_avisa(aislado):
    (aislado / "velas").mkdir(parents=True)
    datos.exportar_csv(_velas_falsas(AHORA.normalize() - pd.Timedelta(days=5)),
                       aislado / "velas" / "BTCUSDT_1d.csv")
    v = mercado.velas("BTC", AHORA)
    assert "no respondio" in v.attrs["aviso"]


def test_sin_red_ni_guardadas(aislado):
    with pytest.raises(ValueError, match="no hay par XYZ/USDT en Binance o no hay conexion"):
        mercado.velas("XYZ", AHORA)
    assert not (aislado / "velas" / "XYZUSDT_1d.csv").exists()


def test_una_descarga_mas_corta_no_borra_lo_guardado(aislado, monkeypatch):
    """Si la API falla y solo quedan los archivos mensuales, los ultimos dias no se pierden."""
    (aislado / "velas").mkdir(parents=True)
    hasta = mercado.ultimo_cierre(AHORA) - pd.Timedelta(days=3)
    datos.exportar_csv(_velas_falsas(hasta), aislado / "velas" / "BTCUSDT_1d.csv")
    monkeypatch.setattr(datos, "descargar_velas",
                        lambda *a, **k: _velas_falsas(hasta - pd.Timedelta(days=27), n=300))
    v = mercado.velas("BTC", AHORA)
    assert v.index[-1] == hasta and v.index.is_unique and v.index.is_monotonic_increasing
    assert len(v) == 300 + 27                        # lo bajado y los dias que solo estaban guardados
    assert datos.leer_csv(aislado / "velas" / "BTCUSDT_1d.csv").index[-1] == hasta
    assert "solo dio velas hasta el" in v.attrs["aviso"]


def test_cache_danada_se_vuelve_a_bajar(aislado, monkeypatch):
    (aislado / "velas").mkdir(parents=True)
    (aislado / "velas" / "BTCUSDT_1d.csv").write_text("basura,sin,columnas\n1,2,3\n")
    monkeypatch.setattr(datos, "descargar_velas", lambda *a, **k: _velas_falsas(mercado.ultimo_cierre(AHORA)))
    assert len(mercado.velas("BTC", AHORA)) == 100


@pytest.mark.parametrize("malo", ["../../etc", "BTC/../../x", "C:\\x", "", "B", "A" * 20, "btc usdt"])
def test_ticker_no_llega_a_una_ruta(aislado, malo):
    with pytest.raises(ValueError, match="simbolo no valido"):
        mercado.velas(malo, AHORA)
    assert not (aislado / "velas").exists()


def test_precios_ahora(monkeypatch):
    class Resp:
        def __init__(self, cuerpo, estado=200):
            self.cuerpo, self.estado = cuerpo, estado

        def raise_for_status(self):
            if self.estado != 200:
                raise RuntimeError(self.estado)

        def json(self):
            if isinstance(self.cuerpo, Exception):
                raise self.cuerpo
            return self.cuerpo

    respuestas = {"BTCUSDT": Resp({"price": "84000.5"}), "ETHUSDT": Resp({"price": "0"}),
                  "SOLUSDT": Resp({"price": "nan"}), "ADAUSDT": Resp(ValueError("no json")),
                  "XRPUSDT": Resp({}, 400)}

    def get(url, params, timeout):
        assert url.startswith("https://data-api.binance.vision/") and timeout <= 10
        if params["symbol"] == "DOGEUSDT":
            raise ConnectionError("sin red")
        return respuestas[params["symbol"]]
    monkeypatch.setattr(mercado.requests, "get", get)
    assert mercado.precios_ahora(["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"]) == {"BTC": 84000.5}
    assert mercado.precios_ahora(["../x"]) == {}


# --------------------------------------------------------------------------
# Sesion: el recorrido completo, con datos inyectados
# --------------------------------------------------------------------------
def _fuentes(c: pd.DataFrame, precios: dict | None = None, fallan=()):
    pedidos = []

    def cierres_de(activos):
        pedidos.append(tuple(activos))
        if any(a in fallan for a in activos):
            raise ValueError("no hay par")
        return c[list(activos)], []
    return cierres_de, (lambda activos: dict(precios if precios is not None else _precios(c))), pedidos


def test_sesion_completa():
    c = _cierres()
    cierres_de, precios_de, pedidos = _fuentes(c)
    s = sesion.calcular({"BTC": 1, "USDT": 500}, "actual", None, [], motor.Ajustes(), cierres_de, precios_de)
    assert s.plan.total == pytest.approx(_precios(c)["BTC"] + 500)
    assert s.mezcla is None and s.mezcla_efectiva.keys() == {"BTC"} and pedidos == [("BTC",)]
    assert s.caida == 0 and s.sin_precio_ahora == []


def test_sesion_sin_precio_de_ahora_usa_el_cierre():
    c = _cierres()
    cierres_de, precios_de, _ = _fuentes(c, precios={})
    s = sesion.calcular({"BTC": 1}, "actual", None, [], motor.Ajustes(), cierres_de, precios_de)
    assert s.sin_precio_ahora == ["BTC"] and s.precios["BTC"] == c["BTC"].iloc[-1]


def test_sesion_monedas_de_fotos_viejas():
    c = _cierres()
    fotos = [_foto(c.index[-100], {"SOL": 5.0}), _foto(c.index[-50], {"ADA": 5.0}),
             _foto(c.index[-10], {"BTC": 1.0})]
    cierres_de, precios_de, pedidos = _fuentes(c, fallan=("ADA",))
    s = sesion.calcular({"BTC": 1}, "actual", None, fotos, motor.Ajustes(), cierres_de, precios_de)
    assert ("SOL",) in pedidos and ("ADA",) in pedidos
    assert any("ADA" in a for a in s.avisos) and "SOL" in s.cierres
    assert len(s.valor) > 1


def test_sesion_todo_en_estables_y_vacia():
    c = _cierres()
    cierres_de, precios_de, _ = _fuentes(c)
    s = sesion.calcular({"USDT": 1000}, "actual", None, [], motor.Ajustes(), cierres_de, precios_de)
    assert s.candidatas == {"BTC": 1.0} and s.plan.motivo == "aumentar"
    s = sesion.calcular({"USDT": 1000}, "igual_riesgo", {"ETH": 1, "SOL": 1}, [], motor.Ajustes(),
                        cierres_de, precios_de)
    assert set(s.mezcla) == {"ETH", "SOL"}
    with pytest.raises(ValueError, match="vacia"):
        sesion.calcular({"USDT": 0}, "actual", None, [], motor.Ajustes(), cierres_de, precios_de)


# --------------------------------------------------------------------------
# Formato y tablas del usuario
# --------------------------------------------------------------------------
def test_formatos():
    assert vista.num(1234567.891, 2) == "1\u00a0234\u00a0567,89"
    assert vista.num(-5) == "−5" and vista.num(float("nan")) == "—"
    assert vista.pct(0.1234, 1) == "12,3\u00a0%" and vista.pct(-0.5) == "−50\u00a0%"
    assert vista.usdt(float("inf")) == "—" and vista.usdt(10) == "10\u00a0USDT"
    assert vista.cantidad(0.00012300) == "0,000123" and vista.cantidad(2.5) == "2,5"
    assert vista.cantidad(500.0) == "500" and vista.cantidad(12345.678) == "12 345,68"
    assert vista.cantidad(0.03372963) == "0,03372963" and vista.cantidad(-1.25) == "−1,25"
    assert vista.precio(84000) == "84\u00a0000,00" and vista.precio(0.000012345678) == "1,23457e-05"


def test_leer_tabla():
    t = pd.DataFrame({"Activo": ["btc", "BTC", " eth ", None, "", "USDT"],
                      "Cantidad": [0.5, 0.25, 1.0, None, None, 100.0]})
    assert vista.leer_tabla(t) == {"BTC": 0.75, "ETH": 1.0, "USDT": 100.0}


@pytest.mark.parametrize("activo, cant, mensaje", [
    ("BTC", None, "falta la cantidad"), ("BTC", -1, "positivo"), ("BTC", float("inf"), "positivo"),
    ("BTC", "abc", "no es un número"), ("../x", 1, "no valido"), (None, 5, "sin moneda")])
def test_leer_tabla_errores(activo, cant, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        vista.leer_tabla(pd.DataFrame({"Activo": [activo], "Cantidad": [cant]}, dtype=object))


# --------------------------------------------------------------------------
# Seguridad
# --------------------------------------------------------------------------
def test_sin_claves_ni_ordenes():
    """Solo recomienda: ni claves de API, ni endpoints privados, ni HTML sin escapar."""
    codigo = "\n".join(p.read_text(encoding="utf-8") for p in (RAIZ / "cryptoquant" / "piloto").glob("*.py"))
    prohibido = re.compile(r"api_?key|secret|create_?order|private|unsafe_allow_html|/api/v3/order",
                           re.IGNORECASE)
    assert not prohibido.findall(codigo)


def test_datos_personales_fuera_de_git():
    if not (RAIZ / ".git").exists() or shutil.which("git") is None:
        pytest.skip("sin repositorio git: la carpeta se bajo como ZIP o falta git")
    for ruta in ("data/piloto/cartera.json", "data/piloto/fotos.jsonl", "data/piloto/velas/BTCUSDT_1d.csv"):
        r = subprocess.run(["git", "-c", "safe.directory=*", "check-ignore", "-q", ruta], cwd=RAIZ)
        assert r.returncode == 0, f"{ruta} no esta ignorado por git"


def test_la_app_solo_escucha_en_este_equipo(monkeypatch, capsys):
    from cryptoquant import cli

    ordenes = []
    monkeypatch.setattr(subprocess, "call", lambda orden: ordenes.append(orden) or 0)
    assert cli.main(["piloto", "--sin-navegador", "--puerto", "8765"]) == 0
    assert cli.main(["piloto", "--sin-navegador", "--red", "--puerto", "8765"]) == 0
    local, red = ordenes
    assert local[local.index("--server.address") + 1] == "127.0.0.1"
    assert local[local.index("--server.headless") + 1] == "true"
    assert local[local.index("--browser.gatherUsageStats") + 1] == "false"
    assert red[red.index("--server.address") + 1] == "0.0.0.0"
    assert "cualquiera en su red" in capsys.readouterr().out


def test_puerto_libre_salta_uno_ocupado():
    import socket

    from cryptoquant import cli

    with socket.socket() as ocupado:
        ocupado.bind(("127.0.0.1", 0))
        ocupado.listen()
        puerto = ocupado.getsockname()[1]
        assert cli._puerto_libre(puerto, "127.0.0.1") != puerto


# --------------------------------------------------------------------------
# Consola
# --------------------------------------------------------------------------
def test_texto_sin_cartera(capsys):
    from cryptoquant import cli

    assert cli.main(["piloto", "--texto"]) == 2
    assert "No hay cartera guardada" in capsys.readouterr().out


def test_texto_con_cartera(monkeypatch, capsys):
    from cryptoquant import cli

    c = _cierres()
    monkeypatch.setattr(mercado, "cierres", lambda activos: (c[list(activos)], []))
    monkeypatch.setattr(mercado, "precios_ahora", lambda activos: _precios(c))
    diario.guardar_cartera({"BTC": 3.0, "USDT": 10.0}, None, {**vars(motor.Ajustes()), "reparto": "actual"})
    assert cli.main(["piloto", "--texto"]) == 0
    salida = capsys.readouterr().out
    assert "ACCION: pasar a estables" in salida and "Vender" in salida and "BTC" in salida


def test_texto_explica_las_ordenes_que_no_se_ensenan(monkeypatch, capsys):
    """Como la app: dice que ordenes se omiten por pequenas y cuales se recortan por el efectivo."""
    from cryptoquant import cli

    c = _cierres(activos=("BTC", "ETH", "SOL", "ADA"))
    px = _precios(c)
    monkeypatch.setattr(mercado, "cierres", lambda activos: (c[list(activos)], []))
    monkeypatch.setattr(mercado, "precios_ahora", lambda activos: px)
    ten = {"BTC": 9 / px["BTC"], "ETH": 9 / px["ETH"], "SOL": 9 / px["SOL"], "ADA": 200 / px["ADA"],
           "USDT": 20.0}
    diario.guardar_cartera(ten, {"ADA": 1}, {"objetivo": 4.9, "banda": 0.0, "reparto": "medida"})
    assert cli.main(["piloto", "--texto"]) == 0
    salida = capsys.readouterr().out
    assert re.search(r"Comprar +[\d.]+ ADA +20\.00 USDT", salida)      # el plan ideal pedia 47
    assert not re.search(r"Comprar .* 47\.00 USDT", salida)
    assert "menos de 10.00 USDT" in salida and "BTC, ETH, SOL" in salida
    assert "efectivo" in salida and "ADA" in salida


def test_texto_con_datos_imposibles(monkeypatch, capsys):
    from cryptoquant import cli

    c = _cierres(30)       # historia corta
    monkeypatch.setattr(mercado, "cierres", lambda activos: (c[list(activos)], []))
    monkeypatch.setattr(mercado, "precios_ahora", lambda activos: _precios(c))
    diario.guardar_cartera({"BTC": 1.0}, None, {})
    assert cli.main(["piloto", "--texto"]) == 2
    assert "ERROR: hacen falta 60 dias" in capsys.readouterr().out


def test_el_json_guardado_es_legible(aislado):
    diario.guardar_cartera({"BTC": 1}, None, {"objetivo": 0.15})
    json.loads((aislado / "cartera.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Contraseña para publicarlo en un servidor
# --------------------------------------------------------------------------
def test_la_contrasena_se_guarda_derivada_y_se_comprueba():
    guardada = acceso.crear("una contraseña larga ñ", iteraciones=1000)
    assert "contraseña" not in guardada and "$" not in guardada   # Compose interpreta los '$'
    assert re.fullmatch(r"pbkdf2_sha256:1000:[0-9a-f]{32}:[0-9a-f]{64}", guardada)
    assert acceso.comprobar("una contraseña larga ñ", guardada)
    assert not acceso.comprobar("una contraseña larga n", guardada)
    assert acceso.crear("una contraseña larga ñ", iteraciones=1000) != guardada   # sal nueva cada vez


@pytest.mark.parametrize("guardada", ["", "basura", "pbkdf2_sha256:1000:zz:00", "md5:1000:00:00",
                                      "pbkdf2_sha256:0:00:00", "pbkdf2_sha256:1000:00"])
def test_una_derivada_mal_escrita_no_deja_entrar(guardada):
    assert not acceso.comprobar("cualquier contraseña", guardada)


def test_contrasena_demasiado_corta():
    with pytest.raises(ValueError, match="al menos 12"):
        acceso.crear("corta")


def test_configuracion_del_acceso():
    assert acceso.configuracion({}) == (None, False)
    assert acceso.configuracion({"PILOTO_CLAVE_HASH": "  ", "PILOTO_EXIGIR_CLAVE": "1"}) == (None, True)
    assert acceso.configuracion({"PILOTO_CLAVE_HASH": " x:1:2:3\n"}) == ("x:1:2:3", False)


def test_crear_clave_desde_la_consola(monkeypatch, capsys):
    """Por la salida solo sale la derivada: desplegar.ps1 la recoge tal cual."""
    import getpass

    from cryptoquant import cli

    respuestas = iter(["una contraseña larga", "una contraseña larga"])
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": next(respuestas))
    assert cli.main(["piloto", "--crear-clave"]) == 0
    salida = capsys.readouterr().out.strip()
    assert re.fullmatch(r"pbkdf2_sha256:\d+:[0-9a-f]+:[0-9a-f]+", salida)
    assert acceso.comprobar("una contraseña larga", salida)

    respuestas = iter(["una contraseña larga", "otra distinta larga"])
    assert cli.main(["piloto", "--crear-clave"]) == 1
    capt = capsys.readouterr()
    assert capt.out == "" and "no coinciden" in capt.err
