"""Piloto de riesgo: la interfaz (Streamlit).

Se abre con `python -m cryptoquant piloto`. El calculo esta en `sesion` y
`motor`; aqui solo se leen los controles y se muestra el resultado.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from cryptoquant.laboratorio import control  # noqa: E402
from cryptoquant.piloto import diario, mercado, motor, sesion  # noqa: E402
from cryptoquant.piloto.vista import cantidad, fecha, leer_tabla, num, pct, precio, usdt  # noqa: E402
from cryptoquant.riesgo import cartera as rc  # noqa: E402
from cryptoquant.riesgo.cartera import EFECTIVO  # noqa: E402

NIVELES = {"Prudente": 0.15, "Moderado": 0.25, "Decidido": 0.40}
REPARTOS = {"actual": "Como lo tengo ahora", "igual_riesgo": "Igual riesgo por moneda",
            "medida": "A medida"}
METODOS = {"historico": "Histórico", "normal": "Normal", "t": "t de Student", "garch_t": "GARCH-t"}
SERIES = ["Con la regla", "Comprar y mantener", "Mantener a igual volatilidad"]
COLORES = dict(zip(SERIES, ["#2c7fb8", "#e6550d", "#31a354"]), **{"Su cartera": "#2c7fb8"})


def entre(x: float, lo: float, hi: float) -> float:
    """Un ajuste guardado fuera del rango de un control lo romperia: se acota."""
    return min(max(x, lo), hi)


# --------------------------------------------------------------------------
# Datos en cache: las velas cambian una vez al dia; el precio, cada minuto
# --------------------------------------------------------------------------
@st.cache_data(ttl=3 * 3600, show_spinner=False)
def _cierres(activos: tuple[str, ...], dia: str) -> tuple[pd.DataFrame, list[str]]:
    return mercado.cierres(activos)


@st.cache_data(ttl=60, show_spinner=False)
def _precios(activos: tuple[str, ...], minuto: str) -> dict[str, float]:
    return mercado.precios_ahora(activos)


@st.cache_data(show_spinner=False)
def _simular(cierres: pd.DataFrame, mezcla: tuple, ajustes: tuple) -> pd.DataFrame:
    return motor.simular(cierres, dict(mezcla), motor.Ajustes(**dict(ajustes)))


@st.cache_data(show_spinner=False)
def _analizar(tenencias: tuple, cierres: pd.DataFrame):
    return rc.analizar(dict(tenencias), cierres)


def lineas(ancho: pd.DataFrame, titulo_y: str, formato: str, log: bool = False,
           alto: int = 280) -> alt.Chart:
    d = ancho.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    # Fechas sin nombres de meses: Vega los escribiria en ingles.
    largo = len(d) > 1 and (d.index.max() - d.index.min()).days > 400
    eje_x = alt.Axis(format="%Y", tickCount="year") if largo else alt.Axis(format="%d-%m-%Y")
    d = d.rename_axis("fecha").reset_index().melt("fecha", var_name="serie", value_name="valor")
    series = list(ancho.columns)
    color = alt.Color("serie:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=0),
                      scale=alt.Scale(domain=series, range=[COLORES.get(s, "#888") for s in series]))
    escala = alt.Scale(type="log", nice=False) if log else alt.Scale(zero=False)
    return (alt.Chart(d).mark_line(strokeWidth=1.4)
            .encode(x=alt.X("fecha:T", title=None, axis=eje_x),
                    y=alt.Y("valor:Q", title=titulo_y, scale=escala, axis=alt.Axis(format=formato)),
                    color=color,
                    tooltip=[alt.Tooltip("fecha:T", title="fecha"), "serie:N",
                             alt.Tooltip("valor:Q", format=formato)])
            .properties(height=alto))


# --------------------------------------------------------------------------
st.set_page_config(page_title="Piloto de riesgo", page_icon=":material/shield:", layout="wide")

guardada, aviso_cartera = diario.leer_cartera()
fotos, avisos_fotos = diario.leer_fotos()

aviso_ajustes = None
try:
    aj0 = motor.Ajustes.desde(guardada.get("ajustes"))
except (ValueError, TypeError) as exc:
    aj0, aviso_ajustes = motor.Ajustes(), f"Ajustes guardados no válidos ({exc}); se usan los de fábrica."
reparto0 = (guardada.get("ajustes") or {}).get("reparto", "actual")
reparto0 = reparto0 if reparto0 in REPARTOS else "actual"
ten0 = guardada.get("tenencias") or {"BTC": 0.0, "USDT": 0.0}

if st.session_state.pop("guardado", False):
    st.toast("Cartera y ajustes guardados.", icon=":material/check:")

# --------------------------------------------------------------------------
# Barra lateral: la cartera y el riesgo que se acepta
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Su cartera")
    st.caption("Lo que tiene ahora. USDT, USDC, FDUSD y DAI cuentan como efectivo.")
    tabla = st.data_editor(
        pd.DataFrame({"Activo": list(ten0), "Cantidad": [float(v) for v in ten0.values()]}),
        num_rows="dynamic", hide_index=True, key="tabla",
        column_config={
            "Activo": st.column_config.TextColumn(required=True, max_chars=15,
                                                  validate=r"^[A-Za-z0-9]{2,15}$"),
            "Cantidad": st.column_config.NumberColumn(min_value=0.0, required=True),
        })

    st.header("Riesgo")
    opciones = [*NIVELES, "A medida"]
    inicial = next((i for i, v in enumerate(NIVELES.values()) if abs(v - aj0.objetivo) < 1e-9),
                   len(NIVELES))
    nivel = st.radio("¿Cuánto movimiento aguanta su cartera?", opciones, index=inicial,
                     captions=["15 % al año, el de la tesis", "25 % al año", "40 % al año",
                               "elija la cifra"],
                     help="Volatilidad anual de TODA la cartera, contando las estables. "
                          "Comprar y mantener bitcoin se ha movido un 50-60 % al año.")
    if nivel in NIVELES:
        objetivo = NIVELES[nivel]
    else:
        objetivo = st.slider("Volatilidad anual objetivo", 5, 80, int(entre(round(aj0.objetivo * 100), 5, 80)),
                             format="%d %%") / 100

    st.header("Reparto de la parte cripto")
    claves = list(REPARTOS)
    reparto = st.radio("Reparto", claves, index=claves.index(reparto0),
                       format_func=REPARTOS.get, label_visibility="collapsed",
                       help="Cómo se reparte entre monedas lo que se tenga en cripto. "
                            "El piloto decide cuánto en cripto; el reparto lo elige usted.")
    tabla_mezcla = None
    if reparto == "medida":
        base = guardada.get("mezcla") or {a: 1.0 for a in ten0 if a not in EFECTIVO} or {"BTC": 1.0}
        tot = sum(base.values()) or 1.0
        tabla_mezcla = st.data_editor(
            pd.DataFrame({"Activo": list(base), "Reparto %": [100 * v / tot for v in base.values()]}),
            num_rows="dynamic", hide_index=True, key="mezcla",
            column_config={"Activo": st.column_config.TextColumn(required=True, max_chars=15,
                                                                 validate=r"^[A-Za-z0-9]{2,15}$"),
                           "Reparto %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0,
                                                                      required=True)})
        st.caption("Si no suma 100, se reescala.")

    with st.expander("Más ajustes"):
        max_expo = st.slider("Máximo en cripto", 10, 100,
                             int(entre(round(aj0.max_exposicion * 20) * 5, 10, 100)), step=5,
                             format="%d %%", help="Con 80 %, siempre queda al menos un 20 % en estables.")
        banda = st.slider("No operar si el cambio es menor que", 0, 20,
                          int(entre(round(aj0.banda * 100), 0, 20)), format="%d %% de la cartera",
                          help="Cada orden cuesta. Con 5 %, en el pasado bastaron unas 2 operaciones "
                               "al mes en lugar de una diaria, con el mismo resultado.")
        freno = st.toggle("Freno por caídas", aj0.freno,
                          help="Si su cartera cae desde su máximo reciente, invierte menos aún.")
        inicio, tope, minimo, ventana = aj0.freno_inicio, aj0.freno_tope, aj0.freno_minimo, aj0.freno_ventana
        if freno:
            ini, top = st.slider("Empieza a frenar y frena del todo con una caída de", 1, 60,
                                 (int(entre(round(aj0.freno_inicio * 100), 1, 59)),
                                  int(entre(round(aj0.freno_tope * 100), 2, 60))),
                                 format="%d %%")
            inicio, tope = ini / 100, max(top, ini + 1) / 100
            minimo = st.slider("Frenado del todo, deja invertido", 0, 100,
                               int(entre(round(aj0.freno_minimo * 20) * 5, 0, 100)),
                               step=5, format="%d %% de lo normal",
                               help="Con 0 %, una caída grande le saca del todo y el freno no se "
                                    "suelta hasta que la caída sale de la ventana.") / 100
            ventana = st.number_input("Caída medida desde el máximo de los últimos (días)", 30, 730,
                                      int(entre(aj0.freno_ventana, 30, 730)), step=30)
        minimo_orden = st.number_input("Orden mínima (USDT)", 0.0, 10_000.0,
                                       float(entre(aj0.minimo_orden, 0, 10_000)),
                                       step=5.0, help="Binance rechaza órdenes de menos de 5-10 USDT.")

    quiere_guardar = st.button("Guardar cartera y ajustes", type="primary", width="stretch",
                               help="Guárdela cada vez que compre, venda, ingrese o retire: "
                                    "así el piloto mide su caída y el freno funciona.")
    if st.button("Actualizar precios", width="stretch"):
        _precios.clear()
    st.caption(f"Se guarda en {diario.carpeta()} (fuera de git).")

# --------------------------------------------------------------------------
st.title("Piloto de riesgo")
st.caption("Cuánto tener en cripto para que su cartera no se mueva más de lo que usted eligió. "
           "Solo recomienda: no se conecta a su cuenta ni envía órdenes.")
for aviso in [aviso_cartera, aviso_ajustes, *avisos_fotos]:
    if aviso:
        st.warning(aviso)

try:
    tenencias = leer_tabla(tabla)
    aj = motor.Ajustes(objetivo=objetivo, max_exposicion=max_expo / 100, banda=banda / 100, freno=freno,
                       freno_inicio=inicio, freno_tope=tope, freno_minimo=minimo,
                       freno_ventana=int(ventana), minimo_orden=float(minimo_orden)).validar()
    mezcla_medida = None
    if tabla_mezcla is not None:
        mezcla_medida = {a: v for a, v in leer_tabla(tabla_mezcla, "Reparto %").items() if v > 0}
        motor.normalizar_mezcla(mezcla_medida)
    cripto, efectivo = motor.separar(tenencias)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not cripto and efectivo <= 0:
    st.info("Anote en la barra lateral lo que tiene (por ejemplo BTC 0,05 y USDT 500) y pulse "
            "**Guardar cartera y ajustes**.")
    st.stop()

dia = str(mercado.ultimo_cierre().date())
minuto = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M")
try:
    with st.spinner("Descargando precios de Binance..."):
        s = sesion.calcular(tenencias, reparto,
                            mezcla_medida if reparto == "medida" else guardada.get("mezcla"),
                            fotos, aj, cierres_de=lambda a: _cierres(tuple(a), dia),
                            precios_de=lambda a: _precios(tuple(a), minuto))
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - que el usuario vea un mensaje, no una traza
    st.error("Algo ha fallado al calcular el plan. Su cartera guardada no se ha tocado. "
             "Pruebe «Actualizar precios» o vuelva a abrir la app.")
    with st.expander("Detalle técnico"):
        st.exception(exc)
    st.stop()

plan, cierres = s.plan, s.cierres
ajustes_guardables = {**asdict(aj), "reparto": reparto}
mezcla_guardable = mezcla_medida if reparto == "medida" else guardada.get("mezcla")

# Se guarda antes de pintar nada mas: la pasada que guarda termina en st.rerun().
if quiere_guardar:
    diario.guardar_cartera(tenencias, mezcla_guardable, ajustes_guardables)
    if not fotos or fotos[-1]["tenencias"] != tenencias:
        diario.anotar_foto(tenencias, {a: s.precios[a] for a in cripto})
    st.session_state["guardado"] = True
    st.rerun()

if (tenencias != guardada.get("tenencias") or mezcla_guardable != guardada.get("mezcla")
        or ajustes_guardables != guardada.get("ajustes")):
    st.sidebar.warning("Hay cambios sin guardar.")

for a in s.avisos:
    st.warning(a)
if s.sin_precio_ahora:
    st.warning(f"Sin precio de este momento para {', '.join(s.sin_precio_ahora)}: se usa el último cierre.")
if not cripto and reparto != "medida":
    total_c = sum(s.candidatas.values())
    st.info("Todo está en estables: para decidir cuánto comprar se usa el reparto "
            + ", ".join(f"{a} {pct(v / total_c)}" for a, v in s.candidatas.items())
            + ". Cámbielo en «Reparto de la parte cripto» → «A medida».")

hoy, duros, pasado, diario_tab, como = st.tabs(
    ["Hoy", "Tiempos duros", "La regla en el pasado", "Mi diario", "Cómo funciona"])

# --------------------------------------------------------------------------
with hoy:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor de la cartera", usdt(plan.total), border=True)
    c2.metric("En cripto ahora", pct(plan.exposicion_actual), usdt(plan.valor_cripto),
              delta_color="off", delta_arrow="off", border=True)
    dif = (plan.exposicion_objetivo - plan.exposicion_actual) * 100
    c3.metric("Recomendado en cripto", pct(plan.exposicion_objetivo),
              f"{dif:+.0f} puntos".replace("-", "−"), delta_color="off", border=True,
              delta_arrow="down" if dif <= -0.5 else "up" if dif >= 0.5 else "off")
    c4.metric("Movimiento previsto de su mezcla", f"{pct(plan.sigma_prevista)} al año",
              f"su objetivo: {pct(aj.objetivo)}", delta_color="off", delta_arrow="off", border=True)

    mover = abs(plan.mover)
    if plan.motivo == "en_banda":
        st.success(f"**No hace falta hacer nada.** Le separan {usdt(mover)} ({pct(mover / plan.total, 1)} "
                   f"de la cartera) de lo recomendado: menos que su banda del {pct(aj.banda)}.")
    elif plan.motivo == "freno":
        st.error(f"**Freno activo: pase {usdt(mover)} de cripto a estables.** Su cartera ha caído un "
                 f"{pct(-plan.caida, 1)} desde su máximo de los últimos {int(aj.freno_ventana)} días.")
    elif plan.motivo == "reducir":
        st.warning(f"**Pase {usdt(mover)} de cripto a estables.** Su mezcla se mueve más de lo que "
                   "su objetivo admite.")
    elif plan.motivo == "aumentar":
        st.info(f"**Puede pasar {usdt(mover)} de estables a cripto.** Su mezcla está más tranquila: "
                "su objetivo admite más.")
    elif plan.motivo == "reequilibrar":
        st.info("**Reequilibre el reparto entre monedas.** El total en cripto ya está bien.")
    else:
        st.info("Con su objetivo y el mercado de hoy, no toca tener cripto.")

    todas = plan.ordenes
    o = todas[todas["ejecutar"]]
    if len(o):
        st.dataframe(pd.DataFrame({
            "Operación": np.where(o["importe_orden"] > 0, "Comprar", "Vender"),
            "Moneda": o.index,
            "Cantidad": o["cantidad_orden"].abs().map(cantidad),
            "Importe": o["importe_orden"].abs().map(usdt),
            "Precio ahora": o["precio"].map(precio)}), hide_index=True, width="stretch")
    elif plan.actuar:
        st.caption("Hoy no hay ninguna orden que se pueda ejecutar.")
    if plan.actuar:
        pequenas = todas[~todas["ejecutar"] & ~todas["recortada"] & (todas["importe"].abs() >= 0.01)]
        if len(pequenas):
            st.caption(f"Se omiten órdenes de menos de {usdt(aj.minimo_orden)}: "
                       + ", ".join(pequenas.index) + ".")
        if todas["recortada"].any():
            st.caption("Compras recortadas a lo que pagan su efectivo y las ventas de arriba: "
                       + ", ".join(todas.index[todas["recortada"]]) + ".")
    if len(o):
        despues = plan.valor_cripto + float(o["importe_orden"].sum())
        st.caption(f"Después: {pct(despues / plan.total)} en cripto y "
                   f"{usdt(plan.total - despues)} en estables. Guarde la cartera al terminar.")

    st.subheader("Por qué")
    escala_libre = aj.objetivo / plan.sigma_prevista if plan.sigma_prevista > 0 else float("inf")
    tope_txt = (f", limitado a su máximo del {pct(aj.max_exposicion)}"
                if escala_libre > aj.max_exposicion else "")
    if not aj.freno:
        freno_txt = "desactivado."
    elif not fotos:
        freno_txt = "aún sin historial: empieza a medir la caída de su cartera cuando la guarde."
    elif plan.factor_freno < 1:
        freno_txt = (f"**activo**. Su cartera ha caído un {pct(-plan.caida, 1)} desde su máximo de "
                     f"{int(aj.freno_ventana)} días: se invierte el {pct(plan.factor_freno)} de lo "
                     "que permitiría el objetivo.")
    else:
        freno_txt = (f"inactivo. Su cartera está un {pct(-plan.caida, 1)} por debajo de su máximo de "
                     f"{int(aj.freno_ventana)} días; empieza a frenar con un {pct(aj.freno_inicio)}.")
    st.markdown(
        f"- Su mezcla de cripto se está moviendo a un ritmo de **{pct(plan.sigma_prevista)} al año** "
        f"(previsión EWMA con los cierres hasta el {fecha(plan.fecha_datos)}).\n"
        f"- Para que la cartera entera se mueva un **{pct(aj.objetivo)}**, la parte en cripto debe ser "
        f"{pct(aj.objetivo)} ÷ {pct(plan.sigma_prevista)} = **{pct(min(escala_libre, 9.99))}**{tope_txt}.\n"
        f"- Freno: {freno_txt}")

    if plan.valor_cripto > 0:
        st.subheader("Dónde está el riesgo")
        rg = plan.riesgo[plan.riesgo["peso_dinero"] > 0]
        d = pd.DataFrame({"Moneda": np.repeat(rg.index, 2),
                          "Medida": ["del dinero", "del riesgo"] * len(rg),
                          "Parte": np.ravel(np.column_stack([rg["peso_dinero"], rg["peso_riesgo"]]))})
        st.altair_chart(
            alt.Chart(d).mark_bar().encode(
                y=alt.Y("Moneda:N", title=None), x=alt.X("Parte:Q", title=None, axis=alt.Axis(format="%")),
                yOffset="Medida:N",
                color=alt.Color("Medida:N", title=None, legend=alt.Legend(orient="bottom"),
                                scale=alt.Scale(range=["#9ecae1", "#e6550d"])),
                tooltip=["Moneda", "Medida", alt.Tooltip("Parte:Q", format=".1%")])
            .properties(height=60 + 45 * len(rg)), width="stretch")
        st.caption("Una moneda puede ser el 30 % del dinero y el 50 % del riesgo: pesa cuánto se "
                   "mueve y cuánto lo hace a la vez que las demás. Las estables no aportan riesgo.")

# --------------------------------------------------------------------------
with duros:
    valores_hoy = (plan.ordenes["unidades"] * plan.ordenes["precio"]).to_dict()
    valores_plan = (plan.ordenes["unidades_objetivo"] * plan.ordenes["precio"]).to_dict()
    efectivo_plan = plan.total - plan.cripto_objetivo
    activos_riesgo = [a for a, v in {**valores_hoy, **valores_plan}.items() if v > 0]
    desde = cierres[activos_riesgo].dropna().index[0] if activos_riesgo else None

    st.subheader("Si se repitiera lo peor" + (f" desde el {fecha(desde)}" if desde is not None else ""))
    st.caption("Qué perdería sin hacer nada durante ese tramo, con las mismas monedas y cantidades.")
    e_hoy = motor.escenarios(cierres, valores_hoy, plan.efectivo)
    e_plan = motor.escenarios(cierres, valores_plan, efectivo_plan)

    def _perd(fila) -> str:
        return "—" if pd.isna(fila["perdida"]) else f"{usdt(fila['perdida'])} ({pct(fila['perdida_pct'])})"

    st.dataframe(pd.DataFrame({
        "Escenario": e_hoy.index,
        "Cuándo pasó": [f"{fecha(a)} a {fecha(b)}" if pd.notna(b) else "—"
                        for a, b in zip(e_hoy["desde"], e_hoy["hasta"])],
        "Con su cartera de hoy": [_perd(f) for _, f in e_hoy.iterrows()],
        "Siguiendo la recomendación": [_perd(f) for _, f in e_plan.iterrows()]}),
        hide_index=True, width="stretch")

    st.subheader("Un mal día y el próximo mes")
    m_hoy, m_plan = (motor.mal_dia(cierres, valores_hoy, plan.efectivo),
                     motor.mal_dia(cierres, valores_plan, efectivo_plan))
    mes_hoy, mes_plan = (motor.proximo_mes(cierres, valores_hoy, plan.efectivo),
                         motor.proximo_mes(cierres, valores_plan, efectivo_plan))
    st.dataframe(pd.DataFrame({
        "Medida": ["Un mal día (1 de cada 20)", "Si se pasa, pierde de media",
                   "Próximo mes: probabilidad de caer más del 10 %",
                   "Próximo mes: probabilidad de caer más del 20 %",
                   "Próximo mes: el peor 5 % de los casos"],
        "Con su cartera de hoy": [usdt(-m_hoy["var"]), usdt(-m_hoy["cvar"]),
                                  pct(mes_hoy["prob_caida_10"]), pct(mes_hoy["prob_caida_20"]),
                                  pct(mes_hoy["retorno_p5"])],
        "Siguiendo la recomendación": [usdt(-m_plan["var"]), usdt(-m_plan["cvar"]),
                                       pct(mes_plan["prob_caida_10"]), pct(mes_plan["prob_caida_20"]),
                                       pct(mes_plan["retorno_p5"])]}),
        hide_index=True, width="stretch")
    st.caption("Mal día: método histórico con los dos últimos años. Próximo mes: 10 000 escenarios "
               "remuestreando bloques de días reales de todas sus monedas a la vez.")

    if st.button("Comprobar el mal día con los 4 métodos validados (unos 10 s)",
                 disabled=plan.valor_cripto <= 0):
        st.session_state["validar"] = True
    if st.session_state.get("validar") and plan.valor_cripto > 0:
        try:
            with st.spinner("Contrastando cada método con el pasado de su cartera..."):
                r = _analizar(tuple(sorted(tenencias.items())), cierres)
            v = r.validacion
            st.dataframe(pd.DataFrame({
                "Método": [METODOS.get(m, m) for m in r.var_manana.index],
                "Mal día (1 de cada 20)": [usdt(-x) for x in r.var_manana["var"]],
                "Si se pasa, de media": [usdt(-x) for x in r.var_manana["cvar"]],
                "¿Habría acertado con su cartera?": [
                    ("sin datos suficientes" if not v.loc[m, "suficiente"] else
                     "sí" if v.loc[m, "aceptado"] else "no") + f" (falló el {pct(v.loc[m, 'tasa'], 1)})"
                    for m in r.var_manana.index]}), hide_index=True, width="stretch")
            st.caption(f"Valorado al cierre del {fecha(r.fecha)}. Se esperaba fallar el 5 % de los días; "
                       "«sí» = pruebas de Kupiec y Christoffersen superadas (p > 0,05).")
        except ValueError as exc:
            st.warning(f"No se pudo validar: {exc}")

# --------------------------------------------------------------------------
with pasado:
    try:
        t = _simular(cierres[list(s.mezcla_efectiva)], tuple(sorted(s.mezcla_efectiva.items())),
                     tuple(sorted(asdict(aj).items())))
    except ValueError as exc:
        st.warning(f"No se puede simular: {exc}")
        t = None
    if t is not None:
        res = control.resumen(t)
        anos = len(t) / 365
        reparto_txt = ", ".join(f"{a} {pct(w)}"
                                for a, w in motor.normalizar_mezcla(s.mezcla_efectiva).items())
        st.markdown(f"Si desde el **{fecha(t.index[0])}** hubiera seguido esta regla con su reparto "
                    f"({reparto_txt}) y sus ajustes, con 10 000 USDT:")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Caída máxima", pct(res.loc["estrategia", "caida_maxima"]),
                  f"comprar y mantener: {pct(res.loc['pasiva', 'caida_maxima'])}",
                  delta_color="off", delta_arrow="off", border=True)
        c2.metric("Rentabilidad anual", pct(res.loc["estrategia", "rentabilidad_anual"], 1),
                  f"comprar y mantener: {pct(res.loc['pasiva', 'rentabilidad_anual'], 1)}",
                  delta_color="off", delta_arrow="off", border=True)
        c3.metric("Volatilidad", pct(res.loc["estrategia", "volatilidad"]),
                  f"objetivo: {pct(aj.objetivo)}", delta_color="off", delta_arrow="off", border=True)
        c4.metric("Operaciones al año", num(t.attrs["operaciones"] / anos),
                  f"en cripto de media: {pct(t['exposicion'].mean())}",
                  delta_color="off", delta_arrow="off", border=True)
        if anos < 1:
            st.caption(f"Historia corta ({len(t)} días): tómelo como una ilustración, no como evidencia.")

        capital = 10_000 * (1 + t[["estrategia", "pasiva", "pasiva_igual_vol"]]).cumprod()
        capital.columns = SERIES
        st.altair_chart(lineas(capital, "USDT (log)", "~s", log=True, alto=320), width="stretch")
        caidas = (capital / capital.cummax() - 1)[["Con la regla", "Comprar y mantener"]]
        st.altair_chart(lineas(caidas, "caída", ".0%", alto=200), width="stretch")
        st.altair_chart(lineas(pd.DataFrame({"Con la regla": t["exposicion"]}), "en cripto",
                               ".0%", alto=190), width="stretch")

        st.markdown(
            "**Cómo leerlo.** La regla no gana más que comprar y mantener: arriesga mucho menos. "
            "La comparación justa es con *mantener a igual volatilidad* (una parte fija en cripto "
            f"que da el mismo riesgo: {pct(t.attrs['fraccion_igual_vol'])}), que solo se puede "
            "elegir conociendo el futuro. La regla llegó a ese riesgo sin conocerlo.\n\n"
            "La tesis (2021-2026, 8 criptoactivos) confirma que el control de volatilidad reduce "
            "el riesgo de forma clara (HE3); que además gane más que la pasiva a igual riesgo no "
            "está demostrado (HE4). El pasado no garantiza el futuro.")

# --------------------------------------------------------------------------
with diario_tab:
    if not fotos:
        st.info("Aún no hay fotos. Pulse **Guardar cartera y ajustes**: desde entonces el piloto mide "
                "la caída de su cartera día a día, aunque no vuelva a abrirlo.")
    else:
        if len(s.valor) > 1:
            st.altair_chart(lineas(pd.DataFrame({"Su cartera": 100 * s.valor / s.valor.iloc[0]}),
                                   "valor (base 100)", ".0f", alto=240), width="stretch")
            st.caption("Sin contar ingresos ni retiros: cada día se valoran las unidades de su última "
                       f"foto. Caída vigente: {pct(s.caida, 1)} desde el máximo de "
                       f"{int(aj.freno_ventana)} días.")
        filas = []
        for f in reversed(fotos):
            ten, px = f["tenencias"], f.get("precios", {})
            valor_f = sum(u if a in EFECTIVO else u * px.get(a, np.nan) for a, u in ten.items())
            en_cripto = sum(u * px.get(a, np.nan) for a, u in ten.items() if a not in EFECTIVO)
            filas.append({"Fecha (UTC)": pd.Timestamp(f["fecha_utc"]).strftime("%d-%m-%Y %H:%M"),
                          "Valor": usdt(valor_f),
                          "En cripto": pct(en_cripto / valor_f) if valor_f else "—",
                          "Tenencias": ", ".join(f"{a} {cantidad(u)}" for a, u in ten.items() if u > 0)})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
with como:
    st.markdown(f"""
**La regla**

    parte en cripto = mínimo(máximo, objetivo ÷ volatilidad prevista) × freno

1. **Volatilidad prevista** de su mezcla, con una media exponencial (EWMA, λ = 0,94) de los
   cierres diarios de Binance. No predice el precio: predice cuánto se va a mover, y eso sí se
   puede, porque los días agitados vienen en rachas.
2. **Objetivo**: cuánto acepta que se mueva la cartera entera. Si el mercado se agita, se invierte
   menos; si se calma, más. El resto queda en estables.
3. **Freno**: si su cartera cae más de un {pct(aj.freno_inicio)} desde su máximo de los últimos
   {int(aj.freno_ventana)} días, invierte menos aún; con una caída del {pct(aj.freno_tope)} o más,
   solo un {pct(aj.freno_minimo)} de lo normal. Se suelta solo: la caída se mide contra el máximo
   de una ventana móvil.
4. **Banda**: si el cambio es menor que un {pct(aj.banda)} de la cartera, no se opera.

**Por qué funciona y hasta dónde.** Es la conclusión de la tesis y del laboratorio (paso 9): la
dirección del precio no se puede prever, su volatilidad sí. Controlarla recorta las caídas grandes.
No promete ganar más.

**Límites que conviene tener presentes**

- Un desplome de un día no avisa. La previsión reacciona al día siguiente, no antes.
- Las estables no son riesgo cero: USDT o USDC pueden perder la paridad, y un exchange puede fallar.
- Las cifras del pasado usan comisiones del 0,15 %; si las suyas son mayores, opere menos.
- El piloto solo mide y recomienda. Las decisiones y las órdenes son suyas.
""")

st.caption(f"Velas cerradas hasta el {fecha(cierres.index.max())} (UTC) y precios de este momento, "
           "de data-api.binance.vision.")
