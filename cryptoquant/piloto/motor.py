"""Cuanto tener en cripto para no pasar de un riesgo elegido.

Es el control de volatilidad de la tesis (paso 9 del laboratorio) aplicado a
la cartera real del usuario, no a la del sistema:

    exposicion = min(maxima, objetivo / volatilidad_prevista) x freno(caida)

- `volatilidad_prevista`: EWMA (lambda 0.94) de la parte cripto con su mezcla,
  con datos hasta el ultimo cierre diario. No predice el precio: la direccion
  no se puede prever (tesis, HE4 no concluyente; filtro de evidencia); la
  volatilidad si, porque los dias agitados vienen en rachas.
- `freno`: la regla lineal del sistema (`portfolio.sizing.drawdown_scalar`),
  con dos cambios para que no se quede atascado. La caida se mide contra el
  maximo de una ventana movil, no contra el maximo historico, y el freno tiene
  un minimo. Con la regla del sistema, una caida del 25 % dejaria la cuenta en
  efectivo; en efectivo la cuenta ya no se mueve, la caida no se recupera y el
  freno no se soltaria nunca.
- `banda`: si el cambio es pequeno no se opera. Cada orden cuesta, y quien
  ejecuta a mano no debe tocar la cartera todos los dias.

Todo es causal: la simulacion decide con el cierre de t y aplica en t+1.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..laboratorio.control import sigma_ewma
from ..laboratorio.datos import validar_ticker
from ..portfolio.sizing import drawdown_scalar
from ..riesgo import futuro, var
from ..riesgo.cartera import EFECTIVO

ANUAL = 365.0
MIN_DIAS = 60      # EWMA 0.94 pesa sobre todo el ultimo mes; con menos no hay prevision
ARRANQUE = 30


@dataclass
class Ajustes:
    objetivo: float = 0.15          # volatilidad anual de TODA la cartera (cripto + estables)
    max_exposicion: float = 1.0     # parte maxima en cripto; 0.8 deja siempre un 20 % en estables
    banda: float = 0.05             # no operar si el cambio es menor que esta fraccion del total
    freno: bool = True
    freno_inicio: float = 0.10      # caida desde el maximo a partir de la que se recorta
    freno_tope: float = 0.25        # caida con la que el freno llega a su minimo
    freno_minimo: float = 0.25      # el freno nunca deja menos de esta fraccion de la escala
    freno_ventana: int = 180        # dias del maximo contra el que se mide la caida
    minimo_orden: float = 10.0      # USDT; Binance rechaza ordenes mas pequenas
    coste: float = 0.0015           # comision + deslizamiento por unidad negociada
    lam: float = 0.94

    def validar(self) -> "Ajustes":
        if not isinstance(self.freno, bool):
            raise ValueError("freno debe ser verdadero o falso")
        for k, v in asdict(self).items():
            if k != "freno" and (isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v)):
                raise ValueError(f"{k}: debe ser un numero finito")
        comprobar = [
            (0 < self.objetivo < 5, "el objetivo de volatilidad debe estar entre 0 y 5 (0.15 = 15 %)"),
            (0 < self.max_exposicion <= 1, "la exposicion maxima debe estar entre 0 y 1 (sin apalancamiento)"),
            (0 <= self.banda < 1, "la banda debe estar entre 0 y 1"),
            (0 <= self.freno_inicio < self.freno_tope < 1,
             "el freno debe empezar antes de su tope, y ambos entre 0 y 1"),
            (0 <= self.freno_minimo <= 1, "el minimo del freno debe estar entre 0 y 1"),
            (int(self.freno_ventana) == self.freno_ventana and 2 <= self.freno_ventana <= 3650,
             "la ventana del freno debe ser un numero entero de dias entre 2 y 3650"),
            (0 <= self.minimo_orden < 1e9, "el minimo por orden no puede ser negativo"),
            (0 <= self.coste < 0.1, "el coste debe estar entre 0 y 0.1"),
            (0 < self.lam < 1, "lambda debe estar entre 0 y 1"),
        ]
        for ok, msg in comprobar:
            if not ok:
                raise ValueError(msg)
        return self

    @classmethod
    def desde(cls, d: dict | None) -> "Ajustes":
        """Desde un dict guardado; ignora claves desconocidas (versiones futuras)."""
        if d is not None and not isinstance(d, dict):
            raise ValueError("los ajustes guardados no tienen el formato esperado")
        campos = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in campos}).validar()


def factor_freno(caida: float, aj: Ajustes) -> float:
    if not aj.freno:
        return 1.0
    return max(aj.freno_minimo, drawdown_scalar(caida, aj.freno_inicio, aj.freno_tope))


def es_cantidad(v) -> bool:
    return (isinstance(v, numbers.Real) and not isinstance(v, bool)
            and math.isfinite(v) and v >= 0)


def separar(tenencias: dict[str, float]) -> tuple[dict[str, float], float]:
    """-> (unidades de cada cripto, efectivo en estables)."""
    for a, v in tenencias.items():
        if not es_cantidad(v):
            raise ValueError(f"{a}: la cantidad debe ser un numero finito y no negativo")
    cripto = {a: float(v) for a, v in tenencias.items() if a not in EFECTIVO and v > 0}
    return cripto, float(sum(v for a, v in tenencias.items() if a in EFECTIVO))


def normalizar_mezcla(mezcla: dict[str, float]) -> pd.Series:
    if not isinstance(mezcla, dict) or not all(es_cantidad(v) for v in mezcla.values()):
        raise ValueError("la mezcla solo admite pesos finitos y no negativos")
    w = pd.Series([float(v) for v in mezcla.values()],
                  index=[validar_ticker(str(a)) for a in mezcla], dtype=float)
    w = w.groupby(level=0).sum()          # 'btc' y 'BTC' son la misma moneda
    w = w[w > 0]
    if w.empty:
        raise ValueError("la mezcla esta vacia: indique en que repartir la parte cripto")
    if any(a in EFECTIVO for a in w.index):
        raise ValueError("la mezcla es el reparto de la parte cripto: sin estables")
    with np.errstate(over="ignore"):
        suma = float(w.sum())
    if not math.isfinite(suma):
        raise ValueError("los pesos de la mezcla son demasiado grandes")
    return w / w.sum()


def retornos_log(cierres: pd.DataFrame, activos) -> pd.DataFrame:
    """Retornos logaritmicos diarios en las fechas comunes a todos los activos."""
    faltan = [a for a in activos if a not in cierres.columns]
    if faltan:
        raise ValueError(f"sin precios para: {', '.join(faltan)}")
    px = cierres[list(activos)].dropna()
    if (px <= 0).any().any():
        raise ValueError("hay precios nulos o negativos en el historico")
    lr = np.log(px).diff().dropna()
    if len(lr) < MIN_DIAS:
        cortos = [f"{a} ({cierres[a].notna().sum()} dias)" for a in activos
                  if cierres[a].notna().sum() < MIN_DIAS + 1]
        detalle = ", ".join(cortos) if cortos else f"solo {len(lr)} dias en comun"
        raise ValueError(f"hacen falta {MIN_DIAS} dias de historia para prever el riesgo: {detalle}")
    return lr


def sigma_mezcla(lr: pd.DataFrame, w: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidad ANUAL prevista para el dia siguiente, de la mezcla `w`.

    La varianza EWMA de una mezcla fija es la EWMA de su retorno: w'V_t w
    sigue la misma recursion que cada V_t. Con un solo activo coincide con
    `laboratorio.control.sigma_ewma`.
    """
    return sigma_ewma(lr[w.index] @ w, lam, ARRANQUE) * math.sqrt(ANUAL)


def covarianza_ewma(lr: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    x = lr.to_numpy(dtype=float)
    v = np.cov(x[:ARRANQUE].T, bias=True).reshape(x.shape[1], x.shape[1])
    for r in x:
        v = lam * v + (1 - lam) * np.outer(r, r)
    return pd.DataFrame(v, index=lr.columns, columns=lr.columns)


def pesos_igual_riesgo(cierres: pd.DataFrame, activos, lam: float = 0.94) -> dict[str, float]:
    """Reparto en que cada moneda aporta lo mismo si se movieran por separado (1 / vol)."""
    lr = retornos_log(cierres, list(activos))
    vol = np.sqrt(np.diag(covarianza_ewma(lr, lam).to_numpy()))
    if not (vol > 0).all():
        raise ValueError("alguna moneda no se ha movido nada: no se puede repartir por riesgo")
    w = (1 / vol) / (1 / vol).sum()
    return dict(zip(lr.columns, map(float, w)))


REPARTOS = ("actual", "igual_riesgo", "medida")


def candidatas_mezcla(reparto: str, cripto: dict[str, float],
                      medida: dict[str, float] | None) -> dict[str, float]:
    """Monedas entre las que se reparte la parte cripto (con sus pesos, si los hay).

    Con todo en estables no hay reparto actual que mantener: se usa el ultimo
    reparto a medida guardado o, si no hay, todo en BTC.
    """
    if reparto not in REPARTOS:
        raise ValueError(f"reparto desconocido: {reparto!r} (use {', '.join(REPARTOS)})")
    if reparto == "medida":
        if not medida:
            raise ValueError("el reparto a medida esta vacio")
        return normalizar_mezcla(medida).to_dict()
    if cripto:
        return dict(cripto)
    return normalizar_mezcla(medida).to_dict() if medida else {"BTC": 1.0}


def resolver_mezcla(reparto: str, cripto: dict[str, float], candidatas: dict[str, float],
                    cierres: pd.DataFrame, lam: float = 0.94) -> dict[str, float] | None:
    """La mezcla para `decidir`; None = la que se tiene ahora."""
    if reparto == "igual_riesgo":
        return pesos_igual_riesgo(cierres, sorted(candidatas), lam)
    if reparto == "medida" or not cripto:
        return dict(candidatas)
    return None


# --------------------------------------------------------------------------
# La decision de hoy
# --------------------------------------------------------------------------
@dataclass
class Plan:
    total: float
    efectivo: float
    valor_cripto: float
    exposicion_actual: float
    sigma_prevista: float        # anual, de la parte cripto con la mezcla objetivo
    escala_vol: float            # min(max_exposicion, objetivo / sigma)
    caida: float                 # caida vigente de la cuenta (<= 0)
    factor_freno: float
    exposicion_objetivo: float
    cripto_objetivo: float
    mover: float                 # > 0 comprar cripto, < 0 pasar a estables (USDT)
    actuar: bool
    motivo: str                  # categoria estable; el texto lo pone quien lo muestra
    ordenes: pd.DataFrame        # por activo: precio, unidades, objetivo, diferencia, importe (el plan);
                                 # importe_orden, cantidad_orden, ejecutar, recortada (lo que se ensena)
    riesgo: pd.DataFrame         # por activo: peso en dinero, peso en riesgo, vol anual
    fecha_datos: pd.Timestamp


def decidir(tenencias: dict[str, float], precios: dict[str, float], cierres: pd.DataFrame,
            mezcla: dict[str, float] | None = None, caida: float = 0.0,
            ajustes: Ajustes | None = None) -> Plan:
    """Cuanto tener hoy en cripto y que ordenes lo consiguen.

    Args:
        tenencias: unidades por activo; las estables cuentan como efectivo.
        precios: precio actual de cada cripto (valora la cartera y las ordenes).
        cierres: cierres diarios; solo los cerrados deciden (causal).
        mezcla: reparto deseado de la parte cripto. Por defecto, el actual.
        caida: caida vigente de la cuenta (<= 0), de `caida_vigente`.
    """
    aj = (ajustes or Ajustes()).validar()
    if not (math.isfinite(caida) and -1 <= caida <= 0):
        raise ValueError("la caida debe estar entre -1 y 0")
    cripto, efectivo = separar(tenencias)
    w = normalizar_mezcla(mezcla) if mezcla else None
    activos = sorted(set(cripto) | set(w.index if w is not None else ()))
    for a in activos:
        p = precios.get(a)
        if p is None or not (isinstance(p, (int, float)) and math.isfinite(p) and p > 0):
            raise ValueError(f"{a}: sin precio actual valido")
    unidades = pd.Series({a: cripto.get(a, 0.0) for a in activos}, dtype=float)
    precio = pd.Series({a: float(precios[a]) for a in activos}, dtype=float)
    valor = unidades * precio
    total = float(valor.sum() + efectivo)
    if not math.isfinite(total):
        raise ValueError("las cantidades son demasiado grandes para valorarlas")
    if total <= 0:
        raise ValueError("la cartera no tiene valor: anote lo que tiene")

    if w is not None:
        pass
    elif valor.sum() > 0:
        w = normalizar_mezcla(valor.to_dict())
    else:
        raise ValueError("todo esta en estables: indique la mezcla en la que invertiria")

    lr = retornos_log(cierres, activos)
    sigma = float(sigma_mezcla(lr, w, aj.lam).iloc[-1])
    # Sin movimiento no hay riesgo que limitar: se admite el maximo.
    escala = min(aj.max_exposicion, aj.objetivo / sigma) if sigma > 0 else aj.max_exposicion
    freno = factor_freno(caida, aj)
    expo = escala * freno
    cripto_obj = expo * total

    peso_obj = w.reindex(activos, fill_value=0.0)
    objetivo_u = cripto_obj * peso_obj / precio
    importe = (objetivo_u - unidades) * precio
    mover = float(cripto_obj - valor.sum())
    umbral = max(aj.banda * total, aj.minimo_orden)
    actuar = bool(abs(mover) >= umbral or (importe.abs() >= umbral).any())
    ordenes = pd.DataFrame({"precio": precio, "unidades": unidades, "unidades_objetivo": objetivo_u,
                            "diferencia": objetivo_u - unidades, "importe": importe})
    posible = actuar & (ordenes["importe"].abs() >= aj.minimo_orden)
    ordenes["importe_orden"] = _al_efectivo(importe, posible, efectivo, aj.minimo_orden)
    ordenes["cantidad_orden"] = ordenes["importe_orden"] / precio
    ordenes["ejecutar"] = ordenes["importe_orden"] != 0
    ordenes["recortada"] = posible & (importe > 0) & (ordenes["importe_orden"] < importe * (1 - 1e-9))

    if valor.sum() == 0 and cripto_obj < aj.minimo_orden:
        motivo = "sin_cripto"
    elif not actuar:
        motivo = "en_banda"
    elif freno < 1 and mover < 0:
        motivo = "freno"
    elif mover < 0:
        motivo = "reducir"
    elif mover > 0:
        motivo = "aumentar"
    else:
        motivo = "reequilibrar"

    cov = covarianza_ewma(lr, aj.lam)
    peso = valor / total
    marginal = cov.to_numpy() @ peso.to_numpy()
    var_total = float(peso.to_numpy() @ marginal)
    contrib = peso.to_numpy() * marginal / var_total if var_total > 0 else np.zeros(len(peso))
    riesgo = pd.DataFrame({"peso_dinero": peso, "peso_riesgo": contrib,
                           "vol_anual": np.sqrt(np.diag(cov.to_numpy()) * ANUAL)}, index=activos)

    return Plan(total, efectivo, float(valor.sum()), float(valor.sum() / total), sigma, escala,
                float(caida), freno, expo, cripto_obj, mover, actuar, motivo, ordenes, riesgo,
                lr.index[-1])


def _al_efectivo(importe: pd.Series, posible: pd.Series, efectivo: float, minimo: float) -> pd.Series:
    """Importe de cada orden que se puede ejecutar de verdad (0 = no se ensena).

    Las compras se pagan con el efectivo y con las ventas que SI se hacen. Si
    una venta queda por debajo del minimo de Binance no se ejecuta, y la
    compra que pagaba se recorta a lo que hay; si al recortarla queda por
    debajo del minimo, tampoco se ensena.
    """
    orden = importe.where(posible, 0.0)
    disponible = efectivo - float(orden[orden < 0].sum())
    compras = posible & (importe > 0)
    while compras.any():
        pedido = float(importe[compras].sum())
        if pedido <= disponible * (1 + 1e-9):
            break
        orden[compras] = importe[compras] * (disponible / pedido)
        pocas = compras & (orden < minimo)
        if not pocas.any():
            break
        orden[pocas] = 0.0
        compras &= ~pocas
    return orden


# --------------------------------------------------------------------------
# La caida de la cuenta, a partir de las fotos del diario
# --------------------------------------------------------------------------
def valor_cuenta(fotos: list[dict], cierres: pd.DataFrame,
                 ahora: dict[str, float] | None = None) -> pd.Series:
    """Valor de la cuenta por dia, SIN contar ingresos ni retiros.

    Cada dia se valoran las unidades de la ultima foto anterior, y el valor
    se encadena como una rentabilidad (base 1 el dia de la primera foto). Asi
    un ingreso no parece una ganancia ni un retiro una perdida. `ahora`
    (precios actuales) anade el valor de este momento como ultimo punto.
    """
    if not fotos:
        return pd.Series(dtype=float)
    f = sorted(fotos, key=lambda x: _utc(x["fecha_utc"]))
    dias = pd.DatetimeIndex([_utc(x["fecha_utc"]).normalize() for x in f])
    fin = cierres.index.max() if len(cierres) else dias[-1]
    idx = pd.date_range(dias[0], max(fin, dias[0]), freq="D", tz="UTC")

    def valorar(ten: dict, px_de) -> float | None:
        v = 0.0
        for a, u in ten.items():
            if a in EFECTIVO:
                v += u
            elif u > 0:
                p = px_de(a)
                if p is None or not math.isfinite(p):
                    return None
                v += u * p
        return v

    def cierre(dia):
        return lambda a: (float(cierres.at[dia, a]) if a in cierres.columns and dia in cierres.index
                          and pd.notna(cierres.at[dia, a]) else None)

    nav, puntos = 1.0, []
    for i, d in enumerate(idx):
        # tenencias con las que se paso la noche anterior: ultima foto de un dia previo
        previas = [x for x, dx in zip(f, dias) if dx < d]
        if i > 0 and previas:
            ten = previas[-1]["tenencias"]
            v0, v1 = valorar(ten, cierre(idx[i - 1])), valorar(ten, cierre(d))
            if v0 and v1 is not None:
                nav *= v1 / v0
        puntos.append((d, nav))
    s = pd.Series(dict(puntos), dtype=float)
    if ahora is not None:
        ten = f[-1]["tenencias"]
        v0, v1 = valorar(ten, cierre(s.index[-1])), valorar(ten, ahora.get)
        if v0 and v1 is not None:
            s[pd.Timestamp.now(tz="UTC").floor("min")] = s.iloc[-1] * v1 / v0
    return s


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def caida_vigente(valor: pd.Series, ventana: int) -> float:
    """Caida desde el maximo de los ultimos `ventana` dias (<= 0)."""
    if valor.empty:
        return 0.0
    desde = valor.index[-1] - pd.Timedelta(days=ventana)
    tramo = valor[valor.index >= desde]
    return float(min(0.0, tramo.iloc[-1] / tramo.max() - 1))


# --------------------------------------------------------------------------
# La regla aplicada al pasado, con esta mezcla
# --------------------------------------------------------------------------
def simular(cierres: pd.DataFrame, mezcla: dict[str, float],
            ajustes: Ajustes | None = None) -> pd.DataFrame:
    """Rentabilidad diaria de la regla y de sus referencias.

    Columnas (como `laboratorio.control`, para usar `control.resumen`):
    exposicion, pasiva, estrategia, pasiva_igual_vol, y ademas freno y caida.
    La mezcla se rebalancea a diario dentro de la parte cripto; entre cambios
    de exposicion la parte cripto deriva con el precio, y la banda compara el
    objetivo con esa exposicion derivada, como le pasaria a quien opera a mano.
    """
    aj = (ajustes or Ajustes()).validar()
    w = normalizar_mezcla(mezcla)
    lr = retornos_log(cierres, list(w.index))
    escala = (aj.objetivo / sigma_mezcla(lr, w, aj.lam)).clip(upper=aj.max_exposicion).dropna()
    r = (np.expm1(lr[w.index]) @ w).reindex(escala.index).to_numpy()
    e_obj = escala.to_numpy()
    n = len(e_obj)
    if n < 2:
        raise ValueError("historia insuficiente para simular la regla")

    ventana = int(aj.freno_ventana)
    eq = np.ones(n)
    expo = 0.0                                    # se empieza en estables
    filas = np.zeros((n - 1, 6))
    for k in range(n - 1):
        pico = eq[max(0, k - ventana + 1):k + 1].max()
        caida = eq[k] / pico - 1
        freno = factor_freno(caida, aj)
        objetivo = e_obj[k] * freno
        giro = 0.0
        if abs(objetivo - expo) >= aj.banda or (aj.banda > 0 and objetivo == 0 < expo):
            giro, expo = abs(objetivo - expo), objetivo
        ret = expo * r[k + 1] - aj.coste * giro
        eq[k + 1] = eq[k] * (1 + ret)
        filas[k] = (expo, r[k + 1], ret, freno, caida, giro)
        crece = 1 + expo * r[k + 1]
        expo = expo * (1 + r[k + 1]) / crece if crece > 0 else 0.0

    t = pd.DataFrame(filas, index=escala.index[1:],
                     columns=["exposicion", "pasiva", "estrategia", "freno", "caida", "giro"])
    sd = t["pasiva"].std()
    frac = float(t["estrategia"].std() / sd) if sd > 0 else 0.0
    t["pasiva_igual_vol"] = frac * t["pasiva"]
    t.attrs["fraccion_igual_vol"] = frac
    t.attrs["operaciones"] = int((t["giro"] > 0).sum())
    return t


# --------------------------------------------------------------------------
# Tiempos duros: lo peor del historico, con lo que tiene hoy
# --------------------------------------------------------------------------
ESCENARIOS = {"Peor día": 1, "Peor semana": 7, "Peor mes": 30, "Peor trimestre": 90,
              "Peor año": 365}


def escenarios(cierres: pd.DataFrame, valores: dict[str, float], efectivo: float,
               ventanas: dict[str, int] | None = None) -> pd.DataFrame:
    """Si se repitiera el peor tramo del historico, cuanto perderia esta cartera.

    `valores`: dinero en cada cripto HOY. Las unidades quedan fijas durante el
    tramo (quien no hace nada). Solo fechas en que todos los activos cotizaban.
    """
    ventanas = ventanas or ESCENARIOS
    v = pd.Series({a: x for a, x in valores.items() if x > 0}, dtype=float)
    total = float(v.sum() + efectivo)
    filas = {}
    if v.empty:
        return pd.DataFrame({"perdida": {k: 0.0 for k in ventanas}, "perdida_pct": 0.0,
                             "desde": pd.NaT, "hasta": pd.NaT})
    px = cierres[list(v.index)].dropna()
    if len(px):
        px = px.asfreq("D").ffill()   # h filas = h dias aunque falte alguna vela
    for nombre, h in ventanas.items():
        if len(px) <= h:
            filas[nombre] = {"perdida": np.nan, "perdida_pct": np.nan, "desde": pd.NaT, "hasta": pd.NaT}
            continue
        cambio = (px / px.shift(h) - 1).dropna() @ v
        fin = cambio.idxmin()
        perdida = float(min(0.0, cambio.loc[fin]))
        filas[nombre] = {"perdida": perdida, "perdida_pct": perdida / total if total else 0.0,
                         "desde": fin - pd.Timedelta(days=h), "hasta": fin}
    out = pd.DataFrame.from_dict(filas, orient="index")
    return out.astype({"perdida": float, "perdida_pct": float})


def _cartera_diaria(cierres: pd.DataFrame, valores: dict[str, float]) -> tuple[pd.Series, pd.DataFrame]:
    v = pd.Series({a: x for a, x in valores.items() if x > 0}, dtype=float)
    return v, cierres[list(v.index)].dropna().pct_change().dropna()


def mal_dia(cierres: pd.DataFrame, valores: dict[str, float], efectivo: float,
            alpha: float = 0.95, dias: int = 730) -> dict:
    """Perdida de un dia que solo se supera 1 de cada 1/(1-alpha) dias (historico).

    Con los retornos de los ultimos `dias` y las unidades de hoy. Es el metodo
    mas simple; `riesgo.cartera.analizar` lo contrasta con otros tres.
    """
    v, R = _cartera_diaria(cierres, valores)
    total = float(v.sum() + efectivo)
    if v.empty or total <= 0:
        return {"var": 0.0, "cvar": 0.0, "var_pct": 0.0, "cvar_pct": 0.0}
    x = (R.iloc[-dias:] @ (v / total)).to_numpy()
    q, es = var.var_es(x, "historico", alpha)
    return {"var": q * total, "cvar": es * total, "var_pct": q, "cvar_pct": es}


def proximo_mes(cierres: pd.DataFrame, valores: dict[str, float], efectivo: float,
                dias: int = 21, n: int = 10_000, bloque: int = 10, semilla: int = 42) -> dict:
    """Monte Carlo del valor de la cartera, sin tocar las unidades.

    Remuestrea bloques de dias historicos de todos los activos a la vez, asi
    se conservan las correlaciones y las rachas. Mismo metodo que
    `riesgo.cartera.analizar`.
    """
    v, R = _cartera_diaria(cierres, valores)
    total = float(v.sum() + efectivo)
    if v.empty or total <= 0:
        return futuro.resumen_caminos(np.ones((1, dias + 1)))
    idx = futuro.indices_bloques(len(R), dias, n, bloque, np.random.default_rng(semilla))
    crec = np.cumprod(1 + R.to_numpy()[idx], axis=1)
    caminos = (crec @ v.to_numpy() + efectivo) / total
    return futuro.resumen_caminos(np.hstack([np.ones((n, 1)), caminos]))
