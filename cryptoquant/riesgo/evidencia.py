"""Filtro de evidencia: lo que una idea para ganar dinero debe superar.

Casi todas las ideas de trading parecen funcionar cuando se miran sobre el
mismo historico con el que se pensaron. Este filtro existe para que ninguna
llegue al sistema por esa via. Una senal pasa solo si cumple las cinco
condiciones, y se evaluan en este orden porque las primeras invalidan las
demas:

1. Causal. Recalculada sobre la serie cortada, sus valores pasados no
   cambian, en ninguno de los activos. Si cambian, mira al futuro y todo lo
   demas es ficcion.
2. Solo fuera de muestra. Se mide en el ultimo 30% de cada activo.
3. Suficientes eventos (>= 30) para que un porcentaje signifique algo.
4. Bate a su base con correccion por comparaciones multiples. La base es
   operar en la misma direccion todos los dias. El umbral es 0.05 dividido
   entre el numero de ideas distintas evaluadas hasta hoy, que se lleva en
   `data/evidencia/registro.jsonl`: cada idea probada endurece el umbral de
   todas las siguientes. Es el unico antidoto contra probar hasta que salga.
   El registro esta encadenado por hashes, como el diario del forward test:
   borrar o editar una evaluacion pasada (que ablandaria el umbral) rompe la
   cadena, y el filtro se niega a evaluar hasta que se restaure.
5. Gana neto de costes: retorno medio por operacion positivo tras pagar
   comision y deslizamiento de ida y vuelta.

Una senal que pasa no se incorpora sola: pasa a ser candidata. Anadirla al
sistema cambiaria la estrategia del forward test y exigiria una enmienda.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as cfgmod

Generador = Callable[[pd.DataFrame], pd.Series]


class RegistroCorrupto(RuntimeError):
    """El registro de ideas no supera su propia verificacion."""


def ruta_registro():
    d = cfgmod.DATA_DIR / "evidencia"
    d.mkdir(parents=True, exist_ok=True)
    return d / "registro.jsonl"


@dataclass
class Veredicto:
    senal: str
    horizonte: int
    pasa: bool
    motivos: list[str]
    causal: bool
    eventos: int = 0
    acierto: float = np.nan
    acierto_base: float = np.nan
    p_valor: float = np.nan
    umbral: float = np.nan
    ideas_evaluadas: int = 0
    retorno_neto_medio: float = np.nan
    retorno_base_medio: float = np.nan
    desde: str = ""
    por_activo: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# La senal
# --------------------------------------------------------------------------
def _serie(generador: Generador, velas: pd.DataFrame) -> pd.Series:
    """Salida del generador alineada con las velas y en {-1, 0, +1}.

    Admite booleanos (True = compra), magnitudes (solo cuenta el signo), NaN
    (sin senal) y un indice distinto del de las velas (se alinea por fecha).
    """
    try:
        s = generador(velas)
    except Exception as exc:  # noqa: BLE001 - se informa con el nombre de la causa
        raise ValueError(f"el generador de la senal fallo: {type(exc).__name__}: {exc}") from exc
    if not isinstance(s, pd.Series):
        raise ValueError("el generador debe devolver una pandas.Series indexada por fecha")
    s = pd.to_numeric(s.astype(float) if s.dtype == bool else s, errors="coerce")
    return np.sign(s.reindex(velas.index)).fillna(0.0)


def es_causal(generador: Generador, velas: pd.DataFrame, cortes=(0.5, 0.7, 0.9)) -> bool:
    completa = _serie(generador, velas)
    for f in cortes:
        k = int(len(velas) * f)
        if k < 2:
            continue
        corta = _serie(generador, velas.iloc[:k])
        if not np.array_equal(completa.iloc[:k].to_numpy(), corta.to_numpy()):
            return False
    return True


# --------------------------------------------------------------------------
# El registro, encadenado
# --------------------------------------------------------------------------
def _huella(senal: str, params: dict) -> str:
    return hashlib.sha256(json.dumps({"senal": senal, **params}, sort_keys=True)
                          .encode()).hexdigest()[:16]


def _sello(entrada: dict) -> str:
    cuerpo = {k: v for k, v in entrada.items() if k != "hash"}
    return hashlib.sha256(json.dumps(cuerpo, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def _leer_registro() -> list[dict]:
    """Lee y verifica el registro. Cualquier alteracion es un error, no un cero."""
    ruta = ruta_registro()
    if not ruta.exists():
        return []
    entradas, anterior = [], None
    for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip():
            continue
        try:
            e = json.loads(linea)
        except json.JSONDecodeError:
            raise RegistroCorrupto(f"{ruta}: la linea {n} no es JSON valido") from None
        if not isinstance(e, dict) or e.get("anterior") != anterior or e.get("hash") != _sello(e):
            raise RegistroCorrupto(
                f"{ruta}: la cadena de hashes se rompe en la linea {n}. Alguien borro o edito "
                "una evaluacion, y eso ablandaria el umbral de todas las demas. Restaurelo "
                "desde git: git checkout -- data/evidencia/registro.jsonl")
        entradas.append(e)
        anterior = e["hash"]
    return entradas


def _anotar(v: Veredicto, huella: str, params: dict, previas: list[dict]) -> None:
    e = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "huella": huella, "senal": v.senal, "parametros": params,
        "pasa": v.pasa, "p_valor": None if np.isnan(v.p_valor) else v.p_valor,
        "eventos": v.eventos, "anterior": previas[-1]["hash"] if previas else None,
    }
    e["hash"] = _sello(e)
    with ruta_registro().open("a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")


def ideas_evaluadas() -> int:
    return len({r["huella"] for r in _leer_registro()})


# --------------------------------------------------------------------------
# La evaluacion
# --------------------------------------------------------------------------
def evaluar_senal(nombre: str, generador: Generador, velas: dict[str, pd.DataFrame],
                  horizonte: int = 5, frac_prueba: float = 0.3, coste_ida_vuelta: float = 0.003,
                  min_eventos: int = 30, registrar: bool = True) -> Veredicto:
    """Evalua una senal sobre varios activos.

    Args:
        generador: funcion velas -> Serie con +1 (compra), -1 (venta) o 0 en la
            vela en que se conoce la senal. Una probabilidad se convierte antes
            con `np.sign(p - 0.5)`.
        velas: OHLCV diario por activo.
        coste_ida_vuelta: comision + deslizamiento de entrar y salir (0.3% por
            defecto: 10 pb + 5 pb por lado, como el backtest).
    """
    if not velas:
        raise ValueError("no hay velas sobre las que evaluar la senal")
    if isinstance(horizonte, bool) or not isinstance(horizonte, (int, np.integer)) or horizonte < 1:
        raise ValueError("el horizonte debe ser un entero de al menos 1 dia")
    if not 0 < frac_prueba < 1:
        raise ValueError("frac_prueba debe estar entre 0 y 1")
    if not str(nombre).strip():
        raise ValueError("la senal necesita un nombre")
    previas = _leer_registro()  # lo primero: con el registro roto no se evalua nada

    motivos: list[str] = []
    # En todos los activos: una senal puede comportarse distinto en cada uno,
    # y basta que mire al futuro en uno para invalidarla.
    no_causales = [a for a, df in velas.items() if not es_causal(generador, df)]
    causal = not no_causales
    v = Veredicto(nombre, horizonte, False, motivos, causal)
    if not causal:
        motivos.append("mira al futuro en " + ", ".join(no_causales) +
                       ": sus valores pasados cambian al cortar la serie")
    else:
        dir_ret, neto, base_acierto, base_ret = [], [], [], []
        for activo, df in velas.items():
            s = _serie(generador, df)
            lc = np.log(df["close"])
            fwd = lc.shift(-horizonte) - lc
            corte = min(int(len(df) * (1 - frac_prueba)), len(df) - 1)
            s, fwd = s.iloc[corte:], fwd.iloc[corte:]
            ok = fwd.notna()
            ev = (s != 0) & ok
            d = s[ev].to_numpy()
            dr = d * fwd[ev].to_numpy()
            # Base de cada evento: lo que habria dado su misma direccion en un
            # dia cualquiera del tramo de prueba de ese activo.
            p_sube = float((fwd[ok] > 0).mean()) if ok.any() else 0.5
            media = float(fwd[ok].mean()) if ok.any() else 0.0
            dir_ret.append(dr)
            neto.append(d * np.expm1(fwd[ev].to_numpy()))  # un corto gana -(P1/P0 - 1)
            base_acierto.append(np.where(d > 0, p_sube, 1 - p_sube))
            base_ret.append(d * media)
            v.por_activo[activo] = {"eventos": int(ev.sum()),
                                    "acierto": float((dr > 0).mean()) if len(dr) else None}
            v.desde = min(v.desde or str(df.index[corte].date()), str(df.index[corte].date()))
        dr = np.concatenate(dir_ret)
        v.eventos = len(dr)
        if v.eventos:
            v.acierto_base = float(np.concatenate(base_acierto).mean())
            v.retorno_base_medio = float(np.concatenate(base_ret).mean())
            aciertos = int((dr > 0).sum())
            v.acierto = aciertos / v.eventos
            v.p_valor = float(stats.binomtest(aciertos, v.eventos, v.acierto_base,
                                              alternative="greater").pvalue)
            v.retorno_neto_medio = float(np.concatenate(neto).mean() - coste_ida_vuelta)

    params = {"horizonte": int(horizonte), "frac_prueba": frac_prueba, "activos": sorted(velas)}
    huella = _huella(nombre, params)
    v.ideas_evaluadas = len({r["huella"] for r in previas} | {huella})
    v.umbral = 0.05 / v.ideas_evaluadas
    if causal:
        if v.eventos < min_eventos:
            motivos.append(f"solo {v.eventos} eventos fuera de muestra (minimo {min_eventos})")
        if v.eventos and not v.p_valor < v.umbral:
            motivos.append(f"no bate a su base: p = {v.p_valor:.3f}, se exige < {v.umbral:.4f} "
                           f"({v.ideas_evaluadas} ideas evaluadas)")
        if v.eventos and not v.retorno_neto_medio > 0:
            motivos.append(f"pierde tras costes: {v.retorno_neto_medio:+.2%} por operacion")
    v.pasa = causal and not motivos
    if registrar:
        _anotar(v, huella, params, previas)
    return v


def a_dict(v: Veredicto) -> dict:
    return asdict(v)
