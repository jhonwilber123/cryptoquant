"""De la cartera guardada al plan de hoy: el recorrido que comparten la app y
`python -m cryptoquant piloto --texto`.

Las fuentes de datos se inyectan (`cierres_de`, `precios_de`) para que la app
use sus versiones en cache y los tests no necesiten red.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..riesgo.cartera import EFECTIVO
from . import mercado, motor


@dataclass
class Sesion:
    plan: motor.Plan
    cierres: pd.DataFrame
    precios: dict[str, float]
    sin_precio_ahora: list[str]       # se valoraron con el ultimo cierre
    candidatas: dict[str, float]      # monedas (y pesos) entre las que se reparte
    mezcla: dict[str, float] | None   # la que decide; None = la actual
    mezcla_efectiva: dict[str, float]
    valor: pd.Series                  # valor de la cuenta sin ingresos ni retiros
    caida: float
    avisos: list[str]


def calcular(tenencias: dict[str, float], reparto: str, medida: dict[str, float] | None,
             fotos: list[dict], ajustes: motor.Ajustes,
             cierres_de=None, precios_de=None) -> Sesion:
    """`medida`: el reparto a medida (el de la pantalla o el guardado).

    Con todo en estables y sin reparto a medida se decide sobre BTC.
    """
    aj = ajustes.validar()
    cierres_de = cierres_de or mercado.cierres
    precios_de = precios_de or mercado.precios_ahora
    cripto, efectivo = motor.separar(tenencias)
    if not cripto and efectivo <= 0:
        raise ValueError("la cartera esta vacia: anote lo que tiene")
    candidatas = motor.candidatas_mezcla(reparto, cripto, medida)
    necesarios = tuple(sorted(set(cripto) | set(candidatas)))

    cierres, avisos = cierres_de(necesarios)
    avisos = list(avisos)
    # Monedas que solo estan en fotos pasadas: hacen falta para medir la caida.
    viejas = sorted({a for f in fotos for a, u in f["tenencias"].items()
                     if a not in EFECTIVO and u > 0} - set(necesarios))
    for a in viejas:
        try:
            extra, av = cierres_de((a,))
            cierres = cierres.join(extra[[a]], how="outer")
            avisos += list(av)
        except ValueError:
            avisos.append(f"{a}: sin precios; los dias en que la tenia no cuentan para la caida")

    ahora = precios_de(necesarios)
    precios = {a: float(ahora[a]) if a in ahora else float(cierres[a].dropna().iloc[-1])
               for a in necesarios}
    sin_ahora = [a for a in necesarios if a not in ahora]

    mezcla = motor.resolver_mezcla(reparto, cripto, candidatas, cierres, aj.lam)
    valor = motor.valor_cuenta(fotos, cierres, ahora=precios)
    caida = motor.caida_vigente(valor, int(aj.freno_ventana))
    plan = motor.decidir(tenencias, precios, cierres, mezcla, caida, aj)
    efectiva = mezcla or {a: u * precios[a] for a, u in cripto.items()}
    return Sesion(plan, cierres, precios, sin_ahora, candidatas, mezcla, efectiva, valor, caida, avisos)
