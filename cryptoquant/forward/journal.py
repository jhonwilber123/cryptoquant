"""Diario de decisiones a prueba de manipulacion.

Un forward test solo vale si es imposible retocarlo a posteriori. Si puedes
editar una prediccion pasada despues de ver el resultado, no estas midiendo
nada: estas escribiendo un diario favorable.

Por eso el registro es de solo-anadido y esta encadenado por hashes: cada
anotacion incluye el hash de la anterior, asi que alterar cualquier linea
antigua rompe la cadena desde ahi hasta el final y `verify()` lo detecta.
No impide el fraude deliberado -- se puede reescribir todo el fichero -- pero
si hace que el retoque silencioso sea imposible, que es el fallo realista.

Ademas se pre-registra la hipotesis ANTES de empezar. Sin eso, al cabo de seis
meses siempre se encuentra alguna metrica en la que el sistema sale bien; el
pre-registro fija de antemano que se mide y con que umbral.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import DATA_DIR, WORKING_COPY_MARKER

FORWARD_DIR = DATA_DIR / "forward"
JOURNAL_PATH = FORWARD_DIR / "journal.jsonl"
PREREG_PATH = FORWARD_DIR / "preregistration.json"
AMENDMENTS_PATH = FORWARD_DIR / "amendments.json"

GENESIS = "0" * 64


class JournalCorrupt(ValueError):
    """Una linea del diario no se puede leer."""


class WorkingCopyError(PermissionError):
    """Se intento escribir el experimento desde la copia de trabajo."""


def _ensure_writable(path: Path) -> None:
    """El experimento solo se escribe en la carpeta que recolecta.

    Hay dos copias del proyecto: la del PC, donde la tarea programada anota
    cada dia, y la del USB, donde se trabaja. Si la del USB anotara tambien,
    habria dos diarios avanzando en paralelo y el experimento se bifurcaria.
    La copia del USB lleva un fichero marcador y aqui se niega a escribir.
    """
    try:
        inside = path.resolve().is_relative_to(DATA_DIR.resolve())
    except OSError:
        inside = True
    if WORKING_COPY_MARKER.exists() and inside:
        raise WorkingCopyError(
            "esta es la copia de trabajo: el diario solo se escribe en la carpeta "
            "que recolecta (ver COPIA_DE_TRABAJO.txt). Para traer aqui los datos "
            "nuevos: scripts\\sincronizar.ps1 -Datos"
        )


def _atomic_write(path: Path, text: str) -> None:
    """Escribe el fichero entero de golpe: o queda el anterior o el nuevo.

    El equipo se ha apagado de forma inesperada varias veces. Anadir una linea
    con `open("a")` en ese momento puede dejarla a medias, y una linea cortada
    en mitad del diario lo invalida. Con escritura a un temporal + rename
    atomico eso no puede pasar.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _canonical(obj: Any) -> str:
    """Serializacion determinista: el hash debe depender del contenido, nunca
    del orden de las claves ni del espaciado."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def _digest(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Pre-registro
# --------------------------------------------------------------------------
DEFAULT_HYPOTHESES = [
    {
        "id": "H1",
        "rank": "primaria",
        "claim": "El control de riesgo entrega la volatilidad objetivo en vivo.",
        "metric": "volatilidad anualizada realizada mientras se esta invertido",
        "success": "dentro de +-40% relativo del objetivo configurado",
        "min_observations": 180,
        "testable": True,
        "note": "La volatilidad converge rapido: con 180 observaciones el "
                "estimador tiene un error relativo de ~5%. Esto SI se resuelve "
                "en unos seis meses.",
    },
    {
        "id": "H2",
        "rank": "primaria",
        "claim": "Las decisiones en vivo reproducen las del backtest.",
        "metric": "diferencia entre pesos registrados y pesos recalculados",
        "success": "diferencia maxima por activo < 1 punto porcentual",
        "min_observations": 30,
        "testable": True,
        "note": "Detecta fallos de implementacion: datos rancios, desalineacion "
                "de fechas, fugas. Es el que mas probable es que falle.",
    },
    {
        "id": "H3",
        "rank": "secundaria",
        "claim": "El sistema bate al buy & hold ajustado a la misma volatilidad.",
        "metric": "diferencia de retorno frente al indice escalado",
        "success": "IC95% por bootstrap de bloques que excluya el cero",
        "min_observations": 16969,
        "testable": False,
        "note": "NO es resoluble. Con la ventaja observada (+2.5% anual) y el "
                "tracking error (8.8% anual) harian falta ~46 anos para "
                "alcanzar p<0.05, y ~95 para hacerlo con potencia del 80%. Se "
                "registra por honestidad, no porque vaya a concluir nada.",
    },
]


@dataclass
class Preregistration:
    created_utc: str
    hypotheses: list[dict]
    config_snapshot: dict
    universe: list[str]
    code_version: str
    hash: str

    @property
    def short_hash(self) -> str:
        return self.hash[:12]


def write_preregistration(cfg, universe: list[str], code_version: str,
                          path: Path | None = None, overwrite: bool = False) -> Preregistration:
    """Fija la hipotesis antes de tener ni un solo dato."""
    path = path or PREREG_PATH
    _ensure_writable(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"ya existe un pre-registro en {path}. Reescribirlo invalida el "
            "experimento: las hipotesis se fijan una vez, al principio."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_utc": _now(),
        "hypotheses": DEFAULT_HYPOTHESES,
        "config_snapshot": cfg.to_dict(),
        "universe": sorted(universe),
        "code_version": code_version,
    }
    payload["hash"] = _digest(payload)
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return Preregistration(**payload)


def read_preregistration(path: Path | None = None) -> Preregistration | None:
    path = path or PREREG_PATH
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Preregistration(**raw)


def preregistration_is_intact(path: Path | None = None) -> bool:
    prereg = read_preregistration(path)
    if prereg is None:
        return False
    return _digest(prereg.__dict__) == prereg.hash


# --------------------------------------------------------------------------
# Diario
# --------------------------------------------------------------------------
def read_journal(path: Path | None = None) -> list[dict]:
    path = path or JOURNAL_PATH
    if not path.exists():
        return []
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise JournalCorrupt(
                f"la linea {n} del diario no es legible ({exc.msg}); probablemente "
                "una escritura interrumpida. No se toca nada automaticamente."
            ) from exc
    return records


def iter_journal(path: Path | None = None) -> Iterator[dict]:
    yield from read_journal(path)


def last_record(path: Path | None = None) -> dict | None:
    records = read_journal(path)
    return records[-1] if records else None


def append_record(
    decision_date: str,
    prices: dict[str, float],
    weights: dict[str, float],
    gross_exposure: float,
    diagnostics: dict[str, Any],
    cfg,
    code_version: str,
    path: Path | None = None,
    note: str = "",
    state_in: dict[str, Any] | None = None,
) -> dict:
    """Anade una decision al diario, encadenada a la anterior.

    `decision_date` es la fecha de la barra sobre la que se decide. El registro
    debe crearse ANTES de conocer el precio siguiente; de lo contrario el
    ejercicio pierde todo su sentido.

    `state_in` guarda las ENTRADAS con las que se tomo la decision (pesos
    previos y drawdown vigente). Sin ellas la decision no es reproducible: el
    filtro de rotacion depende de la cartera anterior, asi que recomputar con
    otro estado inicial da otro resultado y produciria falsos fallos en H2.
    """
    path = path or JOURNAL_PATH
    _ensure_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = read_journal(path)
    if any(r["decision_date"] == decision_date for r in records):
        raise ValueError(
            f"ya hay una anotacion para {decision_date}. Reescribir una decision "
            "ya registrada es exactamente lo que este diario existe para impedir."
        )

    prereg = read_preregistration()
    amendment = current_amendment()
    if amendment and decision_date < amendment["effective_from"]:
        raise ValueError(
            f"{decision_date} es anterior a la enmienda {amendment['id']} "
            f"(vigente desde {amendment['effective_from']})."
        )
    payload = {
        "seq": len(records) + 1,
        "timestamp_utc": _now(),
        "decision_date": decision_date,
        "prices": {k: float(v) for k, v in prices.items()},
        "weights": {k: round(float(v), 8) for k, v in weights.items()},
        "gross_exposure": round(float(gross_exposure), 8),
        "cash": round(1.0 - float(gross_exposure), 8),
        "diagnostics": diagnostics,
        "target_volatility": float(cfg.risk.target_volatility),
        "config_hash": hashlib.sha256(
            _canonical(cfg.to_dict()).encode("utf-8")).hexdigest(),
        "preregistration_hash": prereg.hash if prereg else None,
        # Protocolo con el que se decidio: la enmienda vigente, si la hay.
        "amendment_hash": amendment["hash"] if amendment else None,
        "code_version": code_version,
        "note": note,
        "state_in": state_in or {"prev_weights": {}, "drawdown": 0.0},
        "prev_hash": records[-1]["hash"] if records else GENESIS,
    }
    payload["hash"] = _digest(payload)

    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old and not old.endswith("\n"):
        old += "\n"
    _atomic_write(path, old + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return payload


# --------------------------------------------------------------------------
# Enmiendas al protocolo
# --------------------------------------------------------------------------
# Corregir un fallo del sistema a mitad de experimento es legitimo; hacerlo en
# silencio no. Una enmienda queda escrita ANTES de la primera anotacion a la
# que afecta, enlazada al pre-registro y a la ultima anotacion existente (asi
# se ve en que momento del experimento se hizo), y con su propio hash. Las
# hipotesis no cambian: ni lo que afirman, ni el umbral, ni cuantos datos
# necesitan. Lo que cambia es desde cuando se cuentan.
AMENDMENT_E1 = {
    "id": "E1",
    "title": "El modo en vivo sigue el calendario del backtest",
    "reason": (
        "Con 7 anotaciones se detecto que el registro diario reentrenaba el "
        "modelo y rebalanceaba en cada ejecucion, mientras el backtest reentrena "
        "cada 63 barras (65 efectivas) y rebalancea cada 5. La rotacion en vivo "
        "fue ~12 veces la del backtest: costes a ritmo de ~11.5% anual frente a "
        "~1%. El forward test estaba midiendo otra estrategia. H2 no lo detecto "
        "porque solo comprobaba que recalcular una fecha diera lo mismo."
    ),
    "changes": [
        "Calendario anclado en 'anchor': rebalanceo cada rebalance_every barras "
        "y reentrenamiento cada 65 (primer multiplo de 5 >= 63), como el "
        "backtest. Entre rebalanceos se mantienen los pesos objetivo.",
        "Si un rebalanceo programado cae en un dia sin anotar, se hace en la "
        "primera anotacion posterior, con el modelo del calendario. No se "
        "rellenan dias.",
        "El filtro de rotacion y el drawdown usan la cartera derivada por precio "
        "hasta la fecha de decision, como el backtest.",
        "Costes realizados: incluyen volver a los pesos objetivo tras la deriva "
        "de precios, como el backtest.",
        "H1: cada tramo se normaliza por su duracion en dias. Antes se trataba "
        "como un dia y la volatilidad salia inflada (19.8% frente a 15.5% con "
        "los datos de la fase 1). Umbral sin cambios.",
        "H2: ademas de los pesos, se comprueba que la accion (rebalancear o "
        "mantener) y la fecha de corte del modelo salen igual al recalcular. "
        "La equivalencia con el backtest la cubre un test automatico que "
        "ejecuta el modo en vivo sobre el estado del backtest.",
        "H1 y H2 cuentan solo desde 'effective_from'. Las anotaciones anteriores "
        "quedan en el diario, sin modificar, como fase 1.",
    ],
    "hypotheses_changed": False,
}


def read_amendments(path: Path | None = None) -> list[dict]:
    path = path or AMENDMENTS_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def current_amendment(path: Path | None = None) -> dict | None:
    amendments = read_amendments(path)
    return amendments[-1] if amendments else None


def amendment_by_hash(h: str | None, path: Path | None = None) -> dict | None:
    if not h:
        return None
    return next((a for a in read_amendments(path) if a["hash"] == h), None)


def write_amendment(spec: dict, anchor: str, cfg, code_version: str,
                    path: Path | None = None) -> dict:
    """Registra una enmienda que rige desde la barra `anchor` inclusive."""
    path = path or AMENDMENTS_PATH
    _ensure_writable(path)
    amendments = read_amendments(path)
    if any(a["id"] == spec["id"] for a in amendments):
        raise FileExistsError(f"la enmienda {spec['id']} ya esta registrada")
    records = read_journal()
    if records and records[-1]["decision_date"] >= anchor:
        raise ValueError(
            f"ya hay anotaciones en o despues de {anchor}: una enmienda no puede "
            "aplicarse a decisiones ya registradas."
        )
    prereg = read_preregistration()
    payload = {
        **spec,
        "created_utc": _now(),
        "anchor": anchor,
        "effective_from": anchor,
        "preregistration_hash": prereg.hash if prereg else None,
        "journal_head": ({"seq": records[-1]["seq"], "hash": records[-1]["hash"]}
                         if records else None),
        "config_hash": hashlib.sha256(_canonical(cfg.to_dict()).encode("utf-8")).hexdigest(),
        "code_version": code_version,
        "prev_hash": amendments[-1]["hash"] if amendments else GENESIS,
    }
    payload["hash"] = _digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(amendments + [payload], indent=2,
                                   ensure_ascii=False, default=str))
    return payload


def active_records(records: list[dict], amendment: dict | None = None) -> list[dict]:
    """Anotaciones que cuentan para las hipotesis: las del protocolo vigente."""
    if amendment is None:
        return records
    return [r for r in records if r["decision_date"] >= amendment["effective_from"]]


@dataclass
class ChainStatus:
    ok: bool
    n_records: int
    first_break: int | None
    problems: list[str]


def verify_chain(path: Path | None = None) -> ChainStatus:
    """Recorre la cadena y comprueba que nadie ha tocado el pasado."""
    try:
        records = read_journal(path)
    except JournalCorrupt as exc:
        return ChainStatus(ok=False, n_records=0, first_break=None, problems=[str(exc)])
    problems: list[str] = []
    first_break: int | None = None
    prev = GENESIS

    for i, rec in enumerate(records, start=1):
        if rec.get("seq") != i:
            problems.append(f"registro {i}: numero de secuencia {rec.get('seq')} inesperado")
            first_break = first_break or i
        if rec.get("prev_hash") != prev:
            problems.append(f"registro {i}: no enlaza con el anterior (cadena rota)")
            first_break = first_break or i
        expected = _digest(rec)
        if rec.get("hash") != expected:
            problems.append(f"registro {i} ({rec.get('decision_date')}): "
                            "el contenido no coincide con su hash (fue modificado)")
            first_break = first_break or i
        prev = rec.get("hash", GENESIS)

    if not preregistration_is_intact() and read_preregistration() is not None:
        problems.append("el pre-registro fue modificado despues de crearse")

    dates = [r["decision_date"] for r in records]
    if dates != sorted(dates):
        problems.append("las fechas de decision no son crecientes")

    # Enmiendas: intactas, encadenadas y escritas antes de lo que gobiernan.
    by_seq = {r.get("seq"): r for r in records}
    amendments = read_amendments()
    prev = GENESIS
    for a in amendments:
        if _digest(a) != a.get("hash") or a.get("prev_hash") != prev:
            problems.append(f"la enmienda {a.get('id')} fue modificada despues de crearse")
        head = a.get("journal_head")
        if head and by_seq.get(head["seq"], {}).get("hash") != head["hash"]:
            problems.append(f"la enmienda {a.get('id')} no enlaza con el diario")
        prev = a.get("hash", GENESIS)
    # Cada anotacion debe seguir el protocolo vigente en su fecha: ni una
    # anterior a la enmienda puede decir que la sigue, ni una posterior saltarsela.
    for r in records:
        rule = [a for a in amendments if a["effective_from"] <= r["decision_date"]]
        expected = rule[-1]["hash"] if rule else None
        if r.get("amendment_hash") != expected:
            problems.append(f"registro {r.get('seq')} ({r.get('decision_date')}): "
                            "no sigue el protocolo vigente en su fecha")

    return ChainStatus(ok=not problems, n_records=len(records),
                       first_break=first_break, problems=problems)
