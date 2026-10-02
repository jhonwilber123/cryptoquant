"""Contraseña del piloto cuando se publica en un servidor.

En su equipo no hace falta: la app solo escucha en 127.0.0.1. En un servidor
(Docker en DigitalOcean, ver deploy/LEEME.md) cualquiera que conozca la
direccion llegaria a ella, asi que la imagen exige una contraseña:

- `PILOTO_CLAVE_HASH` guarda su derivada PBKDF2-SHA256, nunca la contraseña.
  Se crea con `python -m cryptoquant piloto --crear-clave`.
- `PILOTO_EXIGIR_CLAVE=1` (fijado en la imagen) impide abrir la app sin ella.

La derivada se escribe con ':' y no con '$', que Docker Compose interpretaria
como una variable al leer el fichero de entorno.
"""
from __future__ import annotations

import hashlib
import hmac
import os

ALGORITMO = "pbkdf2_sha256"
ITERACIONES = 600_000   # recomendacion de OWASP para PBKDF2-SHA256
MINIMO = 12             # caracteres
ESPERA_TRAS_FALLO = 2.0  # segundos: frena los intentos a ciegas


def _derivar(clave: str, sal: bytes, iteraciones: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), sal, iteraciones)


def crear(clave: str, sal: bytes | None = None, iteraciones: int = ITERACIONES) -> str:
    """'pbkdf2_sha256:iteraciones:sal:derivada' (hexadecimal) para PILOTO_CLAVE_HASH."""
    if len(clave) < MINIMO:
        raise ValueError(f"la contraseña debe tener al menos {MINIMO} caracteres")
    sal = os.urandom(16) if sal is None else sal
    return f"{ALGORITMO}:{iteraciones}:{sal.hex()}:{_derivar(clave, sal, iteraciones).hex()}"


def comprobar(clave: str, guardada: str) -> bool:
    """True si `clave` es la contraseña de `guardada`. Una derivada mal escrita no deja entrar."""
    try:
        algoritmo, iteraciones, sal, derivada = guardada.strip().split(":")
        if algoritmo != ALGORITMO or int(iteraciones) < 1:
            return False
        calculada = _derivar(clave, bytes.fromhex(sal), int(iteraciones))
        return hmac.compare_digest(calculada, bytes.fromhex(derivada))
    except ValueError:
        return False


def configuracion(entorno=None) -> tuple[str | None, bool]:
    """(derivada guardada o None, si se exige contraseña), leidas del entorno."""
    entorno = os.environ if entorno is None else entorno
    guardada = (entorno.get("PILOTO_CLAVE_HASH") or "").strip() or None
    return guardada, entorno.get("PILOTO_EXIGIR_CLAVE", "").strip() == "1"
