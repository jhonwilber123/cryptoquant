"""Recoleccion reproducible de antecedentes y libros para la investigacion.

Por que un script y no una busqueda "a mano" con un modelo de lenguaje: un
modelo puede inventar referencias o elegir distinto cada vez. Aqui las
consultas, filtros y cupos estan fijados en protocolo.yaml; los metadatos salen
de bases de datos (OpenAlex, Crossref, ALICIA/CONCYTEC, Open Library) y no de
la memoria de nadie; y cada ejecucion guarda las respuestas crudas con fecha y
la huella del protocolo. Cualquiera puede auditar por que entro cada documento.

Solo se descargan documentos de acceso abierto o publicados gratis por su autor
o editorial. Del resto se guarda la referencia verificada.

Uso (desde la carpeta del proyecto):
    python investigacion/recolectar.py              todo
    python investigacion/recolectar.py --solo libros
    python investigacion/recolectar.py --sin-pdf    solo metadatos
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml

BASE = Path(__file__).resolve().parent
HOY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OPENALEX = "https://api.openalex.org"
ALICIA = "https://alicia.concytec.gob.pe/vufind/api/v1/search"
UA = "cryptoquant-investigacion/1.0 (revision de literatura academica)"
NIVELES = ("internacional", "nacional", "local")
ORDEN_REVISTA = {"A": 0, "B": 1, "C": 2, "T": 3}
TIPO_TESIS = {"doctoralThesis": "tesis doctoral", "masterThesis": "tesis de maestria",
              "bachelorThesis": "tesis de licenciatura"}


# ==========================================================================
# Utilidades
# ==========================================================================
class Web:
    """Sesion HTTP con reintentos, pausa entre llamadas y cache en disco."""

    def __init__(self, cache_dir: Path, log):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.last = 0.0
        # OpenAlex da a cada IP un presupuesto diario gratuito (se reinicia a
        # medianoche UTC); una clave gratuita da uno propio.
        self.openalex_key = os.environ.get("OPENALEX_API_KEY", "").strip()
        kf = BASE / "clave_openalex.txt"
        if not self.openalex_key and kf.exists():
            self.openalex_key = kf.read_text(encoding="utf-8").strip()
        self.openalex_blocked = False

    def get(self, url, params=None, headers=None, timeout=(10, 40), stream=False, pause=0.15,
            intentos=3):
        if url.startswith(OPENALEX) and self.openalex_key:
            headers = {**(headers or {}), "Authorization": f"Bearer {self.openalex_key}"}
        for intento in range(intentos):
            espera = pause - (time.time() - self.last)
            if espera > 0:
                time.sleep(espera)
            self.last = time.time()
            try:
                r = self.s.get(url, params=params, headers=headers, timeout=timeout,
                               stream=stream, allow_redirects=True)
                if r.status_code == 429 and "budget" in r.text.lower():
                    return r                   # presupuesto agotado: reintentar no sirve
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (intento + 1))
                    continue
                return r
            except requests.RequestException as exc:
                self.log(f"    red: {exc.__class__.__name__} en {url[:90]}")
                time.sleep(2 * (intento + 1))
        return None

    def openalex(self, path: str, params: dict, scope: str) -> dict | None:
        """Consulta a OpenAlex con cache en disco.

        `scope` es la fecha para las busquedas (se repiten, como mucho, una vez
        al dia) o "permanente" para datos que no cambian (ids de revistas,
        ficha de un libro). Repetir una ejecucion el mismo dia no gasta
        presupuesto.
        """
        key = json.dumps([path, sorted(params.items()), scope], ensure_ascii=False)
        p = self.cache / "openalex" / (hashlib.sha1(key.encode()).hexdigest() + ".json")
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        if self.openalex_blocked:
            return None
        r = self.get(f"{OPENALEX}/{path}", params=params, timeout=(15, 120))
        if r is None:
            return None
        if r.status_code == 429 and "budget" in r.text.lower():
            self.openalex_blocked = True
            self.log("    OpenAlex: presupuesto diario agotado (se reinicia a medianoche UTC, "
                     "las 19:00 en Peru). Resultados parciales; repetir despues o con clave.")
            return None
        if not r.ok:
            self.log(f"    OpenAlex respondio {r.status_code} en {path}")
            return None
        j = r.json()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
        return j

    def cached_text(self, key: str, fetch) -> str | None:
        """Guarda en disco respuestas que no cambian (BibTeX, cita APA)."""
        p = self.cache / (hashlib.sha1(key.encode()).hexdigest() + ".txt")
        if p.exists():
            return p.read_text(encoding="utf-8")
        text = fetch()
        if text:
            p.write_text(text, encoding="utf-8")
        return text


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"^the\s+", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def safe_name(s: str, maxlen: int = 90) -> str:
    """Nombre de fichero valido en FAT32 (el USB): ASCII y sin simbolos."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:maxlen].rstrip("_") or "documento"


def abstract_from_index(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def clean_text(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s or ""))
    return re.sub(r"\s+", " ", s).strip()


# ==========================================================================
# OpenAlex
# ==========================================================================
SELECT = ",".join([
    "id", "doi", "display_name", "publication_year", "publication_date", "type",
    "language", "authorships", "primary_location", "locations", "best_oa_location",
    "open_access", "cited_by_count", "fwci", "abstract_inverted_index", "biblio",
    "is_retracted",
])
# Para cribar basta con esto; el resumen y todas las ubicaciones (lo que mas
# pesa) solo se piden para los seleccionados. Con una conexion lenta es la
# diferencia entre minutos y horas.
SELECT_LIGERO = ",".join([
    "id", "doi", "display_name", "publication_year", "type", "authorships",
    "primary_location", "best_oa_location", "open_access", "cited_by_count", "fwci", "biblio",
])


def openalex_works(web: Web, filtro: str, max_items: int, sort: str | None = None,
                   raw_dir: Path | None = None, etiqueta: str = "",
                   select: str = SELECT_LIGERO) -> list[dict]:
    out, cursor, pagina = [], "*", 0
    while cursor and len(out) < max_items:
        # Cada peticion de busqueda cuesta lo mismo sea cual sea su tamano, asi
        # que se piden paginas de 200: el presupuesto diario rinde el doble.
        params = {"filter": filtro, "per-page": min(200, max_items), "cursor": cursor,
                  "select": select}
        if sort:
            params["sort"] = sort
        j = web.openalex("works", params, scope=HOY)
        if j is None:
            if not web.openalex_blocked:
                web.log(f"    OpenAlex no respondio para {etiqueta}")
            break
        if raw_dir is not None:
            (raw_dir / f"openalex_{safe_name(etiqueta, 40)}_{pagina}.json").write_text(
                json.dumps(j, ensure_ascii=False), encoding="utf-8")
        out.extend(j.get("results", []))
        cursor = (j.get("meta") or {}).get("next_cursor")
        pagina += 1
        if not j.get("results"):
            break
    return out[:max_items]


def resolve_tier_a(web: Web, nombres: list[str], log) -> dict[str, str]:
    """Busca cada revista de nivel A en OpenAlex y guarda su id.

    Solo se acepta una coincidencia exacta del nombre normalizado: una
    coincidencia aproximada podria colar una revista distinta.
    """
    ids: dict[str, str] = {}
    for nombre in nombres:
        if web.openalex_blocked:
            log("    sin presupuesto de OpenAlex: las revistas de nivel A se identifican en la proxima ejecucion")
            break
        # Permanente: el id de una revista no cambia, y resolver las 33 cuesta
        # un tercio del presupuesto diario gratuito.
        j = web.openalex("sources", {"search": nombre, "per-page": 25,
                                     "select": "id,display_name,issn_l,type"}, scope="permanente")
        hit = None
        if j is not None:
            for s in j.get("results", []):
                if norm(s["display_name"]) == norm(nombre) and s.get("type") == "journal":
                    hit = s
                    break
        if hit:
            ids[hit["id"]] = hit["display_name"]
        else:
            log(f"    revista de nivel A no encontrada con nombre exacto: {nombre}")
    return ids


def journal_tier(work: dict, tier_a: dict[str, str]) -> str | None:
    src = (work.get("primary_location") or {}).get("source") or {}
    if src.get("type") != "journal":
        return None
    if src.get("id") in tier_a:
        return "A"
    if src.get("is_core"):
        return "B"
    return "C"


def affiliation_level(work: dict, local_ids: set[str], local_re: re.Pattern) -> str:
    peru = False
    for a in work.get("authorships") or []:
        countries = set(a.get("countries") or [])
        insts = a.get("institutions") or []
        raw = " ".join(a.get("raw_affiliation_strings") or [])
        names = " ".join(i.get("display_name") or "" for i in insts)
        if any((i.get("id") or "") in local_ids for i in insts) or \
                (("PE" in countries or re.search(r"per[uú]", raw, re.I)) and local_re.search(f"{raw} {names}")):
            return "local"
        if "PE" in countries or any(i.get("country_code") == "PE" for i in insts):
            peru = True
    return "nacional" if peru else "internacional"


def pdf_candidates(work: dict) -> list[str]:
    urls = []
    for loc in [work.get("best_oa_location")] + list(work.get("locations") or []):
        if not loc or not loc.get("is_oa"):
            continue
        for key in ("pdf_url", "landing_page_url"):
            u = loc.get(key)
            if u and u not in urls:
                urls.append(u)
    oa = (work.get("open_access") or {}).get("oa_url")
    if oa and oa not in urls:
        urls.append(oa)
    return urls


def record_from_openalex(w: dict, eje: str, consulta: str, tier: str, nivel: str,
                         prioridad: int = 0) -> dict:
    src = (w.get("primary_location") or {}).get("source") or {}
    b = w.get("biblio") or {}
    autores = [a["author"]["display_name"] for a in w.get("authorships") or [] if a.get("author")]
    afil = sorted({i["display_name"] for a in w.get("authorships") or []
                   for i in a.get("institutions") or [] if i.get("country_code") == "PE"})
    return {
        "id": w["id"],
        "nivel": nivel,
        "eje": eje,
        "anio": w.get("publication_year"),
        "autores": "; ".join(autores[:12]) + (" y otros" if len(autores) > 12 else ""),
        "primer_autor": autores[0] if autores else "",
        "titulo": clean_text(w.get("display_name") or ""),
        "revista": src.get("display_name") or "",
        "volumen": b.get("volume") or "",
        "numero": b.get("issue") or "",
        "paginas": "-".join(p for p in (b.get("first_page"), b.get("last_page")) if p),
        "nivel_revista": tier,
        "tipo": "revision" if w.get("type") == "review" else "articulo",
        "fwci": w.get("fwci"),
        "citas": w.get("cited_by_count") or 0,
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "url": w.get("doi") or (w.get("primary_location") or {}).get("landing_page_url") or w["id"],
        "acceso_abierto": (w.get("open_access") or {}).get("oa_status", ""),
        "afiliaciones_peru": "; ".join(afil),
        "fuente": "OpenAlex",
        "consulta": consulta,
        "prioridad": prioridad,
        "resumen": clean_text(abstract_from_index(w.get("abstract_inverted_index"))),
        "_pdfs": pdf_candidates(w),
    }


def enrich(web: Web, sel: list[dict], log) -> None:
    """Completa resumen y enlaces de acceso abierto de los seleccionados."""
    ids = [r["id"].rsplit("/", 1)[-1] for r in sel if r["fuente"] == "OpenAlex"]
    full = {}
    for i in range(0, len(ids), 50):
        for w in openalex_works(web, "openalex:" + "|".join(ids[i:i + 50]), 50, select=SELECT):
            full[w["id"]] = w
    for r in sel:
        w = full.get(r["id"])
        if w:
            r["resumen"] = clean_text(abstract_from_index(w.get("abstract_inverted_index")))
            r["_pdfs"] = pdf_candidates(w)
    log(f"  resumen y enlaces completados para {len(full)} articulos de OpenAlex")


def rank_key(rec: dict):
    return (ORDEN_REVISTA.get(rec["nivel_revista"], 9), -(rec["fwci"] or 0), -rec["citas"])


# ==========================================================================
# ALICIA (CONCYTEC)
# ==========================================================================
def alicia_search(web: Web, lookfor: str, filtros: list[str], limit: int,
                  raw_dir: Path, etiqueta: str) -> list[dict]:
    campos = ["id", "title", "authors", "formats", "publicationDates", "urls",
              "institutions", "languages", "summary"]
    out, page = [], 1
    while len(out) < limit:
        params = {"lookfor": lookfor, "limit": min(100, limit), "page": page,
                  "field[]": campos, "filter[]": filtros}
        r = web.get(ALICIA, params=params, pause=0.5)
        if r is None or not r.ok:
            web.log(f"    ALICIA no respondio para {etiqueta}")
            break
        t = r.content.decode("utf-8", errors="replace")
        try:
            j = json.loads(t[t.index("{"):])      # antepone avisos de PHP al JSON
        except ValueError:
            web.log(f"    ALICIA devolvio algo que no es JSON para {etiqueta}")
            break
        (raw_dir / f"alicia_{safe_name(etiqueta, 40)}_{page}.json").write_text(
            json.dumps(j, ensure_ascii=False), encoding="utf-8")
        recs = j.get("records") or []
        out.extend(recs)
        if len(recs) < params["limit"]:
            break
        page += 1
    return out[:limit]


def record_from_alicia(r: dict, nivel: str, consulta: str, prioridad: int = 0) -> dict | None:
    fechas = [int(m) for d in r.get("publicationDates") or [] for m in re.findall(r"(\d{4})", d)]
    anio = min(fechas) if fechas else None
    fmt = (r.get("formats") or ["other"])[0]
    if fmt == "article":
        tipo, tier = "articulo", "C"
    elif fmt in TIPO_TESIS:
        tipo, tier = TIPO_TESIS[fmt], "T"
    else:
        return None
    a = r.get("authors") or {}
    autores = list((a.get("primary") or {}).keys()) + list((a.get("secondary") or {}).keys())
    urls = [u["url"] for u in r.get("urls") or [] if u.get("url")]
    doi = next((m.group(0) for u in urls for m in [re.search(r"10\.\d{4,9}/\S+", u)] if m), "")
    return {
        "id": "alicia:" + r["id"],
        "nivel": nivel,
        "eje": "",
        "anio": anio,
        "autores": "; ".join(autores[:12]),
        "primer_autor": autores[0] if autores else "",
        "titulo": clean_text(r.get("title") or ""),
        "revista": "",
        "volumen": "", "numero": "", "paginas": "",
        "nivel_revista": tier,
        "tipo": tipo,
        "fwci": None,
        "citas": 0,
        "doi": doi,
        "url": urls[0] if urls else "",
        "acceso_abierto": "repositorio",
        "afiliaciones_peru": ", ".join(r.get("institutions") or []),
        "fuente": "ALICIA (CONCYTEC)",
        "consulta": consulta,
        "prioridad": prioridad,
        "resumen": clean_text(" ".join(r.get("summary") or [])),
        "_pdfs": urls,
    }


# ==========================================================================
# Descarga de PDF (solo acceso abierto)
# ==========================================================================
META_PDF = re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', re.I)
META_PDF2 = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url', re.I)
META_REFRESH = re.compile(r"""http-equiv=["']refresh["'][^>]*url='?([^'">]+)""", re.I)
DRIVE_ID = re.compile(r"drive\.google\.com/.*?(?:id=|/d/)([\w-]{20,})")
LINK_PDF = re.compile(r'href=["\']([^"\']+(?:\.pdf|/bitstream/[^"\']+|/download/[^"\']+))["\']', re.I)


def fetch_pdf(web: Web, urls: list[str], dest: Path, max_mb: int = 60) -> str | None:
    """Prueba cada URL; si es una pagina, busca el enlace al PDF dentro.

    Devuelve la URL de la que salio el PDF, o None. Se comprueba la firma
    %PDF: muchas editoriales devuelven una pagina de acceso con codigo 200.
    """
    vistos: set[str] = set()
    cola = list(urls)
    while cola and len(vistos) < 8:
        u = cola.pop(0)
        # Google Drive muestra un aviso antes de los ficheros grandes; el
        # enlace directo lo salta. Lo usan los autores de ESL, ISLP y CASI.
        m = DRIVE_ID.search(u)
        if m:
            u = f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t"
        if u in vistos:
            continue
        vistos.add(u)
        r = web.get(u, timeout=(15, 90), stream=True, pause=0.3)
        if r is None or not r.ok:
            continue
        ctype = r.headers.get("content-type", "").lower()
        first = next(r.iter_content(8), b"") if "html" not in ctype else b""
        if first.startswith(b"%PDF"):
            data = bytearray(first)
            for chunk in r.iter_content(1 << 16):
                data.extend(chunk)
                if len(data) > max_mb * (1 << 20):
                    return None
            if len(data) < 20_000:
                continue
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(bytes(data))
            tmp.replace(dest)
            return u
        if "html" in ctype:
            text = r.text[:400_000]
            found = (META_REFRESH.findall(text) + META_PDF.findall(text) + META_PDF2.findall(text)
                     + LINK_PDF.findall(text))
            for f in found[:5]:
                cola.append(urljoin(r.url, html.unescape(f)))
    return None


# ==========================================================================
# Referencias (Crossref via doi.org)
# ==========================================================================
def fetch_reference(web: Web, doi: str, accept: str) -> str | None:
    def go():
        r = web.get(f"https://doi.org/{doi}", headers={"Accept": accept}, pause=0.2)
        if r is not None and r.ok and "html" not in r.headers.get("content-type", ""):
            return r.content.decode("utf-8", errors="replace").strip()
        return None
    return web.cached_text(f"{accept}|{doi}", go)


def bibtex_manual(rec: dict, key: str) -> str:
    kind = {"tesis doctoral": "phdthesis", "tesis de maestria": "mastersthesis"}.get(rec["tipo"], "misc")
    autores = " and ".join(a.strip() for a in rec["autores"].split(";") if a.strip())
    campos = {"title": rec["titulo"], "author": autores, "year": rec["anio"] or "",
              "school": rec["afiliaciones_peru"], "url": rec["url"],
              "note": rec["tipo"] + " (" + rec["fuente"] + ")"}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in campos.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}"


def apa_manual(rec: dict) -> str:
    autores = "; ".join(a.strip() for a in rec["autores"].split(";")[:6] if a.strip())
    base = f"{autores} ({rec['anio']}). {rec['titulo']}."
    if rec["tipo"].startswith("tesis"):
        base += f" [{rec['tipo'].capitalize()}, {rec['afiliaciones_peru']}]."
    return f"{base} {rec['url']}".strip()


# ==========================================================================
# Libros
# ==========================================================================
_OPENLIBRARY_OK = True


def verify_book(web: Web, b: dict) -> dict:
    global _OPENLIBRARY_OK
    out = {"isbn": "", "verificado_en": "", "citas_openalex": "", "version_libre": ""}
    apellido = re.split(r"[,\s]", b["autores"].strip())[0]
    fuentes: list[str] = []

    def main_title(t: str) -> str:
        return norm(re.split(r"[:(]", t or "")[0])

    def same_title(t: str) -> bool:
        # Titulo principal (antes de los dos puntos) identico. Con "empieza
        # igual" bastaba: "Deep Learning" (Goodfellow) se confundia con "Deep
        # Learning with Differential Privacy", del que Goodfellow es coautor.
        return len(main_title(t)) >= 5 and main_title(t) == main_title(b["titulo"])

    # 1. Crossref (registro de DOI): gratuito y sin presupuesto.
    doi = ""
    r = web.get("https://api.crossref.org/works",
                params={"query.bibliographic": f"{b['titulo']} {apellido}", "rows": 5,
                        "select": "DOI,title,author,type,ISBN"}, timeout=(10, 30))
    if r is not None and r.ok:
        for it in (r.json().get("message") or {}).get("items", []):
            apellidos = " ".join(a.get("family", "") for a in it.get("author") or [])
            if same_title(" ".join(it.get("title") or [])) and norm(apellido) in norm(apellidos):
                doi = it["DOI"]
                fuentes.append(f"Crossref (doi {doi})")
                out["isbn"] = ", ".join((it.get("ISBN") or [])[:2])
                break

    # 2. OpenAlex, para las citas: por DOI cuesta 1 credito; buscando, 10.
    sel = "display_name,cited_by_count,authorships,open_access,best_oa_location,type"
    if doi:
        w = web.openalex(f"works/doi:{doi}", {"select": sel}, scope="permanente")
        cand = [w] if w and w.get("display_name") else []
    else:
        j = web.openalex("works", {"search": b["titulo"], "per-page": 5, "select": sel,
                                   "filter": "type:book|book-chapter|preprint|article"},
                         scope="permanente")
        cand = (j or {}).get("results", [])
    for w in cand:
        nombres = " ".join(a["author"]["display_name"] for a in w.get("authorships") or [])
        if same_title(w["display_name"]) and (doi or norm(apellido) in norm(nombres)):
            out["citas_openalex"] = w.get("cited_by_count")
            fuentes.append("OpenAlex")
            oa = w.get("best_oa_location") or {}
            if (w.get("open_access") or {}).get("is_oa") and oa.get("pdf_url"):
                out["version_libre"] = oa["pdf_url"]
            break

    # 3. Open Library, por el ISBN. Responde a veces con mucho retraso; si
    # falla una vez no se insiste en esta ejecucion.
    if _OPENLIBRARY_OK and not out["isbn"]:
        r = web.get("https://openlibrary.org/search.json",
                    params={"title": b["titulo"], "author": apellido, "limit": 5,
                            "fields": "title,author_name,isbn"},
                    timeout=(8, 20), pause=0.5, intentos=1)
        if r is None:
            _OPENLIBRARY_OK = False
            web.log("    Open Library no responde: se omite en esta ejecucion")
        elif r.ok:
            docs = [d for d in r.json().get("docs") or [] if same_title(d.get("title", ""))]
            if docs:
                isbns = [i for d in docs for i in d.get("isbn") or [] if len(i) == 13]
                out["isbn"] = ", ".join(sorted(set(isbns))[:3])
                fuentes.append("Open Library")
    out["verificado_en"] = " + ".join(fuentes)
    return out


def collect_books(web: Web, cfg: dict, out_dir: Path, sin_pdf: bool, log) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filas = []
    for b in cfg["libros"]:
        info = verify_book(web, b)
        fila = {**{k: b.get(k, "") for k in ("titulo", "autores", "anio", "edicion", "editorial",
                                           "eje", "por_que", "acceso", "url")}, **info, "archivo": ""}
        if b["acceso"] == "libre_pdf" and not sin_pdf:
            dest = out_dir / f"{b['anio']}_{safe_name(b['autores'].split(',')[0].split(' y ')[0], 25)}_{safe_name(b['titulo'], 60)}.pdf"
            if dest.exists():
                fila["archivo"] = dest.name
            elif fetch_pdf(web, [b["url"]], dest, max_mb=120):
                fila["archivo"] = dest.name
            else:
                fila["archivo"] = "NO DESCARGADO (enlace roto o bloqueado)"
        estado = fila["archivo"] or {"libre_web": "lectura en linea", "comercial": "referencia"}.get(b["acceso"], "")
        log(f"  {b['anio']} {b['titulo'][:55]:55s} | {info['verificado_en'] or 'SIN VERIFICAR':22s} | {estado}")
        filas.append(fila)
    return filas


# ==========================================================================
# Antecedentes
# ==========================================================================
def collect_articles(web: Web, prot: dict, raw_dir: Path, log) -> tuple[list[dict], dict]:
    y0, y1 = prot["periodo"]["desde"], prot["periodo"]["hasta"]
    tipos = "|".join(prot["tipos"])
    base = f"publication_year:{y0}-{y1},type:{tipos},is_retracted:false,primary_location.source.type:journal"
    lv = prot["niveles"]
    local_ids = {f"https://openalex.org/{k}" for k in lv["local"]["openalex_instituciones"]}
    local_re = re.compile(lv["local"]["patron_afiliacion"], re.I)
    prisma = {"identificados": 0, "duplicados": 0, "fuera_de_criterio": 0}

    log("Resolviendo revistas de nivel A en OpenAlex...")
    tier_a = resolve_tier_a(web, prot["revistas_nivel_a"], log)
    log(f"  {len(tier_a)} de {len(prot['revistas_nivel_a'])} revistas identificadas")

    pool: dict[str, dict] = {}
    alias: dict[str, str] = {}

    def add(rec: dict):
        prisma["identificados"] += 1
        # Un mismo trabajo puede llegar con DOI por una fuente y sin el por
        # otra: se reconoce por cualquiera de las dos claves.
        keys = [k for k in (rec["doi"].lower(), "t:" + norm(rec["titulo"])) if k and k != "t:"]
        hit = next((alias[k] for k in keys if k in alias), None)
        if hit:
            prisma["duplicados"] += 1
            old = pool[hit]
            old["prioridad"] = min(old["prioridad"], rec["prioridad"])
            if rec["eje"] and rec["eje"] not in old["eje"].split(","):
                old["eje"] = ",".join(filter(None, [old["eje"], rec["eje"]]))
            if not old["doi"] and rec["doi"]:
                old["doi"] = rec["doi"]
            for k in keys:
                alias[k] = hit
            return
        pool[keys[0]] = rec
        for k in keys:
            alias[k] = keys[0]

    # --- Internacional, por eje ------------------------------------------
    for eje, spec in prot["ejes"].items():
        for q in spec["consultas"]:
            log(f"[{eje}] {spec['nombre']}")
            # Los 200 mas pertinentes en cualquier revista, mas todos los de las
            # revistas de nivel A (que la relevancia podria dejar fuera).
            works = openalex_works(web, f"{base},title_and_abstract.search:{q}", 200,
                                   sort="relevance_score:desc", raw_dir=raw_dir, etiqueta=f"{eje}_{q[:30]}")
            if tier_a:
                fuentes_a = "|".join(i.rsplit("/", 1)[-1] for i in tier_a)
                works += openalex_works(
                    web, f"{base},primary_location.source.id:{fuentes_a},title_and_abstract.search:{q}", 200,
                    sort="relevance_score:desc", raw_dir=raw_dir, etiqueta=f"{eje}_A_{q[:28]}")
            for w in works:
                tier = journal_tier(w, tier_a)
                nivel = affiliation_level(w, local_ids, local_re)
                if tier is None:
                    prisma["fuera_de_criterio"] += 1
                    continue
                add(record_from_openalex(w, eje, q, tier, nivel))

    # --- Nacional y local: autores con afiliacion peruana ------------------
    for k, q in enumerate(prot["consultas_nacionales"]):
        log(f"[Peru] {q[:70]}")
        works = openalex_works(web, f"{base},authorships.countries:{lv['nacional']['pais']},title_and_abstract.search:{q}",
                               200, sort="relevance_score:desc", raw_dir=raw_dir, etiqueta=f"PE_{q[:30]}", select=SELECT)
        for w in works:
            tier = journal_tier(w, tier_a)
            if tier is None:
                prisma["fuera_de_criterio"] += 1
                continue
            add(record_from_openalex(w, "", q, tier, affiliation_level(w, local_ids, local_re), k))

    # Produccion de instituciones de Puno en OpenAlex, con todas las consultas.
    inst = "|".join(lv["local"]["openalex_instituciones"])
    for k, q in enumerate(prot["consultas_nacionales"]):
        works = openalex_works(web, f"{base},authorships.institutions.lineage:{inst},title_and_abstract.search:{q}",
                               200, sort="relevance_score:desc", raw_dir=raw_dir, etiqueta=f"PUNO_{q[:30]}", select=SELECT)
        for w in works:
            tier = journal_tier(w, tier_a) or "C"
            add(record_from_openalex(w, "", q, tier, "local", k))

    # --- ALICIA: revistas y tesis peruanas ---------------------------------
    puno = lv["local"]["alicia_instituciones"]
    for k, q in enumerate(prot["consultas_alicia"]):
        log(f"[ALICIA] {q[:70]}")
        for r in alicia_search(web, q, [], 300, raw_dir, f"PE_{q[:30]}"):
            nivel = "local" if set(r.get("institutions") or []) & set(puno) else "nacional"
            rec = record_from_alicia(r, nivel, q, k)
            if rec is None or not rec["anio"] or not (y0 <= rec["anio"] <= y1):
                prisma["fuera_de_criterio"] += 1
                continue
            add(rec)
        for codigo in puno:
            for r in alicia_search(web, q, [f'institution:"{codigo}"'], 200, raw_dir, f"{codigo}_{q[:25]}"):
                rec = record_from_alicia(r, "local", q, k)
                if rec is None or not rec["anio"] or not (y0 <= rec["anio"] <= y1):
                    prisma["fuera_de_criterio"] += 1
                    continue
                add(rec)

    prisma["openalex"] = ("incompleto: se agoto el presupuesto diario" if web.openalex_blocked
                          else "completo")
    return list(pool.values()), prisma | {"revistas_nivel_a": tier_a}


def select(records: list[dict], prot: dict) -> list[dict]:
    lv = prot["niveles"]
    elegidos: list[dict] = []

    inter = [r for r in records if r["nivel"] == "internacional"
             and r["nivel_revista"] in lv["internacional"]["niveles_revista"]]
    usados: set[str] = set()
    for eje in prot["ejes"]:
        cand = sorted((r for r in inter if eje in r["eje"].split(",") and r["id"] not in usados), key=rank_key)
        for r in cand[: lv["internacional"]["cupo_por_eje"]]:
            usados.add(r["id"])
            elegidos.append({**r, "eje": eje})

    # Nacional y local: solo lo pertinente para un sistema de inversion.
    patron = re.compile(r"\b(" + "|".join(re.escape(norm(t)) for t in prot.get("terminos_pertinencia", [])) + ")")
    for r in records:
        texto = norm(f"{r['titulo']} {r['resumen']}")
        r["pertinencia"] = len(set(patron.findall(texto))) if patron.pattern != r"\b()" else 1

    nac = [r for r in records if r["nivel"] == "nacional" and r["pertinencia"] > 0]
    prio = lambda r: (ORDEN_REVISTA.get(r["nivel_revista"], 9), r["prioridad"], -r["pertinencia"],
                      -(r["fwci"] or 0), -r["citas"])
    art = sorted((r for r in nac if r["nivel_revista"] in "ABC"), key=prio)
    elegidos += art[: lv["nacional"]["cupo_articulos"]]
    tes = [r for r in nac if r["tipo"] in ("tesis doctoral", "tesis de maestria")]
    tes.sort(key=lambda r: (r["tipo"] != "tesis doctoral", r["prioridad"], -r["pertinencia"], -(r["anio"] or 0)))
    elegidos += tes[: lv["nacional"]["cupo_tesis"]]

    loc = [r for r in records if r["nivel"] == "local" and r["pertinencia"] > 0
           and (lv["local"]["incluir_tesis"] or r["nivel_revista"] != "T")]
    loc.sort(key=lambda r: (r["nivel_revista"] == "T", r["prioridad"], -r["pertinencia"], rank_key(r),
                            -(r["anio"] or 0)))
    elegidos += loc[: lv["local"]["cupo"]]
    return elegidos


# ==========================================================================
# Salidas
# ==========================================================================
COLUMNAS = ["nivel", "eje", "rango", "anio", "autores", "titulo", "revista", "volumen", "numero",
            "paginas", "tipo", "nivel_revista", "fwci", "citas", "doi", "url", "acceso_abierto",
            "pdf", "afiliaciones_peru", "pertinencia", "fuente", "consulta", "cita_apa", "resumen"]


def write_outputs(sel: list[dict], libros: list[dict], prisma: dict, prot: dict,
                  run_info: dict, web: Web, sin_pdf: bool, log):
    ant = BASE / "antecedentes"
    for n in NIVELES:
        (ant / n).mkdir(parents=True, exist_ok=True)

    # PDFs, referencias y rango dentro de cada grupo.
    bib, contador = [], {}
    grupos: dict[tuple, int] = {}
    for rec in sel:
        g = (rec["nivel"], rec["eje"])
        grupos[g] = grupos.get(g, 0) + 1
        rec["rango"] = grupos[g]
        apellido = safe_name(re.split(r"[,;]", rec["primer_autor"] or "anon")[0].split()[-1] if rec["primer_autor"] else "anon", 20)
        stem = f"{rec['anio']}_{apellido}_{safe_name(rec['titulo'], 60)}"
        key = f"{apellido}{rec['anio']}"
        contador[key] = contador.get(key, 0) + 1
        if contador[key] > 1:
            key += "abcdefghijklmnopqrstuvwxyz"[contador[key] - 2]
        rec["pdf"] = ""
        dest = ant / rec["nivel"] / f"{stem}.pdf"
        apartado = ant / "_fuera_de_seleccion" / rec["nivel"] / f"{stem}.pdf"
        if not dest.exists() and apartado.exists():
            apartado.replace(dest)                 # vuelve a estar seleccionado
        if dest.exists():
            rec["pdf"] = str(dest.relative_to(BASE)).replace("\\", "/")
        elif not sin_pdf and rec["_pdfs"]:
            if fetch_pdf(web, rec["_pdfs"], dest):
                rec["pdf"] = str(dest.relative_to(BASE)).replace("\\", "/")
        if rec["doi"]:
            bt = fetch_reference(web, rec["doi"], "application/x-bibtex")
            apa = fetch_reference(web, rec["doi"], "text/x-bibliography; style=apa")
        else:
            bt, apa = None, None
        if bt and bt.startswith("@"):
            bt = re.sub(r"^(@\w+\{)[^,]+,", lambda m: m.group(1) + key + ",", bt, count=1)
        bib.append(bt if bt and bt.startswith("@") else bibtex_manual(rec, key))
        rec["cita_apa"] = clean_text(apa) if apa else apa_manual(rec)
        rec["bibtex_key"] = key
        log(f"  {rec['nivel'][:5]} {rec['eje'] or '-':3s} {rec['anio']} {rec['nivel_revista']} "
            f"{'PDF' if rec['pdf'] else '   '} {rec['titulo'][:70]}")

    with open(ant / "matriz_antecedentes.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, extrasaction="ignore")
        w.writeheader()
        for rec in sel:
            w.writerow(rec)
    (BASE / "referencias.bib").write_text("\n\n".join(bib) + "\n", encoding="utf-8")

    # PDFs de ejecuciones anteriores que ya no estan en la seleccion: se
    # apartan, no se borran (alguien puede estar leyendolos).
    vigentes = {rec["pdf"] for rec in sel if rec["pdf"]}
    for n in NIVELES:
        for f in (ant / n).glob("*.pdf"):
            if str(f.relative_to(BASE)).replace("\\", "/") not in vigentes:
                destino = ant / "_fuera_de_seleccion" / n / f.name
                destino.parent.mkdir(parents=True, exist_ok=True)
                f.replace(destino)

    # Version legible.
    L = ["# Antecedentes", "",
         f"Generado el {run_info['fecha']} con `recolectar.py` (protocolo `{run_info['protocolo_hash']}`).",
         "Cada referencia sale de una base de datos (OpenAlex, Crossref o ALICIA); ninguna se escribio de memoria.",
         "La matriz completa, con resumenes, esta en `matriz_antecedentes.csv` (se abre con Excel).", ""]
    if prisma.get("openalex") != "completo":
        L += ["> **Seleccion incompleta:** OpenAlex agoto su presupuesto diario gratuito durante esta "
              "ejecucion. Los niveles nacional y local desde ALICIA estan completos; lo que depende de "
              "OpenAlex se completa en la siguiente ejecucion (la tarea programada la repite).", ""]
    L += [
         "Nivel de revista: **A** referencia del area (FT50 y equivalentes) · **B** nucleo cientifico "
         "internacional · **C** otra revista indexada · **T** tesis (complemento).", ""]
    for n in NIVELES:
        items = [r for r in sel if r["nivel"] == n]
        L += [f"## {n.capitalize()} ({len(items)})", ""]
        if not items:
            L += ["_Sin resultados que cumplan el protocolo._", ""]
            continue
        ejes = [e for e in prot["ejes"]] + [""]
        for e in ejes:
            sub = [r for r in items if r["eje"] == e]
            if not sub:
                continue
            if e:
                L += [f"### {e}. {prot['ejes'][e]['nombre']}", f"_Modulo del prototipo: {prot['ejes'][e]['modulo']}_", ""]
            for r in sub:
                pdf = f" · [PDF]({r['pdf'].split('/', 1)[1]})" if r["pdf"] else ""
                L.append(f"{r['rango']}. **[{r['nivel_revista']}]** {r['cita_apa']}{pdf}")
            L.append("")
    (ant / "README.md").write_text("\n".join(L), encoding="utf-8")

    # Libros.
    lb = BASE / "libros"
    with open(lb / "indice_libros.csv", "w", encoding="utf-8-sig", newline="") as fh:
        cols = ["titulo", "autores", "anio", "edicion", "editorial", "eje", "acceso", "archivo", "url",
                "isbn", "verificado_en", "citas_openalex", "version_libre", "por_que"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for b in libros:
            w.writerow(b)
    L = ["# Libros de referencia", "",
         "Los marcados *en esta carpeta* son copias gratuitas y legales publicadas por sus autores o editoriales.",
         "Los de pago se listan para pedirlos en la biblioteca; no se descargan copias no autorizadas.", ""]
    for acc, titulo in (("libre_pdf", "En esta carpeta"), ("libre_web", "Gratis para leer en linea"),
                        ("comercial", "De pago (pedir en biblioteca)")):
        L += [f"## {titulo}", ""]
        for b in sorted((x for x in libros if x["acceso"] == acc), key=lambda x: (x["eje"], x["anio"])):
            extra = []
            if b.get("edicion"):
                extra.append(f"{b['edicion']} ed.")
            if b.get("isbn"):
                extra.append(f"ISBN {b['isbn'].split(',')[0]}")
            if b.get("citas_openalex") not in ("", None):
                extra.append(f"{b['citas_openalex']} citas en OpenAlex")
            destino = b["archivo"] if acc == "libre_pdf" else b.get("url") or ""
            L.append(f"- **{b['autores']} ({b['anio']}).** *{b['titulo']}*. {b['editorial']}."
                     f" {' · '.join(extra)}  \n  {b['por_que']} `{b['eje']}` {destino}")
        L.append("")
    (lb / "README.md").write_text("\n".join(L), encoding="utf-8")

    # Registro de la ejecucion (PRISMA).
    run_info["prisma"] = {k: v for k, v in prisma.items() if k != "revistas_nivel_a"} | {
        "incluidos": len(sel),
        "por_nivel": {n: sum(r["nivel"] == n for r in sel) for n in NIVELES},
        "con_pdf": sum(bool(r["pdf"]) for r in sel),
    }
    run_info["revistas_nivel_a_identificadas"] = prisma.get("revistas_nivel_a", {})
    (BASE / "busquedas" / run_info["fecha"] / "ejecucion.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_info


# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", choices=["libros", "antecedentes"])
    ap.add_argument("--sin-pdf", action="store_true", help="no descargar PDFs")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = BASE / "busquedas" / fecha
    raw.mkdir(parents=True, exist_ok=True)
    logf = open(raw / "registro.log", "a", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    prot_text = (BASE / "protocolo.yaml").read_text(encoding="utf-8")
    prot = yaml.safe_load(prot_text)
    libros_cfg = yaml.safe_load((BASE / "libros.yaml").read_text(encoding="utf-8"))
    web = Web(BASE / "busquedas" / "cache", log)
    run_info = {"fecha": fecha, "inicio_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocolo_hash": hashlib.sha256(prot_text.encode()).hexdigest()[:12]}
    log(f"=== Recoleccion {run_info['inicio_utc']} · protocolo {run_info['protocolo_hash']} ===")

    try:
        return run(args, prot, prot_text, libros_cfg, web, run_info, raw, log)
    finally:
        logf.close()


def run(args, prot, prot_text, libros_cfg, web, run_info, raw, log) -> int:
    libros = []
    if args.solo != "antecedentes":
        log("\nLibros")
        libros = collect_books(web, libros_cfg, BASE / "libros", args.sin_pdf, log)
    if args.solo == "libros":
        return 0

    log("\nAntecedentes")
    records, prisma = collect_articles(web, prot, raw, log)
    sel = select(records, prot)
    log(f"\nSeleccionados {len(sel)} de {len(records)} registros unicos.")
    enrich(web, sel, log)
    log("Descargando y citando...")
    if not libros:
        libros = collect_books(web, libros_cfg, BASE / "libros", True, lambda *_: None)
    info = write_outputs(sel, libros, prisma, prot, run_info, web, args.sin_pdf, log)
    p = info["prisma"]
    log(f"\nPRISMA: identificados {p['identificados']} · duplicados {p['duplicados']} · "
        f"fuera de criterio {p['fuera_de_criterio']} · incluidos {p['incluidos']} "
        f"({p['por_nivel']}) · con PDF {p['con_pdf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
