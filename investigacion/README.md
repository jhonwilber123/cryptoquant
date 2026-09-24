# Investigación: libros y antecedentes

Base documental para seguir desarrollando el prototipo con fundamento en la
literatura, y no en lo que un modelo de lenguaje recuerde o elija cada vez.

## Qué hay aquí

| Carpeta o archivo | Contenido |
|---|---|
| `libros/` | Libros de referencia del área. Los gratuitos y legales están descargados; los de pago, listados para pedirlos en biblioteca. Índice en `libros/README.md` e `indice_libros.csv`. |
| `antecedentes/internacional/` | PDFs de artículos sin autores con afiliación peruana. |
| `antecedentes/nacional/` | PDFs de artículos y tesis con autores de Perú (fuera de Puno). |
| `antecedentes/local/` | PDFs de artículos y tesis con autores de Puno. |
| `antecedentes/README.md` | Todos los antecedentes, por nivel y eje, con su cita en APA. |
| `antecedentes/matriz_antecedentes.csv` | La matriz completa, con el resumen de cada trabajo. Se abre con Excel. |
| `referencias.bib` | Todas las referencias en BibTeX, para importarlas en Zotero, Mendeley o LaTeX. |
| `protocolo.yaml` | Las consultas, filtros y cupos exactos que se aplicaron. |
| `libros.yaml` | La lista curada de libros. |
| `recolectar.py` | El script que lo hace todo. |
| `busquedas/AAAA-MM-DD/` | Las respuestas crudas de cada base de datos y el registro de la ejecución, con los recuentos al estilo PRISMA. |

## Cómo se eligió cada documento

**Fuentes.** Las referencias salen de bases de datos, no de la memoria de nadie:

- **OpenAlex:** índice académico abierto con más de 250 millones de trabajos.
- **ALICIA (CONCYTEC):** repositorio nacional de revistas y tesis peruanas.
- **Crossref:** verifica cada DOI y genera las citas APA y BibTeX.
- **Open Library:** verifica los libros.

**Periodo de los antecedentes:** de 2021 a 2026, es decir, no más de 5 años. Los libros no
tienen límite de fecha: se eligen por ser los más representativos del área, y el índice muestra
cuántas veces los cita la literatura según OpenAlex.

**Niveles.** Se asignan según la afiliación de los autores que declara cada artículo:

- **Local:** al menos un autor de una institución de Puno (UNA Puno, UNAJ, UANCV o UPSC).
- **Nacional:** al menos un autor de una institución peruana fuera de Puno.
- **Internacional:** ningún autor con afiliación en Perú.

**Nivel de la revista.** Cada antecedente se etiqueta con su nivel:

- **A:** revistas de referencia del área, del estilo del FT50 (Journal of Finance, Journal of
  Econometrics, International Journal of Forecasting…). La lista completa está en
  `protocolo.yaml`.
- **B:** revistas del núcleo científico internacional según OpenAlex (derivado del CWTS Leiden
  Ranking; se aproxima a Web of Science o Scopus).
- **C:** otras revistas académicas indexadas.
- **T:** tesis. Se incluyen solo en los niveles nacional y local, y como complemento.

**Criterios de selección por nivel:**

- **Internacional:** solo revistas A o B, 10 por eje, ordenadas primero por nivel de revista y
  luego por impacto normalizado (FWCI, que compara con trabajos del mismo campo y año).
- **Nacional:** los 30 mejores artículos, priorizando cripto y finanzas sobre temas afines, más
  hasta 10 tesis de maestría o doctorado.
- **Local:** hasta 30 trabajos. La producción de Puno en revistas es escasa, así que aquí sí
  entran tesis.

**Ejes.** Cada eje responde a un módulo del prototipo:

| Eje | Tema | Módulo |
|---|---|---|
| E1 | Volatilidad y riesgo de los criptoactivos | `econometrics/` |
| E2 | Previsión de volatilidad y su evaluación | `econometrics/volatility.py` |
| E3 | Construcción de carteras y control de riesgo | `portfolio/` |
| E4 | Predicción de rendimientos con aprendizaje automático | `models/` |
| E5 | Factores, valoración y eficiencia del mercado cripto | `backtest/engine.py` |
| E6 | Rigor estadístico: sobreajuste, pruebas múltiples, replicación | `models/validation.py`, `forward/` |
| E7 | Blockchain, DeFi, stablecoins, regulación y adopción | contexto |

## Qué no se hace

- **No se descargan copias no autorizadas.** Solo se descargan PDFs de acceso abierto o publicados
  gratis por sus autores o editoriales. De los demás se guarda la referencia verificada y el
  enlace: se consiguen por la biblioteca de la universidad, escribiendo al autor o buscando una
  versión preliminar (SSRN, NBER, arXiv).
- **No se usan los cuartiles de SCImago (SJR).** Su web bloquea las descargas automáticas, así que
  el nivel de revista se basa en los criterios A/B/C de arriba. Si tu universidad exige cuartiles,
  compruébalos a mano en scimagojr.com para los artículos que cites.

## Repetir o ampliar la búsqueda

Desde la carpeta del proyecto:

```
C:\Users\ADMIN\.venvs\cryptoquant\Scripts\python.exe investigacion\recolectar.py
```

- **Por pasos:** `--sin-pdf` actualiza solo los metadatos, que es rápido; sin esa opción descarga
  también los PDFs. Los PDFs que ya están no se vuelven a descargar.
- **Cada ejecución queda fechada** en `busquedas/`, con la huella del protocolo. Si cambias
  `protocolo.yaml`, la huella cambia, y así queda constancia de qué criterios produjeron cada
  selección.
- **Funciona igual en el PC** (`C:\Users\ADMIN\Proyectos\Blockchain y criptoactivos`), donde queda
  una copia para seguir recolectando. `scripts\sincronizar.ps1 -Codigo` lleva allí esta carpeta.

## Recolección automática en el PC

La tarea programada `CryptoQuant-Literatura` ejecuta `recolectar.py` en el PC cada lunes a las
09:00. Si el equipo está apagado a esa hora, se ejecuta al encenderlo. Lo nuevo llega a esta
carpeta del USB con la sincronización que ya hace la tarea diaria.

- Ver su estado: `scripts\literatura.ps1 -Status`
- Ejecutarla a mano: `scripts\literatura.ps1`

**Presupuesto de OpenAlex.** Sin clave, OpenAlex reparte entre todos los usuarios de una misma
conexión (IP) un presupuesto diario gratuito, que se reinicia a medianoche UTC (las 19:00 en
Perú). Cada búsqueda gasta una parte, y una recolección completa consume casi todo el
presupuesto del día.

- El script **guarda en caché** lo ya consultado: repetirlo el mismo día no gasta nada.
- Si el presupuesto se agota, **avisa** y deja la selección marcada como incompleta hasta la
  siguiente ejecución.
- Para tener un presupuesto propio, pide una **clave gratuita** en
  <https://help.openalex.org/api/authentication/> y guárdala en `investigacion\clave_openalex.txt`.
  El script la usará sola.
