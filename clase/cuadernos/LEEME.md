# Cuadernos por tema

Un cuaderno por unidad del temario del módulo 8 (*Análisis cuantitativo y modelamiento de datos en
criptoactivos con Python y RStudio*), en **Python** (`python/`, para Google Colab o Jupyter) y en **R**
(`r/`, para RStudio o Posit Cloud). Cada uno explica el tema con datos reales de BTC, ETH y SOL y termina
con una sección **«Así lo usa el piloto»** que conecta lo aprendido con la app de riesgo. Son para
estudiar: traen el código completo, sus resultados y ejercicios.

| # | Tema | Python | R | Qué parte del piloto explica |
|---|---|---|---|---|
| 1 | Entorno de trabajo y datos | `01_datos.ipynb` | `01_datos.Rmd` | la descarga de velas de Binance y su caché |
| 2 | Procesamiento y transformación | `02_procesamiento.ipynb` | `02_procesamiento.Rmd` | **Mi diario**: la cartera valorada día a día |
| 3 | Preparación de variables | `03_variables.ipynb` | `03_variables.Rmd` | las dos variables de la regla: volatilidad EWMA y caída de 180 días |
| 4 | Regresión lineal | `04_regresion.ipynb` | `04_regresion.Rmd` | **Hoy**: qué parte del riesgo aporta cada moneda (beta) |
| 5 | Series de tiempo (ARIMA, SARIMA) | `05_series_de_tiempo.ipynb` | `05_series_de_tiempo.Rmd` | por qué el piloto no pronostica el precio |
| 6 | Volatilidad (ARCH, GARCH, EWMA) | `06_volatilidad.ipynb` | `06_volatilidad.Rmd` | **Hoy**: la volatilidad prevista y la parte en cripto |
| 7 | Machine learning | `07_machine_learning.ipynb` | `07_machine_learning.Rmd` | por qué no usa ML para la dirección, y sí para la volatilidad |
| 8 | Monte Carlo | `08_monte_carlo.ipynb` | `08_monte_carlo.Rmd` | **Tiempos duros**: un mal día y el próximo mes |
| 9 | El piloto por dentro | `09_el_piloto_por_dentro.ipynb` | `09_el_piloto_por_dentro.Rmd` | todas las pantallas, reconstruidas y contrastadas con la app |

La versión de R de cada cuaderno ya renderizada está en `r/*.html`: se puede leer en el navegador sin
instalar nada.

## Mismos datos, mismas cifras

Todos usan los cierres diarios de Binance del **01-01-2020 al 26-09-2026**, el último día cerrado antes de
la primera clase. Fijar el periodo hace que todos veamos las mismas cifras, que el texto pueda citarlas y
que Python y R se puedan comparar: dan los mismos resultados salvo donde el método usa azar o un
optimizador propio (random forest, Monte Carlo, GARCH), y cada cuaderno lo dice donde pasa.

- Los cuadernos descargan solos de `data-api.binance.vision` la primera vez y guardan `BTCUSDT_1d.csv`,
  `ETHUSDT_1d.csv` y `SOLUSDT_1d.csv` junto al cuaderno.
- Si Binance no responde, basta subir los CSV de la carpeta de datos de respaldo de la clase
  (`sesion_1_2026-09-27/04_datos_respaldo/`): mismos nombres y mismo formato.
- Para trabajar con datos hasta ayer: `HASTA = None` en la celda de preparación (Python) o
  `HASTA <- NA` en `r/comun.R`. Las cifras dejarán de coincidir con el texto.

La Unidad 9 reproduce las cifras de la app para la cartera de ejemplo del manual (0,05 BTC, 1,2 ETH,
10 SOL y 500 USDT) con los cierres del 26-09-2026. Ejecutado dentro del proyecto, el cuaderno de Python
las compara una por una con el motor del piloto (`cryptoquant/piloto/motor.py`); el de R, con esas mismas
cifras.

## Cómo se abren

**Python en Colab.** *Archivo → Subir cuaderno* y elegir el `.ipynb`. La primera celda de código
(*Preparación*) aparece plegada: se ejecuta con ▶ y carga los datos. La Unidad 6 instala `arch` sola.

**Python en el equipo.** Con el entorno del proyecto (`%USERPROFILE%\.venvs\cryptoquant`) y Jupyter o
VS Code, abriendo el cuaderno desde la carpeta `python/`.

**R en RStudio o Posit Cloud.** Abrir `r/cuadernos.Rproj` (en Posit Cloud, subir la carpeta `r/` entera a
un proyecto) y después el `.Rmd`. Cada cuaderno, salvo el primero, empieza con `source("comun.R")`.
Paquetes, una vez:

```r
install.packages(c("dplyr", "tidyr", "readr", "ggplot2", "lubridate", "jsonlite", "zoo",
                   "lmtest", "forecast", "tseries", "rugarch", "rpart", "ranger", "rmarkdown"))
```

## Cómo se mantienen

Los `.ipynb` **no se editan a mano**: se generan desde `fuentes/*.py` (formato *percent*: `# %%` abre una
celda de código, `# %% [markdown]` una de texto). La celda de preparación común está en
`fuentes/_preparacion.py`. Los `.Rmd` se editan directamente.

```bash
python clase/cuadernos/generar.py --ejecutar          # regenera y ejecuta los 9 .ipynb
python clase/cuadernos/generar.py --ejecutar 06       # solo la Unidad 6
python clase/cuadernos/generar.py --r                 # renderiza los 9 .Rmd a HTML
```

Ejecutarlos necesita `nbclient` e `ipykernel` en el mismo Python. Renderizar los de R necesita `Rscript` y
el pandoc de RStudio (el script lo busca en su carpeta de instalación si no está la variable
`RSTUDIO_PANDOC`).

`tests/test_cuadernos.py` comprueba que cada tema existe en los dos lenguajes, que los `.ipynb` están al
día con sus fuentes y publicados ejecutados sin errores, que la preparación es la misma en todos y que
Python y R usan el mismo periodo. Los CSV no van a git (`.gitignore` excluye `clase/**/*.csv`).
