# Laboratorio: de las velas al modelo

Nueve pasos, en Python y en R, sobre las velas horarias de BTC/USDT en Binance
desde 2022: los ocho del temario y el control de volatilidad de la tesis. El resto de `cryptoquant` es un sistema en producción; esto es el
recorrido a la vista de cómo se construye uno, para enseñarlo y discutirlo.

| Tema | Qué se hace | Python | R |
|---|---|---|---|
| Entorno | DataFrame de pandas, velas de Binance, gráfico de velas, exportación a CSV | `datos.py` | `descargar_binance()` |
| Procesamiento | Leer el CSV, filtrar y agrupar con dplyr, rendimientos simples y log, crecimiento del capital, ggplot2 | — | § 2 |
| Variables | Medias, RSI, MACD, Bollinger, volatilidad móvil, rezagos y el objetivo (sube o baja) | `variables.py` | § 3 |
| Cuantitativo | Colas gruesas frente a la normal; acierto y retorno medio tras cada señal | `cuantitativo.py` | § 4 |
| Series de tiempo | Retornos y no precios, autocorrelación, patrones por hora y día, ARIMA | `series.py` | § 5 |
| Volatilidad | Agrupamiento, GARCH(1,1), pronóstico aplicado al stop y al tamaño | `volatilidad.py` | § 6 |
| Machine learning | Random forest con prueba en el futuro, línea base e importancia de variables | `aprendizaje.py` | § 7 |
| Monte Carlo | Trayectorias de precio; simulación de la cuenta con acierto, R:B y riesgo | `montecarlo.py` | § 8 |
| Control de volatilidad | Invertir min(1, objetivo / volatilidad prevista) y compararlo con comprar y mantener a igual volatilidad | `control.py` | § 9 |

Los módulos de Python están en [`cryptoquant/laboratorio/`](../cryptoquant/laboratorio/).

## Cómo se usa

### Python

```bash
python -m cryptoquant laboratorio                 # todo, con resumen en consola
python -m cryptoquant laboratorio --descargar     # forzar velas nuevas
python -m cryptoquant laboratorio --acierto 0.40 --rb 3 --riesgo 0.02
```

La primera vez descarga ~41 000 velas (menos de un minuto) y las guarda en
`reports/laboratorio/velas_BTC-USDT_1h.csv`. Después las reutiliza. Deja allí
también las tablas (`*.csv`) y los gráficos (`*.png`).

El notebook [`python/laboratorio.ipynb`](python/laboratorio.ipynb) recorre lo
mismo celda a celda. Ábralo desde su carpeta (`jupyter lab` o VS Code).

### R

Abra [`r/laboratorio.Rproj`](r/laboratorio.Rproj) en RStudio y luego
`laboratorio.Rmd` → *Knit*. Paquetes:

```r
install.packages(c("dplyr", "tidyr", "readr", "ggplot2", "lubridate", "zoo",
                   "forecast", "rugarch", "ranger", "tseries", "jsonlite", "rmarkdown"))
```

Lee el CSV que exporta Python. Si no existe, lo descarga él mismo de Binance.

`riesgo.Rmd` es el contraste en R del módulo de riesgo: tras
`python -m cryptoquant riesgo`, recalcula desde cero las pruebas de Kupiec y
Christoffersen sobre los VaR pronosticados y comprueba que coinciden.
Sin RStudio: `Rscript -e "rmarkdown::render('laboratorio.Rmd')"` desde `r/`.

## Las dos versiones dan las mismas cifras

Los indicadores de R replican las fórmulas de pandas, incluidas las medias
exponenciales ajustadas del RSI. La sección 3 del R Markdown compara las 21
variables con las de Python: la diferencia máxima es de 10⁻⁸ sobre precios
de 10⁵, es decir, redondeo. Las señales coinciden al decimal. El GARCH se
estima con optimizadores distintos (`arch` frente a `rugarch`) y los
parámetros difieren en la tercera cifra.

## Lo que sale, y por qué importa

Con BTC de 2022 a 2026:

- **Colas.** Días de más de 4 desviaciones: 10 observados, 0,1 esperados si
  los retornos fueran normales. Un stop "a 3 sigmas" salta seis veces más de
  lo que promete la normal.
- **Señales.** Ninguna de las 24 combinaciones de señal y horizonte supera a
  su base tras corregir por comparaciones múltiples.
- **Memoria.** Los retornos casi no se autocorrelacionan (Ljung-Box p = 0,17),
  sus cuadrados sí (p ≈ 10⁻¹⁸). El ARIMA elegido es (0,0,0): ruido blanco.
  La volatilidad depende mucho de la hora; el retorno, no.
- **GARCH.** Persistencia 0,99: un sobresalto tarda unos 70 días en
  disiparse a la mitad. Es lo que se puede prever, y por eso decide el stop
  y el tamaño.
- **Random forest.** Acierto del 49,6 % en la prueba, por debajo de decir
  siempre "baja" (50,6 %) y de repetir la dirección de hoy (51,6 %). La
  importancia de variables no significa que alguna sirva: la de permutación
  ronda cero.
- **Monte Carlo.** Con acierto del 45 % y R:B 1:2 (esperanza +0,35 R), arriesgar
  el 1 % por operación da una caída mediana del 9 % en 200 operaciones. Al 10 %,
  la mediana final es mayor, pero el 94 % de las cuentas cae más del 50 % por
  el camino; casi nadie aguanta eso sin abandonar.

Es la misma conclusión que el sistema principal: la dirección apenas se puede
predecir; el riesgo sí se puede medir y acotar. Esa medición, validada, está en
`python -m cryptoquant riesgo` y `cartera`; las ideas de ganancia pasan por
`evidencia`. Ver el [README principal](../README.md#riesgo--cryptoquantriesgo).
