# Laboratorio · material de las sesiones 2 a 4

Copias de consulta del laboratorio: los nueve pasos del temario en Python (`python/laboratorio.ipynb`)
y en R (`r/laboratorio.Rmd`, con `funciones.R` y `riesgo.Rmd`).

**Para ejecutarlos, usa los originales de `laboratorio/` en la raíz del repositorio.** Dependen de
la estructura del proyecto: el cuaderno importa `cryptoquant` desde dos carpetas más arriba y el
R Markdown lee y escribe en `../../reports/`. Desde esta carpeta esas rutas no existen.

| Paso | Tema | Python | R |
|---|---|---|---|
| 1 | Velas de Binance, DataFrame, gráfico y CSV | sí | `descargar_binance()` |
| 2 | Procesamiento con dplyr y ggplot2 | — | § 2 |
| 3 | Indicadores como variables, rezagos y objetivo | sí | § 3 |
| 4 | Colas gruesas y estadística de señales | sí | § 4 |
| 5 | Series de tiempo y ARIMA | sí | § 5 |
| 6 | Volatilidad: GARCH(1,1), stop y tamaño | sí | § 6 |
| 7 | Random forest con prueba en el futuro | sí | § 7 |
| 8 | Monte Carlo del precio y de la cuenta | sí | § 8 |
| 9 | Control de volatilidad | sí | § 9 |

Cómo se ejecuta (desde la raíz del repositorio):

```bash
python -m cryptoquant laboratorio          # los nueve pasos; tablas y gráficos en reports/laboratorio/
```

En R: abrir `laboratorio/r/laboratorio.Rproj` en RStudio y pulsar *Knit* sobre `laboratorio.Rmd`.
Más detalle en `laboratorio/README.md` y en el manual (`07_manual_de_usuario.md`, sección B.3).
