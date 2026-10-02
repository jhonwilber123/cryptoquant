# Arquitectura del programa `cryptoquant`

`cryptoquant` es un sistema cuantitativo de inversión en criptoactivos escrito en Python, con una
réplica en R de sus partes didácticas. Su objetivo de diseño no es maximizar el retorno: es que
**el riesgo sea una variable que fija el usuario**, no una que impone el mercado.

Es la misma idea que recorre la sesión 1: la dirección del precio casi no se puede prever, su
volatilidad sí. El programa se organiza alrededor de esa asimetría.

## Vista general

```text
                    Binance · API pública, sin claves
               (ccxt, redirigido a data-api.binance.vision)
                                  │
                                  ▼
                       data/sources.py ─── caché parquet (data/cache/)
                                  │        respaldo sintético si no hay red
       ┌──────────────────────────┼──────────────────────────────┐
       ▼                          ▼                              ▼
 SISTEMA PRINCIPAL          MEDICIÓN DEL RIESGO             DIDÁCTICA
 features/  variables       riesgo/  VaR y CVaR por 4       laboratorio/  9 pasos
 econometrics/ GARCH, VaR            métodos, pruebas de     clase/  cuadernos Colab
 models/  P(operación                Kupiec y Christoffersen,        con verificadores,
          rentable)                  Monte Carlo, filtro de          script de R
 portfolio/ HRP + control            evidencia               laboratorio/r/  réplica en R
          de riesgo                         │
 backtest/  walk-forward                    │
 forward/   diario sellado                  │
       │                                    │
       ├──────────────┬─────────────────────┤
       ▼              ▼                     ▼
 dashboard/      piloto/  app Streamlit   tesis.py  contrastes de las
 panel HTML      para la cartera real     hipótesis de la tesis
       │              │                     │
       ▼              ▼                     ▼
 reports/dashboard.html   data/piloto/     reports/tesis/

 cli.py: un solo punto de entrada · python -m cryptoquant <comando>
 config.py + config.yaml: todos los parámetros de riesgo, a la vista
```

## El sistema principal, capa por capa

```text
Datos → Variables causales → Econometría → Machine learning → Cartera → Control de riesgo → Pesos
                                                                                  │
                                                         backtest (pasado) · forward (en vivo)
```

| Capa | Módulo | Qué hace |
|---|---|---|
| Datos | `data/sources.py` | Descarga velas OHLCV vía `ccxt`, las guarda en parquet y, sin red, usa series sintéticas reproducibles. Siempre informa del origen: si dice `synthetic`, las cifras no hablan del mercado |
| Variables | `features/technical.py` | Momentum ajustado por volatilidad, RSI, MACD, ATR, asimetría, curtosis, Hurst y régimen. Todas estrictamente causales: solo usan el pasado |
| Econometría | `econometrics/volatility.py` | GJR-GARCH(1,1,1) con innovaciones t asimétrica; EWMA si no converge |
| | `econometrics/risk.py` | VaR histórico, paramétrico y de Cornish-Fisher, y CVaR |
| | `econometrics/diagnostics.py` | ADF, KPSS, Ljung-Box, ARCH-LM, Jarque-Bera, Hurst y cointegración |
| Machine learning | `models/labeling.py` | Etiquetado triple barrera: objetivo, stop o vencimiento, con barreras que escalan con la volatilidad |
| | `models/predictor.py` | Gradient boosting calibrado (isotónica): da P(operación rentable), no un precio |
| | `models/validation.py` | Validación temporal purgada con embargo, en lugar de la validación aleatoria que mezcla pasado y futuro |
| Cartera | `portfolio/optimizer.py` | HRP (Hierarchical Risk Parity) sobre covarianza con contracción de Ledoit-Wolf |
| Control de riesgo | `portfolio/sizing.py` | exposición = (vol objetivo ÷ vol prevista) × convicción × drawdown × régimen, con topes por activo |
| Backtest | `backtest/engine.py`, `metrics.py` | Walk-forward: decide en t con datos hasta t, aplica el retorno de t+1 y cobra costes sobre la rotación. Reporta el Sharpe deflactado |
| Forward test | `forward/journal.py`, `evaluate.py` | Diario de decisiones en vivo, pre-registrado y encadenado por hashes SHA-256 |
| Informes | `reporting/report.py`, `dashboard/` | Tablas en consola, gráficos y un panel HTML autocontenido |

### Una sola escala de riesgo

La fórmula del control de riesgo tiene **una sola escala** (volatilidad objetivo ÷ volatilidad
prevista); los demás factores están entre 0 y 1 y solo pueden recortar. Es deliberado: si dos
capas recortaran por el mismo riesgo, la exposición se hundiría y `target_volatility` dejaría de
significar nada.

### Backtest y vida real comparten el mismo método de decisión

`backtest/engine.py` tiene un único método de decisión (`_decide`) que usan tanto el backtest como
el registro diario en vivo. Un test comprueba que el modo diario rebalancea los mismos días y con
los mismos pesos que el backtest.

### El diario sellado

`forward/journal.py` guarda cada decisión en `data/forward/journal.jsonl`, un registro de solo
añadido donde cada línea lleva el hash SHA-256 de la anterior: es la misma cadena de hashes que
los alumnos construyen en el Bloque 1. Editar una decisión pasada rompe su hash; recalcularlo rompe
el enlace de la siguiente. Las hipótesis se pre-registran antes de tener datos, y las correcciones
se añaden como enmiendas selladas (E1), sin tocar el pasado.

## Medición del riesgo · `riesgo/`

Mide y comprueba; no decide nada ni toca el diario.

| Módulo | Qué hace |
|---|---|
| `var.py` | VaR y CVaR a un día desde las posiciones, por cuatro métodos: histórico, normal, t de Student y GARCH-t |
| `pruebas.py` | Pone a prueba el VaR: Kupiec (¿se supera con la frecuencia prometida?) y Christoffersen (¿las excepciones llegan en rachas?) |
| `futuro.py` | Monte Carlo por bloques: probabilidad de caídas a 1, 3 y 12 meses |
| `cartera.py` | El riesgo de una cartera concreta: la que el usuario tiene hoy |
| `evidencia.py` | Filtro que toda idea para ganar dinero debe superar: fuera de muestra, 30 eventos o más, corrección por comparaciones múltiples y después de costes. El umbral se endurece con cada idea evaluada |
| `informe.py` | Riesgo del modelo frente a la inversión pasiva |

## Didáctica · `laboratorio/` y `clase/`

### El laboratorio (sesiones 2 a 4)

Nueve pasos, cada uno en su módulo de `cryptoquant/laboratorio/` y en su sección de
`laboratorio/r/laboratorio.Rmd`:

| Paso | Módulo | Tema |
|---|---|---|
| 1 | `datos.py` | Velas de Binance, DataFrame, gráfico de velas y CSV |
| 2 | (R) | Procesamiento con dplyr y ggplot2 |
| 3 | `variables.py` | Indicadores como columnas, rezagos y objetivo |
| 4 | `cuantitativo.py` | Colas gruesas y estadística de señales |
| 5 | `series.py` | Retornos, autocorrelación, patrones por hora y día, ARIMA |
| 6 | `volatilidad.py` | GARCH(1,1) y su uso en el stop y el tamaño |
| 7 | `aprendizaje.py` | Random forest con prueba en el futuro y línea base |
| 8 | `montecarlo.py` | Trayectorias de precio y simulación de la cuenta |
| 9 | `control.py` | Control de volatilidad frente a comprar y mantener a igual volatilidad |

`informe.py` ejecuta los nueve y deja tablas y gráficos en `reports/laboratorio/`; `paquete.py`
prepara los CSV de respaldo de una clase. Python y R parten del mismo CSV y dan las mismas
variables (diferencia máxima de 10⁻⁸).

### Los materiales de la clase

```text
generar_cuadernos.py ──┬──► alumno.ipynb   prompt · celda para pegar · verificador
   (fuente única)      └──► docente.ipynb  lo mismo + solución de referencia

celda Preparación = clase _Verificador
   cadena() · velas() · csv() · riesgo() · prediccion()
   + un manejador de NameError que explica «Python no conoce X» sin la traza en inglés

cryptoquant paquete ──► datos/*.csv + LEEME.txt (huellas SHA-256)   → Drive y Posit Cloud
posit/clase01.R     ──► el mismo CSV en R (recorta desde 2023 para coincidir con Python)

tests/test_clase.py: los cuadernos están al día con el generador, y cada verificador
                     detecta los errores típicos de la IA y deja pasar el trabajo correcto
```

Los verificadores no juzgan el código: juzgan el resultado. Comparan con valores conocidos (el
cierre real del 01-01-2023), recalculan lo que el alumno debió obtener y reconocen los errores
frecuentes (volatilidad en porcentaje, diaria o con 252 días; fechas de 1970; vela sin cerrar;
exactitud sospechosa por fuga de futuro).

## El piloto de riesgo · `piloto/`

Una app Streamlit que aplica el control de volatilidad a la cartera real del usuario y le dice
cuánto tener en cripto. Solo recomienda: no usa claves de API ni envía órdenes.

| Módulo | Papel |
|---|---|
| `app.py` | La interfaz |
| `sesion.py` | De la cartera guardada al plan de hoy; lo comparten la app y el modo `--texto` |
| `motor.py` | La regla: parte en cripto = mínimo(máximo, objetivo ÷ volatilidad prevista) × freno, con banda de no operar |
| `mercado.py` | Velas diarias de Binance y el precio del momento |
| `diario.py` | La cartera, los ajustes y una foto por cambio, en `data/piloto/` (fuera de git) |
| `vista.py` | Formatos en español y lectura de las tablas que edita el usuario |

Maneja dinero real, así que sus tests exigen invariantes: no vender más de lo que hay, no comprar
sin efectivo y no pasar del máximo.

## La tesis · `tesis.py`

Contrastes retrospectivos de las hipótesis de la tesis doctoral (HE1, HE3 a HE7) sobre la historia
larga de precios, con bootstrap por bloques. Deja cifras y figuras en `reports/tesis/`; la réplica
en R está en `laboratorio/r/tesis.R`. Las cifras de la sesión 1 sobre el control de volatilidad
(9,9 % frente a 68,8 % de volatilidad) salen de aquí.

## Principios que sostienen las cifras

| Principio | Cómo se garantiza |
|---|---|
| Ninguna variable mira al futuro | `test_features_are_causal` calcula las variables sobre la serie entera y sobre la serie cortada, y exige que el pasado coincida exactamente |
| El backtest no se engaña | Decisión en t, retorno en t+1, costes sobre la rotación y Sharpe deflactado por el número de pruebas |
| El pasado no se retoca | Diario de solo añadido encadenado por hashes, pre-registro y enmiendas selladas |
| Las decisiones se reproducen | `forward reproduce` recalcula las decisiones pasadas y exige el mismo resultado |
| Una idea nueva no entra por suerte | Filtro de `evidencia`, con umbral que sube con cada idea probada |
| La medida de riesgo también se prueba | Backtest del VaR con Kupiec y Christoffersen |
| Dos copias no bifurcan el experimento | `COPIA_DE_TRABAJO.txt` impide que la copia de trabajo escriba en el diario |

## Mapa de carpetas del repositorio

| Carpeta | Contenido |
|---|---|
| `cryptoquant/` | El programa (paquete de Python) |
| `clase/` | Materiales de cada sesión del curso; esta carpeta incluida |
| `laboratorio/` | Cuaderno de Python y réplica en R del laboratorio |
| `tests/` | Pruebas automáticas (`python -m pytest tests/ -q`) |
| `scripts/` | Tarea programada, sincronización entre copias, piloto con doble clic |
| `config.yaml` | Todos los parámetros de riesgo |
| `data/` | Caché de precios, diario del forward test, cartera del piloto |
| `reports/` | Resultados: tablas, gráficos, panel |
| `investigacion/` | Base documental de la tesis (antecedentes, referencias) |

## Pruebas automáticas

| Archivo | Qué protege |
|---|---|
| `test_core.py` | Causalidad de las variables, sizing, cartera, backtest |
| `test_forward.py` | Diario sellado: hashes, pre-registro, enmienda, reproducción |
| `test_dashboard.py` | El panel HTML |
| `test_laboratorio.py`, `test_bordes_laboratorio.py` | Los nueve pasos y sus casos límite |
| `test_control_y_fuentes.py` | Control de volatilidad y fuentes de datos |
| `test_riesgo.py`, `test_bordes_riesgo.py` | VaR, pruebas, cartera y evidencia |
| `test_clase.py` | Cuadernos al día y verificadores de la clase |
| `test_piloto.py`, `test_piloto_app.py` | El motor del piloto, sus invariantes de dinero y la app |
| `test_tesis.py` | Los contrastes de la tesis |

## Limitaciones declaradas

Solo posiciones largas y al contado, sin apalancamiento ni derivados. Sin datos on-chain ni de
sentimiento. Sesgo de supervivencia: el universo son monedas que existen hoy. El backtest ejecuta
al precio de cierre. El programa no envía órdenes: ejecutar es una decisión aparte, con sus propios riesgos.
