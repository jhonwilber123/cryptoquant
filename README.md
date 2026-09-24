# cryptoquant

Sistema cuantitativo de inversión en criptoactivos. Combina econometría de
series temporales, aprendizaje automático y teoría de carteras bajo una capa
explícita de control de riesgo.

El objetivo del diseño no es maximizar el retorno: es **hacer que el riesgo sea
una variable que tú fijas**, en lugar de una que el mercado te impone.

---

## Lo primero, sin rodeos

Ningún software elimina el riesgo de invertir en cripto. Lo que sí se puede
hacer, y es lo que hace este sistema, es **medirlo, acotarlo y reducirlo de
forma sistemática**:

- Fijas una volatilidad objetivo (p. ej. 15% anual, frente al ~60-80% típico de
  comprar y mantener BTC) y el sistema ajusta cuánto capital está invertido para
  acercarse a ella. La volatilidad futura es mucho más predecible que la
  dirección futura; ahí es donde hay ventaja real.
- Un cortacircuito reduce la exposición conforme crece el drawdown y la anula
  antes de un umbral que tú defines.
- El criterio de Kelly convierte la ventaja estimada en un factor de convicción
  entre 0 y 1, que solo puede recortar el tamaño, nunca inflarlo.
- El resto del capital permanece en efectivo o stablecoin. Estar fuera del
  mercado es una posición legítima y el sistema la toma con frecuencia.

El precio de esto es real: **con menos riesgo se captura menos subida**. En un
mercado alcista vertical, comprar y mantener gana. Este sistema está construido
para la otra mitad del ciclo.

---

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> **Nota sobre este equipo:** el proyecto vive en
> `C:\Users\ADMIN\Proyectos\Blockchain y criptoactivos` y el entorno virtual en
> `C:\Users\ADMIN\.venvs\cryptoquant`. Estuvo antes en `D:`, pero esa unidad
> es un USB extraíble (FAT32): la tarea programada fallaba si no estaba
> conectado a la hora de dispararse. Para usarlo:
> ```powershell
> C:\Users\ADMIN\.venvs\cryptoquant\Scripts\python.exe -m cryptoquant backtest
> ```

## Uso

```bash
python -m cryptoquant analyze              # diagnóstico econométrico del universo
python -m cryptoquant pairs                # pares cointegrados (valor relativo)
python -m cryptoquant backtest --compare   # backtest walk-forward + comparativa
python -m cryptoquant recommend --capital 5000
python -m cryptoquant dashboard --open    # panel visual en el navegador

python -m cryptoquant forward init       # pre-registrar hipótesis (una vez)
python -m cryptoquant forward record     # registrar la decisión de hoy (a diario)
python -m cryptoquant forward verify     # comprobar que nadie tocó el pasado
python -m cryptoquant forward report     # contrastar las hipótesis
python -m cryptoquant forward reproduce  # recomputar decisiones pasadas
```

Todos los parámetros viven en [config.yaml](config.yaml). El más importante es
`risk.target_volatility`.

---

## Panel visual

```bash
python -m cryptoquant dashboard --open
```

Genera `reports/dashboard.html`: un único fichero que se abre con doble clic,
sin servidor, sin instalar nada y sin conexión. La tarea programada lo
regenera en cada pasada, así que siempre refleja el último estado.

- **Posición recomendada** — cuánto invertir hoy y por qué, en lenguaje llano.
  Escribe tu capital y reparte los importes por activo al momento.
- **El experimento en vivo** — progreso de las hipótesis pre-registradas y
  resultado acumulado frente al mercado.
- **Histórico** — curva de capital y caídas frente a buy & hold, con el
  veredicto honesto: el riesgo está controlado, la ventaja no está demostrada.
  Incluye la comparación justa, contra buy & hold a igual volatilidad.
- **Mercado** — precio, tendencia y volatilidad de cada activo.

Es de solo lectura: no descarga datos, no entrena modelos y no toca el diario.
Sigue el tema claro/oscuro del sistema y la configuración regional para
decimales y fechas. Los gráficos se recorren también con el teclado.

---

## Cómo funciona

```
Datos (ccxt / caché)
  │
  ├─ Features causales ─────── momentum ajustado por vol, RSI, MACD, ATR,
  │                             asimetría, curtosis, Hurst, régimen
  ├─ Econometría ───────────── GJR-GARCH, VaR/CVaR, ADF/KPSS, cointegración
  ├─ Machine learning ──────── etiquetado triple-barrera → P(operación rentable)
  │                             gradient boosting + calibración isotónica
  ├─ Cartera ───────────────── HRP sobre covarianza con shrinkage Ledoit-Wolf
  └─ Control de riesgo ─────── vol objetivo × convicción × drawdown × régimen
                                                    │
                                              Pesos finales
```

### 1. Datos — [`cryptoquant/data/sources.py`](cryptoquant/data/sources.py)

Descarga OHLCV vía `ccxt` (API pública, sin claves), cachea en parquet y, si no
hay red, degrada a series sintéticas reproducibles para que el pipeline siga
siendo ejecutable y testeable. El origen de cada serie se reporta siempre: si
ves `synthetic`, **los números no dicen nada del mercado real**.

### 2. Features — [`cryptoquant/features/technical.py`](cryptoquant/features/technical.py)

Todas estrictamente causales. El momentum se normaliza por volatilidad, lo que
hace comparables activos con perfiles de riesgo muy distintos y permite
rankearlos transversalmente.

### 3. Econometría — [`cryptoquant/econometrics/`](cryptoquant/econometrics/)

- **`volatility.py`** — GJR-GARCH(1,1,1) con innovaciones *t* asimétrica. El
  término GJR captura el efecto apalancamiento: las caídas generan más
  volatilidad futura que las subidas equivalentes. Respaldo EWMA cuando no
  converge.
- **`risk.py`** — VaR histórico, paramétrico y de Cornish-Fisher, más CVaR.
  Se calculan los tres a propósito: la distancia entre el paramétrico y los
  otros dos *es* la medida de cuánto riesgo esconde el supuesto de normalidad.
- **`diagnostics.py`** — ADF, KPSS, Ljung-Box, ARCH-LM, Jarque-Bera, Hurst y
  cointegración de Engle-Granger con corrección de Bonferroni. Responden a la
  pregunta previa a todo modelo: *¿hay estructura explotable, o es ruido?*

### 4. Machine learning — [`cryptoquant/models/`](cryptoquant/models/)

El modelo **no predice el precio**. Predice `P(la operación sea rentable)`, que
es justo lo que necesita el dimensionamiento por Kelly.

- **Etiquetado triple-barrera** — etiqueta según qué toca primero el precio:
  objetivo de beneficio, stop, o vencimiento. Las barreras escalan con la
  volatilidad. Etiquetar con el retorno a N días fijos ignora que una posición
  real tiene stop, y es el error más común al aplicar ML a mercados.
- **Pesos por unicidad** — muestras con horizontes solapados comparten
  información; contarlas como independientes infla el tamaño efectivo de la
  muestra y hace que el modelo parezca más fiable de lo que es.
- **Calibración isotónica** sobre un bloque temporal reservado. Un modelo con
  buen AUC pero probabilidades mal calibradas destruye el sizing de Kelly.
- **Validación purgada con embargo** — la CV aleatoria de sklearn es inválida en
  series temporales: mezcla pasado y futuro y produce métricas que no se
  reproducen en real.

### 5. Cartera — [`cryptoquant/portfolio/optimizer.py`](cryptoquant/portfolio/optimizer.py)

Por defecto **HRP** (Hierarchical Risk Parity). Markowitz clásico funciona mal
aquí: los retornos esperados se estiman con enorme error y el optimizador los
amplifica. HRP no necesita invertir la matriz de covarianzas ni estimar
retornos esperados, lo que importa mucho en cripto, donde casi todo cotiza
contra BTC y las correlaciones convergen a 1 justo en los crashes.

### 6. Control de riesgo — [`cryptoquant/portfolio/sizing.py`](cryptoquant/portfolio/sizing.py)

Este módulo es el que determina de verdad el perfil de la cartera. La selección
de activos decide en qué inviertes; el sizing decide **cuánto pierdes cuando la
selección se equivoca**, que es lo que realmente importa.

```
exposición = (vol_objetivo / vol_prevista) × convicción × drawdown × régimen
             └──── escala de riesgo ────┘   └──── factores en [0,1] ────┘
```

Hay **una sola escala**, y es deliberado. Que dos capas recorten por el mismo
riesgo es el error que hunde estos sistemas: si Kelly multiplica por 0.25 y el
objetivo de volatilidad por otro 0.25, la exposición se queda en el 6% y
`target_volatility` deja de significar nada.

| Capa | Qué hace |
|---|---|
| Objetivo de volatilidad | **Escala** el capital: vol prevista 40%, objetivo 15% → se invierte el 37.5% |
| Convicción (Kelly) | `[0,1]` según cuánta ventaja estimada hay. `kelly_full_size_level` es el Kelly bruto a partir del cual se despliega el presupuesto completo |
| Filtro de régimen | `[0,1]` recorta al 25% si menos de la mitad del universo está en tendencia alcista |
| Freno por drawdown | `[0,1]` desapalanca linealmente entre el −10% y el −25% |
| Topes duros | Máximo **absoluto** por activo y exposición bruta máxima |

### 7. Backtest — [`cryptoquant/backtest/engine.py`](cryptoquant/backtest/engine.py)

Walk-forward con tres garantías: decisión en `t` con datos hasta `t`, retornos
aplicados en `t+1`, y costes cobrados sobre la rotación efectiva. Sin esas tres
cosas, un backtest produce curvas espectaculares que no se reproducen con dinero
real.

Se reporta además el **Sharpe deflactado**: si pruebas 50 estrategias y eliges
la mejor, su Sharpe está sesgado al alza por pura selección. Por debajo del 95%,
el resultado no se distingue de la suerte.

---

## Resultados sobre datos reales

Binance, 8 activos, 2178 barras diarias (2020-09-22 → 2026-09-08), neto de
comisiones (10 pb) y slippage (5 pb), con `target_volatility: 0.15`:

| | Sistema | Sin ML | Equiponderado | Buy & hold |
|---|---|---|---|---|
| CAGR | 6.81% | 4.29% | 5.73% | **10.31%** |
| Volatilidad | **9.20%** | 9.89% | 9.18% | 68.78% |
| Sharpe | **0.74** | 0.43 | 0.62 | 0.15 |
| Max drawdown | **−14.98%** | −18.13% | −15.00% | −81.53% |
| Martin ratio | **0.99** | 0.50 | 0.83 | 0.20 |

**Léelo con cuidado, porque dice dos cosas opuestas.**

Lo bueno: el riesgo baja de forma brutal. La volatilidad pasa de 68.8% a 9.2%
(7.5×) y el peor drawdown de −81.5% a −15.0%. Eso es exactamente lo que se
pedía: la caída del 81% es la que hace que la gente venda en el peor momento y
no vuelva.

Lo malo, y es lo que importa: **el buy & hold gana más dinero** (10.31% vs
6.81% anual). Y sobre todo, el **Sharpe deflactado es del 51.4%**, muy por
debajo del 95% exigible. Traducido: descontando que he probado varias
configuraciones y me he quedado con la que mejor salía, este resultado **no se
distingue estadísticamente de la suerte**. El propio sistema lo dice al final
de cada backtest y recomienda no desplegar capital sobre esa base.

Un solo ciclo de mercado (2020-2026) no basta para demostrar una ventaja. Lo
que sí está demostrado aquí es el **control de riesgo**, que es mecánico y no
depende de acertar: si fijas 15% de volatilidad, el sistema entrega 11.7%
mientras está invertido. Eso funciona tengas ventaja predictiva o no.

---

## Verificación

```bash
python -m pytest tests/ -q
```

El test que sostiene la validez de todo lo demás es
`test_features_are_causal`: calcula las features sobre la serie completa y sobre
la serie truncada, y exige que los valores del pasado coincidan exactamente. Si
alguna feature mira al futuro, ese test falla y todas las métricas del backtest
son ficción.

---

## Forward test — [`cryptoquant/forward/`](cryptoquant/forward/)

El backtest dice que hay un +2,5% anual de ventaja. El forward test existe para
comprobar en vivo qué parte de eso es real. Con una advertencia que va por
delante, porque determina para qué sirve el experimento:

```
Ventaja observada : +2,51% anual        Años para p<0,05  : 46,5
Tracking error    :  8,75% anual        Con 80% potencia  : 95,0
Information ratio :  0,287
```

**Ningún forward test va a demostrar esa ventaja.** Harían falta décadas. La
volatilidad, en cambio, converge rápido: con 180 observaciones se estima con un
error relativo del 5%. Por eso el experimento está pre-registrado con tres
hipótesis y una de ellas se declara irresoluble de entrada:

| | Hipótesis | Resoluble |
|---|---|---|
| **H1** | El control de riesgo entrega la volatilidad objetivo en vivo | Sí, ~6 meses |
| **H2** | Las decisiones en vivo reproducen el backtest | Sí, ~1 mes |
| **H3** | El sistema bate al buy & hold ajustado a igual volatilidad | **No, ~46 años** |

### Por qué el diario está sellado

Un forward test solo vale si es imposible retocarlo después. El registro es de
solo-añadido y está encadenado por hashes SHA-256: cada anotación incluye el
hash de la anterior.

```
$ python -m cryptoquant forward verify
  Cadena de hashes     : ROTA en el registro 1
  - registro 1 (2026-09-08): el contenido no coincide con su hash (fue modificado)
```

Editar un peso pasado rompe el hash de esa línea. Recalcular ese hash rompe el
enlace de la línea siguiente. Ambos casos tienen test. No impide reescribir el
fichero entero a mano —- nada lo impide -— pero sí hace imposible el retoque
silencioso, que es el fallo realista.

### El pre-registro

Las hipótesis y sus umbrales se fijan **antes** de tener un solo dato, y quedan
hasheadas. Sin eso, a los seis meses siempre aparece alguna métrica en la que el
sistema sale bien. `forward init` se niega a sobrescribir un pre-registro
existente sin `--overwrite`.

### Automatización

La anotación diaria corre sola mediante una tarea del Programador de Windows:

```powershell
.\scripts\install_task.ps1 -Install    # registrar
.\scripts\install_task.ps1 -Status     # estado + últimas líneas del log
.\scripts\install_task.ps1 -RunNow     # lanzar sin esperar
.\scripts\install_task.ps1 -Uninstall  # quitar (no toca el diario)
```

Se ejecuta **cada 6 horas**, no una vez al día. La barra diaria cierra a las
00:00 UTC y la primera ventana cae a las 00:30 UTC; las otras tres son
reintentos por si el equipo estaba apagado o sin red. El diario rechaza fechas
duplicadas, así que las ejecuciones sobrantes no hacen nada.

Los días perdidos **no se rellenan después**. La posición simplemente se
mantuvo, que es lo que habría pasado en la realidad. `forward status` muestra la
cobertura y avisa si la última anotación tiene más de 3 días.

Tres cosas que el sistema se niega a hacer sin supervisión:

- **Registrar sobre datos sintéticos.** Si no hay red y no hay caché real, aborta.
- **Registrar sobre caché atrasada.** Si la copia local no cubre la última barra
  cerrada, aborta y espera al siguiente reintento.
- **Registrar sobre la barra en curso.** Se descarta el periodo sin cerrar, más
  un margen de 2 minutos para que el exchange consolide la vela.

### H2: causalidad en producción

Todo el sistema es causal, así que **recalcular hoy la decisión de una fecha
pasada debe dar exactamente el mismo resultado que se registró entonces**.
`forward reproduce` lo comprueba. Si difiere: o el exchange revisó su histórico,
o hay una fuga de futuro que el test unitario no cazó.

Backtest y modo en vivo comparten un único método de decisión
([`_decide`](cryptoquant/backtest/engine.py)), precisamente para que no puedan
divergir.

---

## Limitaciones, explícitas

- **Solo posiciones largas y spot.** Sin cortos, sin derivados, sin apalancamiento.
- **Sin datos on-chain** (flujos de exchanges, direcciones activas, TVL) ni de
  sentimiento. Son señales con valor documentado y no están aquí.
- **Sesgo de supervivencia.** El universo son monedas que existen hoy. Las que
  murieron no aparecen, lo que infla el resultado histórico.
- **El backtest asume que se ejecuta al cierre** al precio de cierre. En la
  realidad hay latencia y el libro de órdenes se mueve.
- **El régimen de mercado cambia.** Un modelo entrenado sobre 2019-2025 codifica
  las regularidades de ese periodo, no leyes.
- **No ejecuta órdenes.** No hay conexión con claves de API ni gestión de
  órdenes. Es deliberado: la ejecución automática es una decisión aparte, con
  su propio conjunto de riesgos.

---

## Aviso

Esto es software de análisis estadístico, no asesoramiento financiero. Sus
salidas son estimaciones de un modelo, con error. El rendimiento pasado no
predice el futuro y en cripto las pérdidas totales son un desenlace posible.
Invierte solo lo que puedas permitirte perder.
