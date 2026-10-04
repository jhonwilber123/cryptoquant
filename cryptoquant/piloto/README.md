# Piloto de riesgo

Una app para el navegador que responde a una pregunta: **con lo que tengo hoy,
¿cuánto debería tener en cripto para que mi cartera no se mueva más de lo que
acepto?** Es el control de volatilidad de la tesis (paso 9 del laboratorio)
aplicado a su cartera real.

Solo recomienda. No se conecta a su cuenta, no necesita claves de API y no
envía órdenes: usted decide y ejecuta en Binance.

## Abrirlo

```bash
python -m cryptoquant piloto
```

O doble clic en [`scripts/piloto.cmd`](../../scripts/piloto.cmd), que usa el
entorno del proyecto (`%USERPROFILE%\.venvs\cryptoquant`) o, si no existe, el
`python` del sistema. Se abre el navegador en `http://127.0.0.1:8501` (o el
siguiente puerto libre). Para cerrarlo, cierre la ventana negra o pulse Ctrl+C
en ella.

Necesita Streamlit, que está en `requirements.txt`. Si falta, el piloto lo dice
y muestra la orden para instalarlo; mientras tanto, `--texto` funciona sin él.

| Opción | Para qué |
|---|---|
| `--texto` | imprime el plan de hoy en la consola, sin abrir la app |
| `--puerto 8600` | otro puerto, si el 8501 está ocupado |
| `--sin-navegador` | arranca el servidor sin abrir el navegador |
| `--red` | deja entrar a otros equipos de su red (ver *Seguridad*) |

## La primera vez

1. En la barra lateral, anote lo que tiene: una fila por moneda (`BTC 0,05`,
   `ETH 1,2`, `USDT 500`...). USDT, USDC, FDUSD y DAI cuentan como efectivo.
2. Elija cuánto movimiento aguanta: **Prudente** (15 % al año, el de la tesis),
   **Moderado** (25 %), **Decidido** (40 %) o una cifra a medida. Comprar y
   mantener bitcoin se ha movido un 50-60 % al año.
3. Pulse **Guardar cartera y ajustes**. Se guarda aunque el plan de hoy no se
   pueda calcular (por ejemplo, sin conexión).

Vuelva a guardar **cada vez que compre, venda, ingrese o retire**. No hace
falta abrirlo a diario: entre una vez y otra, el piloto valora sus mismas
unidades con los cierres diarios de Binance.

## Qué muestra

**Hoy.** Cuánto tiene en cripto, cuánto le recomienda y las órdenes concretas
(vender o comprar tantas unidades de cada moneda) para llegar. Si su efectivo
no alcanza para todas las compras, se recortan a lo que pagan el efectivo y las
ventas de la lista, descontado su coste, y el piloto lo avisa. Debajo, el
porqué con las cifras del día y qué parte del riesgo aporta cada moneda: una
puede ser el 30 % del dinero y el 50 % del riesgo.

**Tiempos duros.** Qué perdería si se repitiera el peor día, semana, mes,
trimestre y año desde 2020, con su cartera de hoy y siguiendo la
recomendación; la pérdida de un mal día (1 de cada 20) y la probabilidad de
caer más de un 10 o un 20 % el próximo mes. Un botón contrasta el mal día con
los 4 métodos de VaR del módulo de riesgo (histórico, normal, t de Student y
GARCH-t) y dice si cada uno habría acertado con su cartera.

**La regla en el pasado.** Qué habría pasado desde 2020 con su reparto y sus
ajustes, frente a comprar y mantener y frente a mantener una parte fija con la
misma volatilidad, que es la comparación justa. Aquí «comprar y mantener» es su
reparto siempre invertido, rebalanceado cada día y sin costes; con una sola
moneda es exactamente comprar y mantener.

**Mi diario.** El valor de su cartera día a día, sin contar ingresos ni
retiros, y sus fotos guardadas.

## La regla

```
parte en cripto = mínimo(máximo, objetivo ÷ volatilidad prevista) × freno
```

- **Volatilidad prevista**: media exponencial (EWMA, λ = 0,94) de los cierres
  diarios de su mezcla. No predice el precio, solo cuánto se va a mover, y eso
  sí se puede prever porque los días agitados vienen en rachas.
- **Freno por caídas**: si su cartera cae más de un 10 % desde su máximo de los
  últimos 180 días, invierte menos aún; con una caída del 25 % o más, solo un
  25 % de lo normal.
  - Se suelta solo, porque la caída se mide contra una ventana móvil.
  - El freno del sistema principal mide contra el máximo histórico. Aplicado
    aquí, una caída del 25 % dejaría la cuenta en efectivo y el freno no se
    soltaría nunca: la cuenta en efectivo ya no se mueve y la caída no se
    recupera. Hay un test que lo demuestra.
- **Banda**: si el cambio es menor que el 5 % de la cartera, no se opera.
- **Reparto**: el piloto decide cuánto en cripto; el reparto entre monedas es
  suyo. Puede ser el que ya tiene, «igual riesgo por moneda» (más peso a las
  que se mueven menos) o uno a medida.

Todo se puede cambiar en **Más ajustes**. Los valores de fábrica salen de
simular la regla con datos reales de 2020 a 2026:

| BTC, ETH y SOL a partes iguales | Rentabilidad anual | Volatilidad | Caída máxima | Operaciones |
|---|---|---|---|---|
| Sin freno, sin banda | +16,1 % | 15,9 % | −29,3 % | 2206 |
| Sin freno, banda del 5 % | +17,3 % | 15,7 % | −29,7 % | 121 |
| **Freno y banda (de fábrica)** | **+18,1 %** | **15,1 %** | **−26,0 %** | **120** |
| Comprar y mantener¹ | +67,5 % | 72,5 % | −85,5 % | — |

¹ Las tres siempre invertidas a partes iguales, rebalanceadas cada día y sin
costes: la referencia que usa el piloto.

- **La banda** evita casi todas las operaciones (de más de 2000 a unas 120 en
  seis años y medio) sin empeorar el resultado.
- **El freno** aporta poco, porque el control de volatilidad ya recorta en las
  caídas: con solo BTC, la caída máxima pasa de −27,6 % a −27,3 %.
- **Comprar y mantener** ganó mucho más en este periodo alcista, a cambio de
  caídas del 85 %. El piloto no promete ganar más, sino caer menos.

La tesis lo confirma en 8 criptoactivos (2021-2026): reducir el riesgo, sí
(HE3); ganar más que la pasiva con el mismo riesgo, no demostrado (HE4).

## Sus datos

Todo lo personal queda en `data/piloto/` y **no va a git** (está en
`.gitignore`, y un test lo comprueba):

| Archivo | Qué es |
|---|---|
| `cartera.json` | su cartera y sus ajustes |
| `fotos.jsonl` | una línea por cada vez que guardó una cartera distinta |
| `velas/` | cierres diarios de Binance, para no volver a bajarlos |

- **Copia de seguridad:** copie la carpeta.
- **Empezar de cero:** bórrela.
- **Otra ubicación:** la variable de entorno `CRYPTOQUANT_PILOTO` cambia la
  carpeta.
- **Archivos dañados:** si `cartera.json` se estropea (un corte de luz a mitad
  de escritura), el piloto lo aparta como `cartera.danada-<fecha>.json`, avisa
  y empieza de cero. Una línea dañada de `fotos.jsonl` se ignora con un aviso.
  Cada guardado escribe primero en un archivo temporal y lo sustituye de golpe.

## Seguridad

- **Solo en este equipo.** El servidor escucha en `127.0.0.1`: nadie más de su
  red puede abrirlo. `--red` lo abre a su red local y avisa de que cualquiera
  en ella vería su cartera. No lo use en redes públicas.
- **En un servidor propio**, para abrirlo desde el móvil: con Docker, HTTPS y
  una contraseña sin la que la app no muestra nada (`acceso.py`). Cómo
  publicarlo, en [`deploy/LEEME.md`](../../deploy/LEEME.md).
- **Sin claves ni órdenes.** No hay código que firme peticiones ni que opere;
  un test busca en el código cualquier rastro de claves o endpoints privados.
- **Qué sale de su equipo.** Solo peticiones públicas a
  `data-api.binance.vision` con los símbolos que consulta (p. ej. `BTCUSDT`).
  Binance sabe qué monedas mira, pero no cuánto tiene. La telemetría de
  Streamlit está desactivada.
- **Entradas validadas.** Los nombres de moneda solo admiten letras y cifras,
  así que nunca forman rutas raras. Las cantidades negativas, infinitas o no
  numéricas se rechazan con un mensaje.

## Si algo falla

| Mensaje | Qué hacer |
|---|---|
| «no hay par XYZ/USDT en Binance o no hay conexión» | Revise el nombre (`ETH`, no `Ethereum`) y la conexión. |
| «Binance no respondió; se usan velas hasta el...» | Sin red, el piloto usa las últimas velas guardadas. Vuelva a intentarlo más tarde. |
| «hacen falta 60 días de historia» | La moneda es demasiado nueva para prever su riesgo. |
| «Sin precio de este momento para...» | Se valoró con el último cierre diario. Pulse *Actualizar precios*. |
| «Algo ha fallado al calcular el plan» | Su cartera guardada no se toca, y puede seguir guardando cambios. Pulse *Actualizar precios* o reinicie. El detalle técnico está debajo. |
| «Falta Streamlit, que la app necesita» | Instálelo con la orden que muestra el mensaje. Mientras tanto, `python -m cryptoquant piloto --texto` da el plan en la consola. |
| La ventana de `piloto.cmd` muestra un error y espera una tecla | El piloto no arrancó. Si falta Streamlit, vea la fila anterior; si no encuentra Python, cree el entorno del proyecto o abra el piloto desde una terminal con su entorno activado. |

## Límites

- Un desplome de un día no avisa: la previsión reacciona al día siguiente.
- Las estables no son riesgo cero: pueden perder la paridad, y un exchange
  puede fallar.
- Las cifras del pasado usan un coste del 0,15 % por operación. Binance
  redondea las cantidades a su paso mínimo por moneda.
- Es software de análisis, no asesoramiento financiero. Invierta solo lo que
  pueda permitirse perder.

## Para quien lo modifique

| Módulo | Qué hace |
|---|---|
| `motor.py` | la decisión, la caída de la cuenta, la simulación y los escenarios, sin red ni interfaz |
| `sesion.py` | de la cartera guardada al plan: lo comparten la app y `--texto` |
| `mercado.py` | velas y precio de ahora, de `data-api.binance.vision` |
| `diario.py` | cartera, ajustes y fotos en disco |
| `vista.py` | formatos en español y lectura de las tablas del usuario |
| `acceso.py` | la contraseña cuando se publica en un servidor |
| `app.py` | la interfaz (Streamlit) |

Tests, todos sin red:

- **`tests/test_piloto.py`**, con los invariantes de dinero sobre 300 carteras
  al azar:
  - nunca vende más de lo que hay;
  - nunca compra sin efectivo, ni en el plan ni en las órdenes que enseña:
    una venta de menos del mínimo por orden no se hace, y la compra que
    pagaba se recorta a lo que hay; las ventas pagan compras descontado su
    coste, y si una compra queda por debajo del mínimo, su efectivo pasa a
    las demás;
  - nunca pasa del máximo.

  Además comprueba que la simulación coincide con el laboratorio, que todo es
  causal, los casos borde y la seguridad.
- **`tests/test_piloto_app.py`**: pulsa cada control de la app con `AppTest`,
  y comprueba que se puede guardar aunque el plan no se pueda calcular.
