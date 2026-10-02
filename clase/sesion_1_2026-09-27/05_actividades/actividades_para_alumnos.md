# Sesión 1 · Actividades

**Módulo:** Análisis cuantitativo y modelamiento de datos en criptoactivos con Python y RStudio
**Fecha:** domingo 27-09-2026, 19:00–21:00

En esta clase no escribes código a mano: se lo pides a **Claude**, lo pegas en Google Colab y
compruebas si funciona. Esa última parte es la importante. La IA se equivoca, y cada actividad
termina en un **verificador** que responde `[OK]` o `[REVISAR]` y te dice qué pedirle a Claude.

Nada de esto es consejo de inversión: aprendemos a **medir**, no a apostar.

## Antes de empezar

1. Abre el enlace del cuaderno `alumno.ipynb` y haz **Archivo › Guardar una copia en Drive**.
2. Ejecuta la celda **Preparación** con el botón ▶. Debe responder `Listo. Ya puedes seguir con el Bloque 1.`
3. Abre <https://claude.ai> en otra pestaña.

**En cada actividad:** copia el prompt, pégalo en Claude, copia el código que te devuelva, pégalo
en la celda *Pega aquí* y ejecútala con **Shift + Enter**. Después ejecuta la celda del verificador.

---

## Actividad 1 · Bloque 1 · Construye una blockchain y falsifícala (10 min)

Una blockchain es una lista de bloques donde **cada bloque guarda la huella (hash) del anterior**.

**Prompt para Claude:**

```text
Escribe en Python, para Google Colab y usando solo la librería estándar, una mini-blockchain:
- cada bloque es un diccionario con indice, fecha, datos (un texto) y hash_anterior;
- una función hash_bloque que calcule el SHA-256 de todo el bloque;
- una función verificar_cadena que diga si cada bloque guarda el hash correcto del anterior.
Crea un bloque génesis y 3 bloques con transacciones como "Ana paga 2 BTC a Luis" e imprime la cadena.
Guarda en valida_antes el resultado de verificar la cadena. Luego cambia el texto del bloque 1 por
"Ana paga 200 BTC a Luis" SIN recalcular ningún hash y guarda en valida_despues el resultado de verificar
otra vez. Explica cada línea con comentarios en español para alguien que nunca programó.
```

**Verificador:** `verificar.cadena(valida_antes, valida_despues)`

**Para pensar:** ¿por qué cambiar el bloque 1 hace que el *bloque 2* deje de cuadrar? ¿Qué
tendría que hacer un falsificador para que la cadena volviera a ser válida, y por qué en Bitcoin
eso es carísimo?

## Actividad 2 · Bloque 2 · Descarga precios reales de Binance (12 min)

Cada fila es una **vela**: precio de apertura, máximo, mínimo, cierre y volumen de un día.

**Prompt para Claude:**

```text
Escribe código Python para Google Colab que descargue las velas diarias de BTCUSDT de Binance
usando la URL https://data-api.binance.vision/api/v3/klines (no uses api.binance.com: está bloqueada
en Colab). Pide los datos desde el 1 de enero de 2023 hasta hoy, en páginas de 1000 velas.
Guarda el resultado en un DataFrame de pandas llamado df con las columnas fecha (la hora de APERTURA,
en UTC), open, high, low, close y volume, con los precios como números. Elimina la vela de hoy si
todavía no ha cerrado. Usa solo requests y pandas. Comenta cada paso en español.
```

**Verificador:** `verificar.velas(df)`. Comprueba, entre otras cosas, que el cierre del
01-01-2023 sea el real de Binance: 16 616,75.

**Plan B**, si la descarga falla: arrastra `BTCUSDT_1d.csv` desde la carpeta de Drive del curso
al panel de archivos de Colab (icono de carpeta, a la izquierda), quita los `#` de la celda
*PLAN B* y ejecútala.

## Actividad 3 · Bloque 2 · Dibuja las velas y guarda el CSV (8 min)

**Prompt para Claude:**

```text
Con el DataFrame df anterior, dibuja con matplotlib un gráfico de velas japonesas de los últimos
90 días (verde si cerró por encima de la apertura, rojo si no), sin instalar librerías nuevas.
Después guarda df en un archivo llamado BTCUSDT_1d.csv con index=False.
```

**Verificador:** `verificar.csv("BTCUSDT_1d.csv")`

Ese archivo es el puente hacia R: descárgalo desde el panel de archivos (clic derecho › Descargar).

## Actividad 4 · Bloque 3 · Mide cuánto se puede perder (12 min)

El **rendimiento** es cuánto sube o baja el precio de un día al siguiente. La **volatilidad** mide
cuánto se mueve. La pregunta: ¿las caídas fuertes son raras, como promete la campana de Gauss?

**Prompt para Claude:**

```text
Con el DataFrame df (velas diarias de BTC con la columna close):
1. Calcula el rendimiento logarítmico diario con np.log(df["close"]).diff().
2. Calcula la volatilidad anual como la desviación estándar de esos rendimientos por la raíz de 365
   (las criptos cotizan todos los días) y guárdala en vol_anual, como fracción (0.45, no 45).
3. Guarda en peor_dia el peor rendimiento simple de un día (df["close"].pct_change().min()).
4. Estandariza los rendimientos (resta la media y divide por la desviación) y guarda en
   dias_mas_de_4sd cuántos días se alejaron más de 4 desviaciones. Compáralo con lo que esperaría
   una distribución normal para el mismo número de días (scipy.stats.norm).
5. Dibuja un histograma de los rendimientos estandarizados con la curva normal encima y el eje
   vertical en escala logarítmica. Comenta todo en español.
```

**Verificador:** `verificar.riesgo(df, vol_anual, peor_dia, dias_mas_de_4sd)`

**Para discutir:** ¿qué le pasa a quien pone un stop «a 3 sigmas» creyendo que casi nunca saltará?

## Bloque 4 · Demo del docente: el CSV en R (10 min)

El docente abre en Posit Cloud el mismo `BTCUSDT_1d.csv` y lo procesa con `dplyr` y `ggplot2`.
Python descarga y R procesa: el CSV es el acuerdo entre ambos. En la sesión 2 lo harás tú.

## Actividad 5 · Bloque 5 · ¿Puede la IA predecir el precio? (8 min)

**Prompt para Claude:**

```text
Con el DataFrame df (velas diarias de BTC), crea en Python un modelo de random forest (scikit-learn)
que prediga si MAÑANA el cierre será mayor que hoy. Usa como variables los rendimientos de los
últimos 5 días, la volatilidad de 20 días y el RSI de 14 días. Entrena con el 70 % más antiguo de los
datos y prueba con el 30 % más reciente, SIN mezclar el orden de las fechas. Guarda la exactitud en
la prueba en exactitud_modelo, y en exactitud_base la de una estrategia tonta que siempre dice "sube".
Comenta cada paso en español.
```

**Verificador:** `verificar.prediccion(exactitud_modelo, exactitud_base)`. Si te sale más del
60 %, desconfía: casi siempre la IA mezcló datos del futuro al separar entrenamiento y prueba.

---

## Cuando algo falla: prompts de rescate

```text
Me salió este error en Google Colab al ejecutar tu código. Explícame en una frase qué significa
y dame el código corregido completo: [pega aquí el error entero]
```

```text
El verificador de mi clase dice: [pega el mensaje de REVISAR]. Corrige el código para que lo cumpla
y explícame qué estaba mal.
```

```text
Revisa este código como un profesor exigente: ¿usa en algún momento información del futuro para
predecir el pasado? Señala la línea exacta si es así. [pega el código]
```

Si aparece `Python no conoce «df»` (u otro nombre), ejecuta antes la celda con el código de Claude.
Si ya lo hiciste, Claude usó otro nombre: pídele que guarde el resultado con el nombre que dice el prompt.

## Preguntas de cierre

1. ¿Qué parte del código de Claude tuviste que corregir? ¿Cómo te diste cuenta?
2. Si nadie puede predecir si BTC sube mañana, ¿qué **sí** se puede hacer con estos datos?
   (Pista: el Bloque 3, la actividad 4, midió algo que cambia despacio y se puede prever.)

## Tarea

Repite los bloques 2 y 3 (actividades 2, 3 y 4) con **ETHUSDT** y compara su volatilidad y su peor día con los de BTC.
Entrega el cuaderno con los verificadores a la vista y un párrafo.

Con ETH, `verificar.velas` dirá `[REVISAR]`: compara con el cierre de BTC del 01-01-2023. No es un
error tuyo. Para la tarea cuentan `verificar.csv` y `verificar.riesgo`.

**Entrega:** antes de la sesión 2 (sábado 03-10, 19:00), por [medio de entrega].

| Criterio | Puntos |
|---|---|
| Los cinco verificadores de clase en `[OK]` | 40 |
| Tarea con ETH: `verificar.csv` y `verificar.riesgo` en `[OK]` | 30 |
| Párrafo: qué se corrigió de Claude y cómo se detectó | 20 |
| Respuesta: qué se puede hacer si no se puede predecir | 10 |

**Uso de la IA:** usar Claude está permitido y se espera. Di qué le pediste. No se acepta nada
entregado sin verificar: se evalúa lo que comprobaste, no el código que escribiste.
