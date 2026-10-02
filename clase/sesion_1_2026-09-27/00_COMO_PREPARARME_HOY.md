# Cómo prepararme para la sesión 1 · hoy, domingo 27-09-2026

- **Clase:** 19:00 a 21:00, hora de Perú. Virtual, unos 38 alumnos que nunca programaron.
- **Módulo:** Análisis cuantitativo y modelamiento de datos en criptoactivos con Python y RStudio (sesión 1 de 4).
- **Ya está hecho y probado:** diapositivas, cuadernos (los 14 tests de la clase pasan), CSV de respaldo con sus huellas y script de R.
- **Lo que falta es tuyo:** subir, probar con otra cuenta, ensayar y avisar a los alumnos.

Tiempo total: unas 2 h 30 min. Termina los pasos 1 a 6 antes de las 17:00 y te sobra margen.

| # | Paso | Tiempo | Hecho |
|---|---|---|---|
| 1 | Revisar las diapositivas y descargarlas | 25 min | ☐ |
| 2 | Subir datos y cuaderno del alumno a Google Drive | 20 min | ☐ |
| 3 | Probar Colab con una cuenta que no sea la tuya | 15 min | ☐ |
| 4 | Ensayar `docente.ipynb` y un prompt en Claude | 30 min | ☐ |
| 5 | Preparar el proyecto de Posit Cloud | 25 min | ☐ |
| 6 | Enviar el mensaje a los alumnos | 10 min | ☐ |
| 7 | 18:00 · Montar el escritorio de la clase | 20 min | ☐ |
| 8 | 18:45 · Entrar a la sala y publicar los enlaces | 15 min | ☐ |

---

## 1. Diapositivas (25 min)

1. Abre `01_diapositivas/Diapositivas sesion 1 (abrir en el navegador).url`
   (<https://claude.ai/artifact/JoV2fcWzkCJkayEsiUaUC8>). Son 31 diapositivas y cada una
   lleva en las notas del orador el tiempo y lo que hay que decir.
2. Hay **un dato pendiente**: en la diapositiva «Tarea» cambia `[medio de entrega]` por
   el medio real (correo, aula virtual de Escuela Global, Classroom…). Se edita en la
   misma página, o me lo pides.
3. Descarga dos copias desde el menú **Compartir › Exportar** y guárdalas en `01_diapositivas/`:
   - **PDF**: conserva las fuentes y el diseño. Es tu respaldo si el navegador falla.
   - **PowerPoint (.pptx)**: por si lo pide Escuela Global o quieres editar. En un equipo
     sin las fuentes (Space Grotesk, IBM Plex Sans, JetBrains Mono) PowerPoint las sustituye.
4. Para presentar, usa el modo **Presentar** de la página o el PDF a pantalla completa.

El enlace es privado: los alumnos no pueden abrirlo mientras no lo compartas.

## 2. Google Drive (20 min)

1. Crea en tu Drive una carpeta llamada `Sesión 1 · datos de respaldo`.
2. Sube los 6 CSV y `LEEME.txt` de `04_datos_respaldo/`.
3. Compártela en **Compartir › Acceso general › Cualquier persona con el enlace › Lector**
   y copia el enlace. Es el plan B del Bloque 2.
4. Sube `02_cuadernos_colab/alumno.ipynb` a Drive, ábrelo con **Google Colaboratory** y
   compártelo igual, como lector. Ese es el enlace de la clase: cada alumno lo abre y hace
   **Archivo › Guardar una copia en Drive**.
5. **No compartas todavía `docente.ipynb`.** Trae las soluciones: se comparte al final de la
   clase, porque si lo tienen desde el principio copian y no aprenden a verificar.

## 3. Probar Colab con otra cuenta (15 min)

Es la única prueba que no se puede hacer desde fuera, y la más importante.

1. Abre una ventana de incógnito, entra con **otra cuenta de Google** y abre el enlace de `alumno.ipynb`.
2. Ejecuta la celda **Preparación** (▶). Debe responder: `Listo. Ya puedes seguir con el Bloque 1.`
3. Crea una celda nueva, pega esto y ejecútalo:

   ```python
   import requests
   for base in ["https://api.binance.com", "https://data-api.binance.vision"]:
       r = requests.get(base + "/api/v3/klines",
                        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 2}, timeout=15)
       print(base, r.status_code)
   ```

   Lo esperable: **451** en el primero y **200** en el segundo.
4. Si el segundo no da 200, la clase se hace con el plan B: los alumnos leen el CSV de Drive.
   Avísalo en el mensaje del paso 6 y ten el enlace de la carpeta a mano.

## 4. Ensayar el cuaderno del docente y un prompt (30 min)

1. Abre `docente.ipynb` en Colab con tu cuenta: **Entorno de ejecución › Ejecutar todas**.
2. Los cinco verificadores deben decir `[OK]`. Antes de las 19:00 deberías ver las mismas
   cifras que las diapositivas:

   | Verificador | Qué debe salir |
   |---|---|
   | `verificar.cadena` | `[OK]` La cadena era válida y, al cambiar un bloque, dejó de serlo |
   | `verificar.velas` | 1 365 velas del 01-01-2023 al 26-09-2026; cierre de control 16 616,75 |
   | `verificar.csv` | `BTCUSDT_1d.csv` con 1 365 filas |
   | `verificar.riesgo` | volatilidad 46 %, peor día −14,0 %, 3 días de más de 4σ (0,09 esperados) |
   | `verificar.prediccion` | modelo 49,3 %, siempre «sube» 48,8 % |

   Durante la clase ya habrá cerrado la vela del 27-09 (la vela diaria cierra a las 19:00 de Perú),
   así que las cifras de los alumnos pueden moverse en los decimales. Es normal.
3. Ensaya en voz alta la demo del hash (diapositiva «Una huella digital»): las dos huellas que
   salgan en Colab deben ser idénticas a las de la diapositiva.
4. En `claude.ai` pega el **Prompt 2**, lleva el código a una copia de `alumno.ipynb` y pásalo
   por `verificar.velas(df)`. Así ves cómo responde Claude hoy y qué error le sale, si le sale alguno.

## 5. Posit Cloud (25 min)

1. En <https://posit.cloud>: **New Project › New RStudio Project**. Nómbralo `Clase 1 · criptoactivos`.
2. En la consola, instala los paquetes (tarda varios minutos; por eso se hace hoy):

   ```r
   install.packages(c("dplyr", "readr", "ggplot2", "jsonlite"))
   ```

3. Sube `03_posit_cloud_R/clase01.R` con **Files › Upload**. Si quieres, sube también
   `BTCUSDT_1d.csv` de `04_datos_respaldo/`.
4. Abre `clase01.R` y ejecútalo con **Source** (o línea a línea con Ctrl + Enter). Debe salir:
   `volatilidad_anual 0.463`, `peor_dia -0.140`, `capital_final 5081` y dos gráficos.
5. Déjalo abierto para la demo del Bloque 4.

> **Cambio de hoy en `clase01.R`:** ahora recorta los datos desde el 01-01-2023. Antes, si se
> subía el CSV de respaldo (que empieza en 2020), R daba 61,3 % de volatilidad y −39,5 % de peor
> día, y no coincidía con Python. Lo probé con los tres orígenes (CSV de respaldo, CSV como el de
> Colab y descarga directa) y los tres dan 46,3 % y −14,0 %.

Para la sesión 2, cuando lo harán ellos, conviene compartir el proyecto desde un espacio
(*Space*) de Posit Cloud para que cada alumno haga su copia. Hoy no hace falta: el Bloque 4 es una demo.

## 6. Mensaje a los alumnos (10 min)

Cópialo, completa los dos enlaces y envíalo por el canal del curso:

```text
¡Hola a todos! Hoy domingo 27 a las 19:00 (hora de Perú) empezamos el módulo
«Análisis cuantitativo y modelamiento de datos en criptoactivos con Python y RStudio».

Para la clase de hoy no necesitas instalar nada ni saber programar. Antes de conectarte:
1. Ten a mano tu cuenta de Google (usaremos Google Colab).
2. Crea tu cuenta gratuita en https://claude.ai y comprueba que puedes enviarle un mensaje.
3. Conéctate desde una computadora (no desde el celular), con Chrome o Edge.

Cuaderno de la clase (ábrelo y haz Archivo > Guardar una copia en Drive):
[enlace de alumno.ipynb]

Desde la sesión 2 trabajaremos también en tu computadora. Instala Python, VS Code,
R y RStudio con esta guía (30-45 min, cerca de 1 GB de descarga). Si puedes, hoy
antes de la clase; si no, antes del sábado 03-10:
[enlace de la guía de instalación]
Termina con las dos pruebas de la guía y escribe el resultado en el chat.
Si no puedes instalar (Chromebook, PC del trabajo), no pasa nada: crea tu cuenta
gratuita en https://posit.cloud y seguirás en Colab y Posit Cloud.

¡Nos vemos a las 19:00!
```

El enlace de la guía es <https://claude.ai/artifact/JYxd359gfh5nrMD1HRSR5c>. **Compártelo antes de
enviar el mensaje** (menú Compartir de la página): si no, los alumnos no pueden abrirlo. Una copia
de la guía y de las dos pruebas queda en `09_instalacion/`.

## 7. Montar el escritorio de la clase (18:00, 20 min)

- Cargador enchufado, audífonos con micrófono, internet por cable si puedes. Cierra lo que no
  uses y silencia las notificaciones.
- Abre estas pestañas, en este orden:
  1. Diapositivas, en modo Presentar.
  2. Colab: `docente.ipynb` ya ejecutado, para rescatar a quien se atasque.
  3. Colab: tu copia de `alumno.ipynb`, para enseñar el flujo en vivo.
  4. `claude.ai`.
  5. Posit Cloud, con `clase01.R` abierto.
  6. La carpeta de Drive con los CSV.
  7. El plan detallado de la clase (`05_actividades/Plan detallado de la clase (documento).url`), en otra pantalla si tienes.
- Prueba en la plataforma de la clase (Zoom o Meet) que se ve bien la pestaña de las diapositivas al compartir pantalla.
- Ten en un bloc de notas los enlaces que vas a pegar en el chat: el cuaderno, `claude.ai` y la carpeta de Drive.

## 8. Entrar a la sala (18:45)

- Entra 15 minutos antes y deja la portada en pantalla.
- Pega en el chat el enlace del cuaderno y el de `claude.ai`.
- A quien vaya llegando: que abra el cuaderno, guarde su copia y ejecute la celda Preparación.

---

## Durante la clase

| Hora | Diapositivas | Qué haces | Si algo falla |
|---|---|---|---|
| 19:00 | Portada → Prepara tu mesa | Bienvenida, regla del curso («se enseña a medir»), celda Preparación | Sin cuenta de Claude: trabaja en pareja |
| 19:10 | Bloque 1 | Demo del hash en vivo; 10 min de práctica; discusión | Si Claude recalcula los hashes, el verificador lo marca |
| 19:30 | Bloque 2 | La vela, el 451, actividades 2 y 3 | Plan B: CSV de Drive en la celda PLAN B |
| 19:55 | Pausa | Ayudar a quien no tiene `[OK]` en el Bloque 2 | Todos salen de la pausa con `df` cargado |
| 20:00 | Bloque 3 | Fórmulas, actividad 4, resultado y discusión | El verificador dice si está en %, diaria o con 252 |
| 20:25 | Bloque 4 | Demo en Posit Cloud con `clase01.R` | Posit lento: mostrar desde tu equipo o dejarlo para la sesión 2 |
| 20:35 | Bloque 5 | Actividad 5, resultado, asimetría y tesis | Más del 60 %: tercer prompt de apoyo |
| 20:50 | Cierre → Próxima sesión | Dos preguntas, tarea, rúbrica y guía de instalación (2 min): pega el enlace en el chat; quien ya instaló, escribe sus dos líneas «TODO LISTO» | — |

**Si se acaba el tiempo:** recorta primero el Bloque 4 y después la discusión del Bloque 3.
**Nunca el Bloque 5**: es la lección de la clase.

Las dos últimas diapositivas (prompts de rescate y errores típicos de la IA) son de apoyo:
déjalas a la vista en los momentos de práctica si muchos se atascan.

## Después de la clase

1. Comparte `docente.ipynb` (súbelo a Drive con enlace de lector).
2. Publica la tarea con la fecha y el medio de entrega. La rúbrica está en
   `05_actividades/actividades_para_alumnos.md`.
3. Anota asistencia y participación: alimentan el registro de notas del módulo.
4. Si cambiaste las diapositivas, vuelve a exportar el PDF a `01_diapositivas/`.

## Pendientes del módulo (no son para hoy)

- Examen de opción múltiple del módulo, con respuestas y solucionario.
- Registro de notas y material bibliográfico para Escuela Global.
- Sesión 2 (sábado 03-10, 19:00): los alumnos trabajan en Posit Cloud y ven el control de
  volatilidad. Material de partida en `06_laboratorio_sesiones_2_a_4/`.
- Sesión 2, primeros 15 minutos: resolver las instalaciones atascadas. Quien no lo consiga sigue
  en Posit Cloud. La tabla «Si algo falla» de la guía cubre los casos habituales.
- Claude Code (sesión 3 o 4): necesita un plan de pago de Claude. Mejor como demo del docente, o
  avisar a los alumnos con tiempo.
