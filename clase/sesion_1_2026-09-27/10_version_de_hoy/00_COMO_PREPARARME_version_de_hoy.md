# Cómo prepararme · sesión 1, versión de hoy (diagnóstico, herramientas e instalación)

Esta es la versión que se da hoy, domingo 27-09-2026, de 19:00 a 21:00. La versión anterior (cinco
bloques de práctica en Colab) sigue intacta en la carpeta de arriba, lista para la sesión 2.

- **Diapositivas de hoy:** <https://claude.ai/artifact/HgMJbvg6RpBM7rnh85oEK4> (22, con notas del orador).
- **Editor:** Antigravity IDE (gratis, con límites semanales), en lugar de VS Code. Es el botón
  «Antigravity IDE (Standalone)», no la app principal «Antigravity» de la misma página.
- **Idea de la clase:** te cansas poco. Solo dos momentos son de los alumnos: el test y la instalación.

Tiempo de preparación: unos 60 minutos.

| # | Paso | Tiempo | Hecho |
|---|---|---|---|
| 1 | Crear el test en Google Forms | 15 min | ☐ |
| 2 | Compartir la guía de instalación y enviar el mensaje | 5 min | ☐ |
| 3 | Probar Antigravity IDE con la prueba de Python (mejor en la laptop limpia: `ensayo_en_la_laptop.md`) | 15 min | ☐ |
| 4 | Probar la demo del piloto con la cartera inventada | 5 min | ☐ |
| 5 | Repasar las diapositivas y poner el enlace del test | 10 min | ☐ |
| 6 | 18:30 · Montar el escritorio | 10 min | ☐ |
| 7 | 18:45 · Entrar a la sala | — | ☐ |

---

## 1. Crear el test (15 min)

Sigue `test_diagnostico.md`: 15 preguntas en 4 secciones, con el texto listo para copiar. Al final,
**Enviar › enlace › Acortar URL** y guarda el enlace en un bloc de notas.

## 2. Compartir la guía y enviar el mensaje (5 min)

1. Abre la guía de instalación, <https://claude.ai/artifact/JYxd359gfh5nrMD1HRSR5c>, y compártela
   desde su menú **Compartir**. Si no, los alumnos no pueden abrirla.
2. Envía este mensaje por el canal del curso (completa el enlace):

```text
¡Hola a todos! Hoy domingo 27 a las 19:00 (hora de Perú) empezamos el módulo
«Análisis cuantitativo y modelamiento de datos en criptoactivos con Python y RStudio».

Hoy no necesitas saber programar: vamos a conocernos, ver las herramientas
y dejar tu computadora lista.

Antes de conectarte:
1. Usa una computadora (Windows 10/11 o Mac con macOS 12 o posterior), no el celular.
2. Crea tu cuenta gratuita en https://claude.ai
3. Si tienes tiempo, empieza a instalar con esta guía:
   [enlace de la guía]
   En el paso de VS Code, instala Antigravity IDE en su lugar:
   https://antigravity.google/download  >  sección «Antigravity for IDEs»
   >  botón «Antigravity IDE (Standalone)» (no la app principal «Antigravity»)
   Si no terminas, lo completamos juntos en la clase.

¡Nos vemos a las 19:00!
```

## 3. Probar Antigravity IDE en tu PC (15 min)

**Mejor en la laptop sin nada instalado:** sigue `ensayo_en_la_laptop.md`. Si lo haces allí, este paso
ya está hecho.

Antigravity IDE ya está instalado en este equipo. La guía de instalación está escrita para VS Code, así que
ensaya tú el recorrido que harán los alumnos con Antigravity:

1. La descarga correcta está en <https://antigravity.google/download>, sección **«Antigravity for
   IDEs»**, botón **«Antigravity IDE (Standalone)»** (versión 2.5.5). La app principal «Antigravity»,
   más arriba en la misma página, **no** es la que usamos: dilo en voz alta en clase.
2. Anota si Antigravity **pide iniciar sesión** (y con qué cuenta). No lo pude confirmar desde aquí.
3. Instala las extensiones **Python** y **Jupyter** si no las tienes.
4. **Archivo › Nuevo archivo › Jupyter Notebook**, pega la prueba de Python de la guía (botón Copiar)
   y ejecútala. En «Select Kernel», elige Python Environments y tu Python 3.
5. Debe terminar en `TODO LISTO` con un gráfico de BTC. Al final dice «Python y VS Code OK» porque
   la guía se escribió para VS Code: en clase pedirás que escriban **Python OK**.
6. En RStudio, pega la prueba de R en la consola: debe terminar en `TODO LISTO`.

Si algo de esto se comporta distinto de lo que dicen las diapositivas «Tres tropiezos» y
«Lo que ves y qué hacer», anótalo y lo explicas de palabra en clase.

## 4. Probar la demo del piloto (5 min)

Doble clic en **`abrir_demo_piloto.cmd`** (en esta carpeta). Abre el piloto en el navegador con la
cartera inventada de `piloto_demo/` (0,05 BTC, 1,2 ETH, 10 SOL, 500 USDT). **Tu cartera real de
`data\piloto\` no se toca ni se ve.** Recorre las cuatro pestañas y déjalo abierto para la clase.

Lo probé a las 17:00: con esa cartera recomienda pasar de 94,6 % a 30,8 % en cripto, con una
volatilidad prevista del 48,7 %. A la hora de la clase las cifras se moverán un poco.

## 5. Diapositivas (10 min)

- En la diapositiva **«Cuéntame cómo llegas»** (la del test), cambia `[enlace del formulario]` por el
  enlace corto de Forms. O déjalo: dice «Abre el enlace del chat».
- Recorre las 22 con sus notas. Para presentar, usa el modo **Presentar**.
- El enlace es privado: si quieres compartirlas después, hazlo desde **Compartir**.

## 6. Montar el escritorio (18:30)

Abre, en este orden:

1. Diapositivas de hoy, en modo Presentar.
2. Google Forms: **Respuestas › Resumen** del test.
3. El piloto de demostración, ya abierto.
4. Antigravity IDE, con tu cuaderno de prueba de Python.
5. RStudio.
6. La guía de instalación.

En un bloc de notas, los tres enlaces para el chat: el test, la guía y <https://antigravity.google/download>
(con la aclaración: sección «Antigravity for IDEs», botón «Antigravity IDE (Standalone)»).

## 7. Entrar a la sala (18:45)

Entra 15 minutos antes, deja la portada en pantalla y pega en el chat el enlace del test: los que
llegan temprano ya pueden ir respondiendo.

---

## Durante la clase

| Hora | Diapositivas | Qué haces |
|---|---|---|
| 19:00 | Portada → Hoy preparamos el terreno → Agenda | Bienvenida y la regla: se aprende a medir, no a apostar |
| 19:10 | Cuéntame cómo llegas | Test: pega el enlace, 10 minutos, avisa cuando quede 1 |
| 19:25 | Lo que dice el grupo | Cambias a Forms y comentas tres gráficos: programación, aspiraciones y equipo |
| 19:30 | Herramientas → Cuatro pasos → Recorrido | Qué es cada herramienta y cómo trabajaremos con la IA |
| 19:50 | A dónde vamos a llegar → Piloto → Tesis | Demo del piloto con la cartera inventada |
| 20:10 | Pausa | Pega el enlace de la guía: quien no instaló, empieza a descargar RStudio |
| 20:15 | Instalación (5 diapositivas) | Muestras los tres tropiezos en tu pantalla; ellos instalan y prueban; reportan «Python OK» y «R OK» en el chat |
| 20:45 | Para llevarse (2) → Tarea → Próxima sesión | Las dos ideas del curso en 5 minutos y la tarea |

**Si el tiempo aprieta:** acorta la demo (19:50), nunca la instalación.

**Regla para no saturarte en la instalación:** los tres tropiezos comunes los muestras una vez para
todos; los problemas individuales se anotan y se resuelven en los primeros 15 minutos de la sesión 2
o por el grupo del curso.

## Después de la clase

1. En Forms, exporta las respuestas a una hoja de cálculo: también sirven de registro de asistencia.
2. Con las reglas de `test_diagnostico.md` decide el ritmo y cuéntaselo al grupo.
3. Haz la lista de quienes no reportaron «Python OK» y «R OK», y de quienes seguirán en Colab y Posit Cloud.
4. Para la sesión 2, el material de práctica ya está listo en `02_cuadernos_colab/` y
   `03_posit_cloud_R/`, en la carpeta de arriba.
