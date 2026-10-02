# Test diagnóstico · sesión 1

Para crear en Google Forms. Son 15 preguntas en 4 secciones; se responde en unos 8 minutos.
**No cuenta para la nota.** Sirve para decidir el ritmo del curso y quién necesita ayuda con la instalación.

## Cómo crearlo rápido (15 min)

1. Abre <https://forms.new> con tu cuenta de Google.
2. Título: `Sesión 1 · ¿Cómo llegas al curso?`. Descripción: el texto de abajo.
3. Crea las 4 secciones con el botón **Agregar sección** (el icono de dos rectángulos) y copia las preguntas.
   Marca como **obligatorias** todas menos la 12.
4. **Configuración**: no hace falta pedir inicio de sesión ni recopilar correos (la pregunta 1 pide el nombre).
5. **Enviar › icono de enlace › Acortar URL › Copiar**. Ese enlace va al chat y a la diapositiva «Test».
6. Durante la clase, los resultados están en la pestaña **Respuestas › Resumen**, con gráficos automáticos.

**Descripción del formulario:**

> Este test no cuenta para la nota. Responde con sinceridad: no hay respuestas malas.
> Con tus respuestas decido el ritmo del curso y en qué temas profundizamos.

---

## Sección 1 · Tú y tu nivel

**1. Nombre y apellidos** · respuesta corta

**2. ¿Has programado alguna vez?** · varias opciones (una sola)
- Nunca
- Un poco: tutoriales o fórmulas avanzadas de Excel
- Sí, en algún lenguaje
- Programo con frecuencia

**3. ¿Qué has usado alguna vez?** · casillas (varias)
- Excel o Google Sheets
- Python
- R
- SQL
- Otro lenguaje de programación
- Ninguno de estos

**4. ¿Cómo te llevas con la estadística?** · varias opciones
- No recuerdo casi nada
- Sé qué es un promedio y una desviación estándar
- He usado regresiones o pruebas de hipótesis
- Trabajo con modelos estadísticos

**5. ¿Has usado una IA como Claude, ChatGPT o Gemini?** · varias opciones
- Nunca
- Para preguntas generales
- Para escribir o revisar textos
- Para escribir código

**6. ¿Qué experiencia tienes con criptomonedas?** · varias opciones
- Ninguna
- Las conozco, pero no he invertido
- He comprado o vendido alguna vez
- Hago trading con frecuencia

## Sección 2 · Tres preguntas rápidas

Descripción de la sección: *No cuentan para nada. Si no sabes, marca «No lo sé»: también me sirve.*

**7. En una vela diaria de Binance, el «cierre» es…** · varias opciones
- El precio más alto del día
- El precio al terminar el día
- El promedio del día
- No lo sé

**8. BTC tiene una volatilidad anual del 50 % y una acción, del 15 %. Eso quiere decir que…** · varias opciones
- BTC sube más que la acción
- El precio de BTC se mueve mucho más que el de la acción
- BTC es más rentable que la acción
- No lo sé

**9. Un modelo acierta el 51 % de las veces si BTC sube o baja al día siguiente. ¿Qué harías?** · varias opciones
- Invertir con él: acierta más de la mitad de las veces
- Compararlo antes con una estrategia simple, como decir siempre «sube»
- Descartarlo porque acierta poco
- No lo sé

## Sección 3 · Lo que buscas

**10. ¿Qué quieres lograr con este módulo? Marca hasta dos.** · casillas
- Entender cómo se analizan los datos de cripto
- Construir mis propias herramientas
- Gestionar mejor el riesgo de mis inversiones
- Automatizar mi trading
- Aprender a programar con ayuda de la IA
- Aplicarlo en mi trabajo o investigación

*(En Forms: tres puntos de la pregunta › Validación de respuesta › Seleccionar como máximo › 2.)*

**11. ¿Qué ritmo prefieres?** · varias opciones
- Paso a paso, aunque avancemos menos
- Equilibrado
- Rápido, aunque tenga que practicar por mi cuenta

**12. ¿Qué te gustaría poder hacer al terminar el curso?** · párrafo (opcional)

## Sección 4 · Tu equipo

**13. ¿Qué computadora usarás en el curso?** · varias opciones
- Windows
- Mac
- Chromebook
- Linux
- Solo celular o tablet

**14. ¿Puedes instalar programas en ella?** · varias opciones
- Sí, es mía
- No estoy seguro
- No: es del trabajo o tiene restricciones

**15. ¿Qué tienes ya listo?** · casillas (varias)
- Python
- Antigravity IDE
- R
- RStudio
- Cuenta de claude.ai
- Todavía nada

---

# Solo para el docente: cómo leer los resultados

## Respuestas de las preguntas rápidas

| Pregunta | Respuesta correcta | Qué te dice si fallan muchos |
|---|---|---|
| 7 · el cierre | El precio al terminar el día | Hay que explicar qué es una vela antes de descargar datos |
| 8 · la volatilidad | El precio de BTC se mueve mucho más | Hay que dedicar tiempo a rendimiento y volatilidad |
| 9 · el modelo del 51 % | Compararlo antes con una estrategia simple | Es la lección central del curso: vale la pena volver a ella |

## Decidir el ritmo con la pregunta 2

| Si… | Ritmo | Qué significa para las sesiones 2 a 4 |
|---|---|---|
| 60 % o más marcan «Nunca» o «Un poco» | Paso a paso | Sesión 2 con el cuaderno de Colab ya preparado (bloques con prompt y verificador) y R como demo guiada. GARCH, random forest y Monte Carlo como demos del docente |
| Entre 30 % y 60 % | Equilibrado | Sesión 2: datos y riesgo. Sesión 3: volatilidad y «¿predice la IA?». Sesión 4: control de volatilidad |
| Menos del 30 % | Rápido | Desde la sesión 2, el laboratorio completo en Python y R, con los alumnos ejecutando cada paso |

La pregunta 11 afina: si el grupo pide «paso a paso» aunque sepa algo, baja un escalón.

## Qué enfatizar según la pregunta 10

| Lo más marcado | Énfasis |
|---|---|
| Gestionar mejor el riesgo | Control de volatilidad y la aplicación (piloto) |
| Automatizar mi trading | Explicar pronto que el curso mide y no envía órdenes; mostrar por qué predecir la dirección no funciona |
| Aprender a programar con IA | Más actividades de prompt y verificador |
| Construir mis propias herramientas | Más tiempo con el laboratorio y la estructura del programa |
| Aplicarlo en el trabajo o investigación | R, econometría y cómo reportar resultados |

## Instalación: a quién hay que atender

- **Preguntas 13 y 14:** quien use Chromebook, celular o tableta, o no pueda instalar, sigue en Google Colab y Posit Cloud. Avísale en privado.
- **Pregunta 15, «Todavía nada»:** es la lista para el recordatorio de mañana y para los 15 minutos del inicio de la sesión 2.
- **Pregunta 5, «Nunca»:** puede necesitar ayuda para crear su cuenta de claude.ai.

Exporta las respuestas (Respuestas › icono de hoja de cálculo): sirven también como registro de asistencia de la sesión 1.
