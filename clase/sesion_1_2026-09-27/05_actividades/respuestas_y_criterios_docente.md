# Sesión 1 · Respuestas esperadas y criterios de corrección

Solo para el docente. Las cifras salen de los datos de Binance hasta el 26-09-2026; las de los
alumnos pueden variar en los decimales, porque cada día suma una vela.

## Cifras de referencia

| Actividad | Verificador | Resultado de referencia |
|---|---|---|
| 1 · Blockchain | `verificar.cadena` | `valida_antes = True`, `valida_despues = False` |
| 2 · Velas | `verificar.velas` | 1 365 velas (1 366 después de las 19:00 del 27-09); cierre del 01-01-2023: 16 616,75 |
| 3 · CSV | `verificar.csv` | `BTCUSDT_1d.csv` con las mismas filas que `df` |
| 4 · Riesgo BTC | `verificar.riesgo` | volatilidad anual 0,463; peor día −0,140 (05-02-2026); 3 días de más de 4σ frente a 0,09 esperados; 21 de más de 3σ frente a 3,7 |
| 5 · Predicción | `verificar.prediccion` | modelo 0,493; siempre «sube» 0,488 (404 días de prueba) |
| Tarea · Riesgo ETH | `verificar.riesgo` | volatilidad anual 0,627; peor día −0,150 |
| Bloque 4 · R | `clase01.R` | volatilidad 0,463; peor día −0,140; 1 000 dólares de enero de 2023 → 5 081 |

Las huellas SHA-256 de la demo del Bloque 1:

```text
Ana paga 2 BTC  a8580df685158be73687c9673ebec535ad61deddd6a5abce6a5d25b7608964fe
Ana paga 3 BTC  5e3a00498f7ffb8fccc02453b9ee4fbc4e9c39f7a1095e0814031eaf29c39410
```

## Respuestas a las preguntas de discusión

**Bloque 1 · ¿Por qué cambiar el bloque 1 rompe el bloque 2?**
Porque el bloque 2 guarda la huella del bloque 1 tal como era. Al cambiar sus datos, la huella
del bloque 1 cambia y deja de coincidir con la que guardó el bloque 2.

**Bloque 1 · ¿Qué tendría que rehacer un falsificador?**
Recalcular la huella del bloque alterado y la de todos los siguientes. En Bitcoin cada bloque
exige una prueba de trabajo, así que rehacer la cadena y alcanzar a la red exige más cómputo que
el de todo el resto de la red junto.

**Bloque 3 · ¿Qué le pasa a quien pone un stop «a 3 sigmas»?**
Salta mucho más de lo que promete la normal: 21 días de más de 3 desviaciones frente a 3,7
esperados, unas seis veces más. Con 4 desviaciones, la normal promete un día cada unos 43 años y
en BTC hubo 3 en menos de 4. Puente con la tesis: el VaR normal se rechazó en todas las carteras
y el GARCH con colas t se aceptó al 95 %.

**Cierre · Si nadie predice el precio de mañana, ¿qué sí se puede hacer?**
Medir y controlar el riesgo: la volatilidad se agrupa en rachas y se puede prever, así que sirve
para decidir cuánto invertir, dónde poner el stop y qué tamaño dar a cada posición. Es el control
de volatilidad de la sesión 2.

## Errores típicos de la IA

| Error | Cómo se nota | Lo detecta | Qué pedirle a Claude |
|---|---|---|---|
| Usa `api.binance.com` | La celda falla con error 451 | La propia celda | Usar `data-api.binance.vision` |
| Fecha las velas con la hora de cierre | Las velas empiezan a las 23:59 | `verificar.velas` | La fecha es la hora de apertura |
| Toma la columna equivocada | El cierre del 01-01-2023 no es 16 616,75 | `verificar.velas` | El cierre es el quinto campo de Binance |
| Lee la marca de tiempo sin `unit="ms"` | Fechas de 1970 | `verificar.velas` | Convertir con `unit="ms"` |
| Incluye la vela de hoy | El último cierre cambia al repetir | `verificar.velas` | Quitar la vela que no ha cerrado |
| Deja los precios como texto | Los cálculos fallan | `verificar.velas` | Convertir con `astype(float)` |
| Recalcula los hashes al falsificar | La cadena sigue siendo válida | `verificar.cadena` | Cambiar el texto sin recalcular nada |
| Volatilidad en %, diaria o con 252 días | La cifra no cuadra | `verificar.riesgo`, que dice cuál | Fracción, anualizada, 365 días |
| Mezcla fechas futuras al entrenar | Exactitud por encima del 60 % | `verificar.prediccion` | Dividir por fecha, sin mezclar |
| Usa otro nombre de variable, o se ejecuta el verificador antes que el código | «Python no conoce df» | La celda de preparación | Guardar el resultado con el nombre que pide el prompt |

Cada uno de estos errores tiene un test en `tests/test_clase.py` que comprueba que el verificador
lo detecta y que el trabajo correcto pasa.

## Criterios de corrección de la tarea

| Criterio | Puntos | Cómo se comprueba |
|---|---|---|
| Los cinco verificadores de clase en `[OK]` | 40 | 8 puntos por verificador visible en `[OK]` en el cuaderno entregado |
| Tarea con ETH: `verificar.csv` y `verificar.riesgo` en `[OK]` | 30 | 15 puntos cada uno, sobre ETHUSDT |
| Párrafo: qué se corrigió de Claude y cómo se detectó | 20 | Nombra un error concreto y el mensaje o la cifra que lo delató |
| Respuesta: qué se puede hacer si no se puede predecir | 10 | Menciona medir o controlar el riesgo, la volatilidad o el tamaño de la posición |

> Con ETHUSDT, `verificar.velas` compara el cierre del 01-01-2023 con el de BTC (16 616,75), marca
> `[REVISAR]` y se detiene ahí. No es un error del alumno: por eso la tarea se corrige con
> `verificar.csv` y `verificar.riesgo`. La diapositiva «Tarea» y la hoja de actividades ya lo dicen.
