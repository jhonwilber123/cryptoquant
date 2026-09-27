# Clase 1 · Blockchain y criptoactivos con IA

Dos horas, alumnos que nunca programaron, Claude como asistente. El plan
detallado (agenda, prompts, plan B) está en el documento compartido de la
clase; aquí están los materiales.

| Archivo | Para quién | Qué es |
|---|---|---|
| `alumno.ipynb` | alumnos, en Colab | cinco bloques: prompt para Claude, celda para pegar el código y un verificador |
| `docente.ipynb` | docente | lo mismo con la solución de referencia en cada celda; ejecutado entero, todo da `[OK]` |
| `posit/clase01.R` | docente, en Posit Cloud | bloque 4: el CSV de Colab leído en R con dplyr y ggplot2 |
| `datos/` | Drive y Posit Cloud | CSV de respaldo (no están en git); se generan con el comando de abajo |
| `generar_cuadernos.py` | quien edite la clase | fuente única de los dos cuadernos |

## Antes de la clase

```bash
python -m cryptoquant paquete        # CSV de respaldo en clase/01_primera_clase/datos/
```

Sube `datos/` a Google Drive y al proyecto de Posit Cloud, abre `alumno.ipynb`
en Colab (Archivo → Subir cuaderno) y ejecuta la celda de preparación.

## Por qué data-api.binance.vision

Colab y Posit Cloud corren en servidores de EE. UU., y `api.binance.com`
rechaza esas conexiones (HTTP 451). `data-api.binance.vision` sirve los mismos
datos públicos de mercado sin clave y sin esa restricción. Todos los prompts y
las soluciones lo usan.

## Editar la clase

Cambia `generar_cuadernos.py` y vuelve a ejecutarlo; no edites los `.ipynb` a
mano. `tests/test_clase.py` comprueba que los cuadernos estén al día con el
generador y que los verificadores detecten los errores típicos de la IA.
