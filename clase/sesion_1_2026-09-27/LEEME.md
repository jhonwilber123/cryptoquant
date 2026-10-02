# Sesión 1 · Análisis cuantitativo y modelamiento de datos en criptoactivos

Todo el material de la clase del domingo 27-09-2026 (19:00–21:00) en una sola carpeta.
Empieza por **`00_COMO_PREPARARME_HOY.md`**.

| Carpeta o archivo | Qué es | Para quién | Cuándo se comparte |
|---|---|---|---|
| `00_COMO_PREPARARME_HOY.md` | La preparación paso a paso, el guion de la clase y qué hacer después | Docente | No se comparte |
| `01_diapositivas/` | Acceso directo a las diapositivas (31, con notas del orador). Guarda aquí el PDF y el PPTX al exportarlos | Docente | El PDF, al final si quieres |
| `02_cuadernos_colab/alumno.ipynb` | Cuaderno de trabajo: cinco bloques con prompt, celda para pegar y verificador | Alumnos, en Colab | Antes o al inicio de la clase |
| `02_cuadernos_colab/docente.ipynb` | El mismo cuaderno con la solución de referencia en cada celda | Docente | Al final de la clase, nunca antes |
| `03_posit_cloud_R/clase01.R` | Bloque 4: el CSV leído en R con dplyr y ggplot2 | Docente (demo); alumnos en la sesión 2 | En Posit Cloud |
| `04_datos_respaldo/` | CSV de BTC, ETH y SOL (diarios desde 2020, horarios desde 2025) con huellas SHA-256 | Plan B, vía Drive | Carpeta de Drive en modo lector |
| `05_actividades/actividades_para_alumnos.md` | Las cinco actividades con sus prompts, la tarea y la rúbrica | Alumnos | Cuando quieras |
| `05_actividades/respuestas_y_criterios_docente.md` | Respuestas esperadas, cifras de referencia y criterios de corrección | Docente | No se comparte |
| `05_actividades/Plan detallado de la clase (documento).url` | El plan completo (agenda, prompts, plan B) en un documento de Claude | Docente | — |
| `06_laboratorio_sesiones_2_a_4/` | Los nueve pasos del laboratorio en Python y R, para las próximas sesiones | Docente | Más adelante |
| `07_manual_de_usuario.md` | Cómo se usa cada pieza: el cuaderno, Posit Cloud, el programa `cryptoquant` y el piloto publicado en internet. Es la fuente del Word | Alumnos y docente | Opcional |
| `07_manual_de_usuario.docx` y `.pdf` | El mismo manual con capturas, para imprimir o compartir. Se rehace con `07_manual/construir_word.py`; las capturas, con `07_manual/capturar.py` | Alumnos y docente | Opcional |
| `08_arquitectura_del_programa.md` | Cómo está construido `cryptoquant`, módulo por módulo | Docente y lectores técnicos | Opcional |
| `09_instalacion/` | Guía para instalar Python, VS Code, R y RStudio (copia de <https://claude.ai/artifact/JYxd359gfh5nrMD1HRSR5c>) y sus dos pruebas, `prueba_python.py` y `prueba_r.R` | Alumnos | Hoy, con el mensaje de la clase |

## De dónde sale cada archivo

Los cuadernos, el script de R, los datos y el laboratorio son **copias** del 27-09-2026. Las
fuentes originales siguen en el repositorio:

| Copia | Original | Se regenera con |
|---|---|---|
| `02_cuadernos_colab/*.ipynb` | `clase/01_primera_clase/` | `python clase/01_primera_clase/generar_cuadernos.py` |
| `03_posit_cloud_R/clase01.R` | `clase/01_primera_clase/posit/clase01.R` | se edita a mano |
| `04_datos_respaldo/` | `clase/01_primera_clase/datos/` | `python -m cryptoquant paquete --destino clase/sesion_1_2026-09-27/04_datos_respaldo` |
| `06_laboratorio_sesiones_2_a_4/` | `laboratorio/` | se edita a mano |

Si cambias un original, vuelve a copiarlo aquí. Los cuadernos no se editan a mano: se cambia
`generar_cuadernos.py` y se vuelve a ejecutar; `tests/test_clase.py` comprueba que estén al día.

## Los CSV no van a git

`.gitignore` excluye `clase/**/*.csv`: los datos se reparten por Drive y Posit Cloud, no por el
repositorio. El resto de la carpeta todavía no está en ningún commit.
