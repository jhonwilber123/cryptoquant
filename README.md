# cryptoquant · Piloto de riesgo

Una app para el navegador que responde a una pregunta: **con lo que tengo hoy,
¿cuánto debería tener en cripto para que mi cartera no se mueva más de lo que
acepto?**

Solo recomienda. No se conecta a ninguna cuenta, no pide claves de API y no
envía órdenes. Puedes probarla con una cartera inventada.

## 1. Descargar

En esta página, botón verde **Code** › **Download ZIP**. Descomprime el ZIP:
queda una carpeta `cryptoquant-main`. Ábrela en tu editor (en Antigravity,
**File › Open Folder**).

Si ya usas git:

```
git clone https://github.com/jhonwilber123/cryptoquant.git
```

## 2. Instalar lo que necesita

Hace falta Python 3.14, el de la guía de instalación. Abre una terminal en la
carpeta (en Antigravity, **Terminal › New Terminal**) y escribe:

```
python -m pip install -r requirements.txt
```

La primera vez tarda unos minutos. En Mac, escribe `python3` en lugar de
`python`, aquí y en los pasos siguientes.

Si la terminal no encuentra `python`, faltó marcar «Add python.exe to PATH» al
instalarlo: repite ese paso de la guía.

## 3. Abrir la app

```
python -m cryptoquant piloto
```

En Windows también vale doble clic en `scripts\piloto.cmd`. El navegador se
abre en <http://127.0.0.1:8501>. Para cerrarla, pulsa Ctrl+C en la terminal o
cierra la ventana negra.

Para probarla, anota en la barra lateral una cartera inventada, una fila por
moneda (`BTC 0,01`, `ETH 0,2`, `USDT 300`), elige cuánto movimiento aguantas y
pulsa **Guardar cartera y ajustes**.

## 4. Revisar el código

La app está en `cryptoquant/piloto/` y aplica una sola regla:

```
parte en cripto = mínimo(máximo, objetivo ÷ volatilidad prevista) × freno
```

| Dónde | Qué mirar |
|---|---|
| [`cryptoquant/piloto/README.md`](cryptoquant/piloto/README.md) | Qué muestra cada pestaña, la regla paso a paso y sus límites |
| [`cryptoquant/piloto/motor.py`](cryptoquant/piloto/motor.py) | La decisión: cuánto en cripto y qué comprar o vender, sin red ni interfaz |
| [`cryptoquant/piloto/app.py`](cryptoquant/piloto/app.py) | La interfaz, hecha con Streamlit |
| [`cryptoquant/laboratorio/control.py`](cryptoquant/laboratorio/control.py) | La volatilidad prevista para mañana (EWMA) |
| [`cryptoquant/portfolio/sizing.py`](cryptoquant/portfolio/sizing.py) | El freno: invertir menos cuando la cartera cae |
| [`cryptoquant/riesgo/`](cryptoquant/riesgo/) | Cuánto se puede perder: VaR por cuatro métodos y escenarios a futuro |
| [`tests/test_piloto.py`](tests/test_piloto.py) | Lo que nunca debe pasar: vender más de lo que hay, comprar sin efectivo o pasar del máximo |

El resto de `cryptoquant/` es el sistema del que la app toma sus cálculos. No
hace falta para usarla.

Para correr las pruebas (unos 2 minutos):

```
python -m pip install pytest
python -m pytest tests -q
```

Debe terminar en «passed», sin ningún «failed». Los avisos («warnings») no son
errores.

## Tus datos

Lo que anotes se guarda en `data/piloto/`, en tu equipo, y no sale de él. La
app solo consulta precios públicos de Binance (`data-api.binance.vision`).

## Aviso

Es software de análisis estadístico, no asesoramiento financiero. Sus cifras
son estimaciones de un modelo, con error. El rendimiento pasado no predice el
futuro y en cripto se puede perder todo lo invertido. Invierte solo lo que
puedas permitirte perder.
