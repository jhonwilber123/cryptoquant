"""Laboratorio: el camino de las velas al modelo, paso a paso.

El resto de `cryptoquant` es un sistema de produccion: decide, registra y se
audita a si mismo. Este subpaquete es otra cosa. Recorre, una a una y a la
vista, las piezas con las que se construye un sistema asi, para poder
ensenarlas, discutirlas y reproducirlas en R.

    datos          velas de Binance, grafico de velas, exportacion a CSV
    variables      indicadores como columnas, rezagos y la variable objetivo
    cuantitativo   colas gruesas frente a la normal, estadistica de senales
    series         por que retornos y no precios, autocorrelacion, ARIMA
    volatilidad    agrupamiento, GARCH(1,1), stop y tamano de posicion
    aprendizaje    random forest con validacion temporal y linea base
    montecarlo     trayectorias de precio y simulacion de la cuenta
    informe        ejecuta todo lo anterior y deja CSV y graficos

La version en R vive en `laboratorio/r/` y parte del CSV que exporta `datos`.
"""
