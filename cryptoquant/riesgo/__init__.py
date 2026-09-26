"""Medicion del riesgo: cuanto se puede perder, y si esa cifra es de fiar.

    var         VaR y CVaR a un dia por cuatro metodos, a partir de posiciones
    pruebas     backtest del VaR: Kupiec, Christoffersen y cola del CVaR
    futuro      Monte Carlo por bloques: caidas y perdidas a N dias
    cartera     el riesgo de una cartera concreta (la suya)
    evidencia   filtro que toda senal nueva debe pasar antes de proponerse
    informe     modelo frente a inversion pasiva; CSV y estado para el panel

Nada de esto decide ni modifica posiciones: lee las del backtest y las mide.
El motor de decision y el diario sellado del forward test no se tocan.
"""
