# %% [markdown]
# # Unidad 2 · Procesamiento y transformación de datos
#
# **Qué vas a aprender**
# - Pasar de datos "crudos" a una tabla lista para analizar: elegir, filtrar, ordenar y crear columnas.
# - Agrupar y resumir: por moneda, por año, por día de la semana.
# - Cambiar entre formato **largo** y **ancho**, unir tablas y decidir qué hacer con los datos que faltan.
# - Dejar preparado el conjunto de datos que usan los modelos de las unidades siguientes.
#
# El temario hace esta unidad en R con `dplyr`. Este cuaderno hace **lo mismo con pandas**, y el cuaderno
# de R de la unidad, lo mismo con `dplyr`: compáralos lado a lado. Las operaciones tienen nombres
# distintos, pero son las mismas ideas:
#
# | Tarea | R (`dplyr`, `tidyr`) | Python (`pandas`) |
# |---|---|---|
# | leer un CSV | `read_csv()` | `pd.read_csv()` |
# | elegir columnas | `select()` | `df[["a", "b"]]` |
# | filtrar filas | `filter()` | `df[condición]` o `df.query()` |
# | ordenar | `arrange()` | `df.sort_values()` |
# | crear una columna | `mutate()` | `df.assign()` o `df["x"] = ...` |
# | agrupar y resumir | `group_by() + summarise()` | `df.groupby().agg()` |
# | ancho ↔ largo | `pivot_wider()` / `pivot_longer()` | `df.pivot()` / `df.melt()` |
# | unir tablas | `inner_join()` / `full_join()` | `pd.merge()` / `df.join()` |
# | el valor de ayer | `lag()` | `df.shift()` |
#
# **Qué tiene que ver con la app.** La pantalla **Mi diario** del piloto valora tu cartera día a día: une
# lo que tienes con los precios de cada día. Al final de este cuaderno la construyes tú.

# %% preparacion

# %% [markdown]
# ## 1. Importar y juntar en formato largo
#
# `velas()` (en la celda de preparación) lee cada moneda del CSV o la descarga. Para trabajar con las
# tres a la vez, se apilan en **formato largo**: una fila por día **y moneda**, con una columna que dice
# de qué moneda es cada fila. Es el formato que prefieren `dplyr`, `ggplot2` y `groupby`.

# %%
largo = pd.concat([velas(a).reset_index().assign(moneda=a) for a in ACTIVOS], ignore_index=True)
largo = largo[["fecha", "moneda", "open", "high", "low", "close", "volume"]]
largo.info()
largo.head()

# %% [markdown]
# ## 2. Crear columnas: el rendimiento diario, y un error que hay que ver una vez
#
# El rendimiento simple de un día es `cierre de hoy / cierre de ayer − 1`. En pandas, `pct_change()`.
#
# Cuidado: en formato largo, la fila anterior a la primera de SOL es la **última de ETH**. Si se calcula
# sobre toda la tabla, el primer "rendimiento" de SOL compara el precio de SOL con el de ETH. Hay que
# calcularlo **por moneda** (`groupby`). Mira la diferencia:

# %%
mal = largo["close"].pct_change()
bien = largo.groupby("moneda")["close"].pct_change()
primera_sol = largo.index[largo["moneda"] == "SOL"][0]
print(f"Primer día de SOL, calculado sobre toda la tabla: {mal[primera_sol]:.2%}")
print(f"Primer día de SOL, calculado por moneda:          {bien[primera_sol]}")
largo["ret"] = bien

# %% [markdown]
# El primero es un desplome del 99 % que nunca existió: SOL valía unos 3 dólares y ETH, unos 380.
# Un solo dato así arruina una volatilidad o un "peor día". El segundo es `NaN`, que es lo correcto:
# el primer día no tiene día anterior.

# %% [markdown]
# ## 3. Filtrar y ordenar: los peores días
#
# ¿Qué días cayó alguna moneda más de un 25 %? Ordenados del peor al menos malo. Hay fechas que
# conviene reconocer: el 12-03-2020 (el pánico del COVID), el 19-05-2021 (China contra la minería y el
# comercio de criptos) y el 09-11-2022 (la quiebra de FTX, con SOL en el centro).

# %%
peores = largo[largo["ret"] < -0.25].sort_values("ret")
peores[["fecha", "moneda", "close", "ret"]].assign(ret=lambda d: d["ret"].map("{:.1%}".format))

# %% [markdown]
# ## 4. Agrupar y resumir: cada moneda, cada año
#
# `groupby` parte la tabla en grupos y `agg` resume cada uno. Por moneda y año:
# - **rentabilidad** del año: se encadenan los rendimientos diarios, `(1 + r).prod() − 1`;
# - **volatilidad anual**: la desviación típica diaria por la raíz de 365 (Unidad 6);
# - **peor día** y porcentaje de días en que subió.
#
# 2020 empieza en agosto para SOL, y 2026 acaba el 26 de septiembre: son años incompletos.

# %%
largo["año"] = largo["fecha"].dt.year
anual = largo.groupby(["moneda", "año"])["ret"].agg(
    rentabilidad=lambda r: (1 + r).prod() - 1,
    volatilidad=lambda r: r.std() * np.sqrt(ANUAL),
    peor_dia="min",
    dias_sube=lambda r: (r.dropna() > 0).mean(),
)
anual.map("{:.0%}".format)

# %% [markdown]
# Lo que dice la tabla:
# - La **volatilidad** anual va del 42 % (BTC en 2025) a más del 150 % (SOL en 2020 y 2021). Un índice de
#   acciones suele moverse entre un 15 y un 20 % al año.
# - La **rentabilidad** de un año a otro va de más del 10 000 % (SOL en 2021) a −94 % (SOL en 2022).
# - Los **días de subida** rondan el 50 % todos los años. Ganar o perder un año no depende de subir más
#   días, sino de cuánto se sube o se baja en unos pocos.

# %% [markdown]
# ### Por día de la semana
#
# Los mercados de acciones cierran el fin de semana; las criptos no. ¿Se nota? Se agrupa por día de la
# semana el volumen (en dólares: unidades × precio) y el tamaño del movimiento (el valor absoluto del
# rendimiento). Para comparar monedas, cada una se divide por su media.

# %%
nombres = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
largo["dia"] = pd.Categorical(largo["fecha"].dt.dayofweek.map(dict(enumerate(nombres))), nombres, ordered=True)
largo["volumen_usd"] = largo["volume"] * largo["close"]
semana = largo.groupby(["dia", "moneda"], observed=True).agg(volumen=("volumen_usd", "mean"),
                                                            movimiento=("ret", lambda r: r.abs().mean()))
semana = semana / semana.groupby("moneda").transform("mean")
semana.unstack("moneda").round(2)

# %% [markdown]
# El sábado y el domingo se negocia entre un 20 y un 35 % menos que la media, y los precios se mueven
# menos. Esa regularidad semanal reaparece en la Unidad 5 como **estacionalidad**.

# %% [markdown]
# ## 5. De largo a ancho, y unir tablas
#
# Para comparar monedas día a día (correlaciones, una cartera) conviene el formato **ancho**: una fila
# por día y una columna por moneda. `pivot` pasa de largo a ancho; `melt`, de ancho a largo.

# %%
ancho = largo.pivot(index="fecha", columns="moneda", values="close")
print(ancho.shape, "| vuelta a largo:", ancho.reset_index().melt(id_vars="fecha").dropna().shape)
ancho.head(3)

# %% [markdown]
# **Unir** dos tablas por la fecha: con `how="inner"` quedan solo los días que están en las dos; con
# `how="outer"`, todos, con huecos donde falte una. Es la diferencia entre `inner_join` y `full_join` en R.

# %%
btc = velas("BTC")[["close"]].rename(columns={"close": "BTC"})
sol = velas("SOL")[["close"]].rename(columns={"close": "SOL"})
print("inner:", len(btc.join(sol, how="inner")), "días | outer:", len(btc.join(sol, how="outer")), "días")

# %% [markdown]
# ## 6. Los datos que faltan
#
# Antes de agosto de 2020 SOL no existía. ¿Qué se hace con esos huecos? Depende de la pregunta:
#
# | Estrategia | Cuándo | Ejemplo en el piloto |
# |---|---|---|
# | **quitar** las filas (`dropna`) | cuando hacen falta todas las monedas a la vez | la volatilidad de la mezcla usa solo los días comunes |
# | **rellenar con el último precio** (`ffill`) | para valorar algo que tienes, si un día falta el precio | "Tiempos duros" rellena días sueltos sin vela |
# | rellenar con el **siguiente** (`bfill`) | **nunca** en series de precios | usaría un precio del futuro |
#
# `bfill` es una forma silenciosa de mirar el futuro: el modelo "sabría" un precio antes de que exista.

# %%
print("Huecos por moneda:\n", ancho.isna().sum())
comunes = ancho.dropna()
print("Días comunes a las tres:", len(comunes), "desde", comunes.index[0].date())

# %% [markdown]
# ## 7. Caso aplicado: datos listos para la econometría
#
# Los modelos de las unidades 4 a 7 trabajan con **rendimientos logarítmicos** diarios (Unidad 3 explica
# por qué), en los días comunes y sin huecos. Se guarda la tabla en un CSV para no repetir el proceso.

# %%
retornos = np.log(comunes).diff().dropna()
print(retornos.shape, "| huecos:", int(retornos.isna().sum().sum()))
retornos.to_csv("retornos_diarios.csv", date_format="%Y-%m-%d")
retornos.describe().round(4)

# %% [markdown]
# ## 8. Así lo usa el piloto: tu cartera, día a día
#
# La cartera de ejemplo del manual del piloto es **0,05 BTC, 1,2 ETH, 10 SOL y 500 USDT**. Para saber
# cuánto valía cada día se une lo que tienes (las unidades) con la tabla de precios: cada moneda vale
# `unidades × cierre`, y los USDT valen lo que dicen. Es una columna nueva y una suma por fila.
#
# El piloto hace esto en **Mi diario**, con un cuidado más: si compras o vendes, cambia las unidades
# desde ese día, y no cuenta los ingresos como ganancias (Unidad 9).

# %%
cartera = {"BTC": 0.05, "ETH": 1.2, "SOL": 10.0}
usdt = 500.0
valor_moneda = comunes[list(cartera)] * pd.Series(cartera)    # unidades × cierre, columna a columna
valor = valor_moneda.sum(axis=1) + usdt

ax = valor.plot(title="La cartera de ejemplo, valorada cada día (USDT)", color="#333333")
ax.set_xlabel("")
plt.show()
hoy = valor_moneda.iloc[-1]
print(f"Valor el {valor.index[-1]:%d-%m-%Y}: {valor.iloc[-1]:,.0f} USDT")
print("Peso de cada parte:", (pd.concat([hoy, pd.Series({"USDT": usdt})]) / valor.iloc[-1]).round(3).to_dict())

# %% [markdown]
# Y la **caída desde el máximo**: cuánto está la cartera por debajo de lo más alto que ha valido en los
# últimos 180 días. El piloto la vigila para su *freno*: si pasa del 10 %, recomienda invertir menos.

# %%
maximo_180 = valor.rolling(180, min_periods=1).max()
caida = valor / maximo_180 - 1
print(f"Caída vigente: {caida.iloc[-1]:.1%} | peor caída en el periodo: {caida.min():.1%} ({caida.idxmin():%d-%m-%Y})")

# %% [markdown]
# ## Ejercicios
#
# 1. **Tu cartera.** Cambia las unidades de `cartera` por otras y repite la sección 8. ¿Qué parte del
#    valor está en cada moneda hoy?
# 2. **Por mes.** Repite la tabla de la sección 4 agrupando por año y mes (`fecha.dt.to_period("M")`).
#    ¿Cuál fue el peor mes de BTC?
# 3. **Domingos.** ¿El rendimiento medio del domingo es distinto del de los demás días? Calcula la media
#    y su error típico (`std / √n`) por día de la semana. ¿La diferencia es mayor que dos errores típicos?
# 4. **El error del `bfill`.** Rellena hacia atrás los precios de SOL antes de agosto de 2020 y calcula la
#    volatilidad de la mezcla de las tres desde 2020. ¿Cambia mucho? ¿Por qué es un error de todos modos?
