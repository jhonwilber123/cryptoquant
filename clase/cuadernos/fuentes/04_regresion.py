# %% [markdown]
# # Unidad 4 · Modelamiento cuantitativo: regresión lineal
#
# **Qué vas a aprender**
# - Ajustar una regresión lineal **simple** y una **múltiple**, e interpretar sus coeficientes.
# - Revisar los supuestos: residuos, heterocedasticidad, colas gruesas, y usar errores típicos robustos.
# - Reconocer la **multicolinealidad** y medirla con el VIF.
# - Evaluar un modelo **fuera de la muestra**, que es la única evaluación que cuenta.
#
# **Qué tiene que ver con la app.** En la pantalla **Hoy**, el piloto muestra qué parte del **riesgo**
# aporta cada moneda: una puede ser el 30 % del dinero y el 50 % del riesgo. Esa cuenta es una
# regresión disfrazada: el peso en riesgo es el peso en dinero por la **beta** de la moneda frente a tu
# cartera. Al final la reproduces.

# %% preparacion

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

precios = cierres().dropna()                       # días comunes a BTC, ETH y SOL
ret = np.log(precios).diff().dropna()              # rendimientos logarítmicos diarios
print(ret.shape[0], "días, del", ret.index[0].date(), "al", ret.index[-1].date())
ret.corr().round(3)

# %% [markdown]
# ## 1. Regresión simple: ¿cuánto se mueve ETH cuando se mueve BTC?
#
# $$r_{ETH,t} = \alpha + \beta\, r_{BTC,t} + \varepsilon_t$$
#
# - $\beta$ (**beta**): cuánto se mueve ETH, de media, cuando BTC se mueve un 1 %.
# - $\alpha$: el rendimiento de ETH los días en que BTC no se mueve.
# - $\varepsilon$: lo que BTC no explica.
#
# `smf.ols` usa la notación de fórmulas de R: `"ETH ~ BTC"` se lee "ETH explicado por BTC".

# %%
m1 = smf.ols("ETH ~ BTC", data=ret).fit()
print(m1.summary().tables[1])
print(f"R² = {m1.rsquared:.3f}   n = {int(m1.nobs)}")

# %%
x = np.linspace(ret["BTC"].min(), ret["BTC"].max(), 2)
plt.figure(figsize=(6, 5))
plt.scatter(ret["BTC"], ret["ETH"], s=4, alpha=0.3, color=COLORES["ETH"])
plt.plot(x, m1.params["Intercept"] + m1.params["BTC"] * x, color="black")
plt.xlabel("rendimiento de BTC"); plt.ylabel("rendimiento de ETH")
plt.title(f"ETH frente a BTC, días comunes (β = {m1.params['BTC']:.2f})"); plt.show()

# %% [markdown]
# **Cómo se lee:**
# - **β ≈ 1,1**: un día en que BTC sube un 1 %, ETH sube de media un 1,1 %. ETH amplifica a BTC.
# - **α** es prácticamente cero y su *p-valor* es alto: no hay un rendimiento "extra" de ETH
#   independiente de BTC que se pueda distinguir del azar.
# - **R² ≈ 0,65**: BTC explica dos tercios de lo que se mueve ETH cada día. El tercio restante es
#   propio de ETH.
# - El *p-valor* de β es 0 (redondeado): con más de 2000 días, una relación así no puede ser casualidad.

# %% [markdown]
# ## 2. Los supuestos, y errores típicos robustos
#
# Los *p-valores* de la tabla suponen que los errores tienen siempre la misma varianza
# (**homocedasticidad**) y, para muestras pequeñas, que son normales. En cripto no se cumple ninguna de
# las dos cosas. Dos pruebas:
# - **Breusch-Pagan**: ¿la varianza del error depende de las variables? (p pequeño = sí).
# - **Gráfico de residuos**: la nube debería tener el mismo ancho en todo el eje.

# %%
lm, p_lm, _, _ = het_breuschpagan(m1.resid, m1.model.exog)
print(f"Breusch-Pagan: estadístico {lm:.1f}, p-valor {p_lm:.2g}")
curtosis = ((m1.resid - m1.resid.mean()) ** 4).mean() / m1.resid.var(ddof=0) ** 2
print(f"Curtosis de los residuos: {curtosis:.1f} (una normal tiene 3)")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
ejes[0].scatter(m1.fittedvalues, m1.resid, s=4, alpha=0.3); ejes[0].axhline(0, color="black")
ejes[0].set_title("Residuos frente a valores ajustados")
sm.qqplot(m1.resid, line="s", ax=ejes[1], markersize=2); ejes[1].set_title("Residuos frente a la normal (QQ)")
plt.tight_layout(); plt.show()

# %% [markdown]
# Hay heterocedasticidad (los días agitados tienen errores más grandes) y colas mucho más gruesas que
# la normal (los puntos del QQ se escapan de la recta en los extremos). La β sigue siendo válida, pero
# su error típico no. La solución estándar son los **errores típicos robustos** (HC1, de White): se
# calculan sin suponer varianza constante.

# %%
m1r = smf.ols("ETH ~ BTC", data=ret).fit(cov_type="HC1")
pd.DataFrame({"coef": m1.params, "ee_clasico": m1.bse, "ee_robusto": m1r.bse}).round(5)

# %% [markdown]
# El error típico robusto de β es mayor que el clásico: la estimación es menos precisa de lo que decía
# la tabla. En datos financieros, usa siempre los robustos.

# %% [markdown]
# ## 3. Regresión múltiple: SOL explicado por BTC y ETH
#
# $$r_{SOL,t} = \alpha + \beta_1\, r_{BTC,t} + \beta_2\, r_{ETH,t} + \varepsilon_t$$
#
# En una regresión múltiple, cada coeficiente es el efecto de su variable **manteniendo las demás
# constantes**. Por eso cambia respecto a la regresión simple:

# %%
m_btc = smf.ols("SOL ~ BTC", data=ret).fit(cov_type="HC1")
m_eth = smf.ols("SOL ~ ETH", data=ret).fit(cov_type="HC1")
m2 = smf.ols("SOL ~ BTC + ETH", data=ret).fit(cov_type="HC1")
pd.DataFrame({"solo BTC": m_btc.params, "solo ETH": m_eth.params, "BTC y ETH": m2.params,
              "ee robusto": m2.bse}).round(3)

# %% [markdown]
# Sola, la β de BTC sobre SOL es 1,17; con ETH en el modelo baja a 0,32, porque parte de lo que
# "explicaba" BTC era en realidad lo que BTC y ETH tienen en común. Cada β múltiple responde a otra
# pregunta: *si BTC se mueve un 1 % y ETH no se mueve, ¿cuánto se mueve SOL?* Con monedas tan
# correlacionadas, esa situación casi no ocurre, y ahí empieza el problema de la sección siguiente.

# %%
print(f"R²: solo BTC {m_btc.rsquared:.3f} | solo ETH {m_eth.rsquared:.3f} | BTC y ETH {m2.rsquared:.3f}")

# %% [markdown]
# ## 4. Multicolinealidad
#
# Cuando dos variables explicativas se mueven casi juntas, el modelo no sabe a cuál atribuir el efecto:
# los coeficientes se vuelven **inestables** y sus errores típicos se **inflan**. Se mide con el **VIF**
# (factor de inflación de la varianza) de cada variable, $1/(1-R^2_j)$, donde $R^2_j$ es el de la
# regresión de esa variable sobre las demás:
#
# | VIF | Lectura |
# |---|---|
# | 1 | sin relación con las demás |
# | 1 a 5 | moderada, normalmente aceptable |
# | más de 10 | grave: los coeficientes no son fiables |

# %%
def vif(datos, columnas):
    X = sm.add_constant(datos[columnas])
    return pd.Series([variance_inflation_factor(X.values, i + 1) for i in range(len(columnas))], index=columnas)


print("BTC y ETH:", vif(ret, ["BTC", "ETH"]).round(2).to_dict())

# %% [markdown]
# Un VIF de casi 3 entre BTC y ETH es moderado. Para ver uno grave, añadamos una variable **casi idéntica**
# a BTC: su rendimiento medido de la apertura al cierre del día, en lugar de cierre a cierre. Como las
# criptos cotizan sin pausa, la apertura de hoy es prácticamente el cierre de ayer.

# %%
btc = velas("BTC").reindex(ret.index)
ret2 = ret.assign(BTC_ac=np.log(btc["close"] / btc["open"]))
print("Correlación entre las dos versiones de BTC:", round(ret2["BTC"].corr(ret2["BTC_ac"]), 5))
print("VIF:", vif(ret2, ["BTC", "BTC_ac", "ETH"]).round(0).to_dict())
m3 = smf.ols("SOL ~ BTC + BTC_ac + ETH", data=ret2).fit(cov_type="HC1")
pd.DataFrame({"coef": m3.params, "ee robusto": m3.bse}).round(3)

# %% [markdown]
# La correlación sale 1,0 porque las dos versiones solo difieren en el sexto decimal. El VIF pasa de
# 300 000, y el modelo reparte el efecto de BTC entre las dos al azar: una sale con +10 y la otra con −10,
# con errores típicos de 20. Su suma (≈ 0,3) es la β de antes, y ETH ni se entera. **Remedio:** quitar
# una de las dos, o combinarlas.

# %% [markdown]
# ## 5. Evaluar el modelo: dentro y fuera de la muestra
#
# El R² dentro de la muestra mide lo bien que el modelo **describe** los datos con los que se ajustó.
# Lo que importa para decidir es lo bien que funciona con datos **nuevos**. Se entrena hasta 2024 y se
# evalúa en 2025-2026, comparando con el modelo más simple posible (predecir la media).

# %%
def r2_fuera(y, pred, referencia):
    """R² fuera de la muestra: 1 - error del modelo / error de predecir siempre la media de entrenamiento."""
    return 1 - ((y - pred) ** 2).sum() / ((y - referencia) ** 2).sum()


entreno, prueba = ret[ret.index < "2025-01-01"], ret[ret.index >= "2025-01-01"]
m_e = smf.ols("SOL ~ BTC + ETH", data=entreno).fit()
pred = m_e.predict(prueba)
print(f"R² en entrenamiento: {m_e.rsquared:.3f}")
print(f"R² fuera de la muestra (2025-2026): {r2_fuera(prueba['SOL'], pred, entreno['SOL'].mean()):.3f}")
print(f"Error medio (RMSE): modelo {np.sqrt(((prueba['SOL'] - pred) ** 2).mean()):.4f}"
      f" | predecir la media {np.sqrt(((prueba['SOL'] - entreno['SOL'].mean()) ** 2).mean()):.4f}")

# %% [markdown]
# La relación entre monedas **del mismo día** se mantiene fuera de la muestra, e incluso mejora: en
# 2025-2026 SOL se movió más pegada a BTC y ETH que en sus primeros años. Es una primera señal de que las
# relaciones cambian. Calculada con ventanas de 180 días, la beta de ETH frente a BTC cambia bastante:

# %%
cov_movil = ret["ETH"].rolling(180).cov(ret["BTC"])
beta_movil = cov_movil / ret["BTC"].rolling(180).var()
ax = beta_movil.plot(title="Beta de ETH frente a BTC, ventanas de 180 días", color=COLORES["ETH"])
ax.axhline(m1.params["BTC"], color="black", linestyle="--", label="toda la muestra"); ax.legend(); ax.set_xlabel("")
plt.show()

# %% [markdown]
# ## 6. Caso aplicado: ¿qué relaciones sirven para **anticipar**?
#
# Que ETH y BTC se muevan juntos el mismo día no sirve para invertir: cuando lo sabes, ya pasó. La
# pregunta útil es si algo de **hoy** anticipa algo de **mañana**. Dos regresiones con las mismas
# variables de BTC (Unidad 3), entrenadas hasta 2024 y evaluadas en 2025-2026:
#
# 1. El **rendimiento** de mañana.
# 2. La **volatilidad** de los próximos 7 días.

# %%
d = pd.DataFrame({"r": np.log(btc["close"]).diff()})
d["r_1"] = d["r"].shift(1)
d["vol_30"] = d["r"].rolling(30).std() * np.sqrt(ANUAL)
d["vol_ewma"] = np.sqrt((d["r"] ** 2).ewm(alpha=0.06, adjust=False).mean() * ANUAL)
d["r_manana"] = d["r"].shift(-1)
d["vol_7"] = d["r"].rolling(7).std().shift(-7) * np.sqrt(ANUAL)
d = d.dropna()
e, p = d[d.index < "2025-01-01"], d[d.index >= "2025-01-01"]

filas = {}
for objetivo in ["r_manana", "vol_7"]:
    m = smf.ols(f"{objetivo} ~ r + r_1 + vol_30 + vol_ewma", data=e).fit(cov_type="HC1")
    filas[objetivo] = {"R² entrenamiento": m.rsquared,
                       "R² fuera de la muestra": r2_fuera(p[objetivo], m.predict(p), e[objetivo].mean())}
pd.DataFrame(filas).T.round(4)

# %% [markdown]
# - **Rendimiento de mañana**: el R² es prácticamente cero dentro y fuera de la muestra: el modelo no
#   mejora en nada a decir siempre "la media". La dirección no se deja anticipar.
# - **Volatilidad de la semana próxima**: se anticipa una cuarta parte de su variación en entrenamiento
#   y más de un tercio en 2025-2026.
#
# Es la misma conclusión de la tesis que sostiene al piloto: no intenta adivinar si el precio sube; mide
# cuánto se va a mover y ajusta cuánto tienes en cripto.

# %% [markdown]
# ## 7. Así lo usa el piloto: cuánto riesgo aporta cada moneda
#
# El riesgo de una cartera no es la suma del de sus monedas: depende de cuánto se mueven **juntas**. El
# piloto reparte la varianza de la cartera entre las monedas con la matriz de covarianzas $\Sigma$:
#
# $$\text{peso en riesgo}_i = \frac{w_i\,(\Sigma w)_i}{w^\top \Sigma\, w} = w_i \times \beta_i$$
#
# donde $\beta_i$ es la **beta de la moneda frente a tu cartera**: la pendiente de la regresión del
# rendimiento de la moneda sobre el de la cartera. Una moneda con β > 1 aporta más riesgo que dinero.
#
# Con la cartera de ejemplo (0,05 BTC, 1,2 ETH, 10 SOL y 500 USDT) y los cierres del 26-09-2026. El piloto
# usa covarianzas EWMA (λ = 0,94, como la volatilidad); aquí, primero, las betas por regresión del último
# año, y después la cuenta exacta del piloto.

# %%
unidades = pd.Series({"BTC": 0.05, "ETH": 1.2, "SOL": 10.0})
usdt = 500.0
valor = unidades * precios.iloc[-1]
total = valor.sum() + usdt
w = valor / total                                   # peso en dinero (el resto, en USDT)

ultimo_ano = ret.iloc[-365:]
r_cartera = ultimo_ano @ w                          # rendimiento de la cartera cada día
betas = pd.Series({a: smf.ols("y ~ x", data=pd.DataFrame({"y": ultimo_ano[a], "x": r_cartera})).fit().params["x"]
                   for a in ACTIVOS})


def covarianza_ewma(lr, lam=0.94, arranque=30):
    """Como el piloto: empieza con la covarianza de los 30 primeros días y la actualiza cada día."""
    x = lr.to_numpy()
    v = np.cov(x[:arranque].T, bias=True)
    for fila in x:
        v = lam * v + (1 - lam) * np.outer(fila, fila)
    return pd.DataFrame(v, index=lr.columns, columns=lr.columns)


S = covarianza_ewma(ret)
marginal = S @ w
pd.DataFrame({"peso en dinero": w, "beta (regresión, último año)": betas,
              "peso en riesgo (regresión)": w * betas / (w * betas).sum(),
              "peso en riesgo (piloto, EWMA)": w * marginal / (w @ marginal),
              "volatilidad anual (EWMA)": np.sqrt(np.diag(S) * ANUAL)}).round(3)

# %% [markdown]
# SOL es el 13 % del dinero y aporta el 18 % del riesgo, porque se mueve más que la cartera (β > 1);
# BTC es el 46 % del dinero y el 43 % del riesgo. Las dos estimaciones de la beta (regresión del último
# año y EWMA) no coinciden porque pesan el pasado de forma distinta: la EWMA mira sobre todo el último
# mes. La última columna es la tabla **"Qué parte del riesgo aporta cada moneda"** del
# piloto, con las mismas cifras que muestra para esta cartera con los cierres de ese día. (El peso en
# dinero del piloto incluye los USDT en el total; por eso las tres monedas suman menos de 1.)

# %% [markdown]
# ## Ejercicios
#
# 1. **Al revés.** Ajusta `BTC ~ ETH`. ¿La β es la inversa de la de `ETH ~ BTC`? ¿Por qué no? (Pista: el
#    R² es el mismo; mira la fórmula de β = cov / var.)
# 2. **Rendimientos en %.** Multiplica los rendimientos por 100 y repite la regresión simple. ¿Qué cambia
#    en α, β, sus errores típicos y el R²?
# 3. **Un año agitado.** Ajusta `ETH ~ BTC` solo con 2022 y solo con 2025. ¿Cambia la β? ¿Y el R²?
# 4. **Tu cartera.** Cambia las unidades en la sección 7. ¿Hay alguna combinación en la que el peso en
#    riesgo de cada moneda sea igual a su peso en dinero? ¿Qué tendrían que cumplir las betas?
