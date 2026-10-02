# Clase 1 · Bloque 4 · El CSV como puente: de Python a R (Posit Cloud)
#
# El mismo archivo que se descargó en Colab, ahora en R. Ejecutar línea a
# línea con Ctrl + Enter. Paquetes: dplyr, readr y ggplot2 (instalados de
# antemano en el proyecto de la clase para no perder 5 minutos en clase).

library(dplyr)
library(readr)
library(ggplot2)

# 1. Leer el CSV ---------------------------------------------------------------
# Sube BTCUSDT_1d.csv al proyecto (panel Files → Upload). Si no está, se
# descarga aquí mismo. Posit Cloud corre en servidores de EE. UU., donde
# api.binance.com está bloqueado: se usa data-api.binance.vision.
if (!file.exists("BTCUSDT_1d.csv")) {
  inicio <- 1672531200000                      # 01-01-2023 en milisegundos
  paginas <- list()
  repeat {                                     # Binance entrega 1 000 velas por página
    url <- sprintf(paste0("https://data-api.binance.vision/api/v3/klines",
                          "?symbol=BTCUSDT&interval=1d&startTime=%.0f&limit=1000"), inicio)
    lote <- jsonlite::fromJSON(url)
    if (length(lote) == 0) break
    paginas[[length(paginas) + 1]] <- lote
    if (nrow(lote) < 1000) break
    inicio <- as.numeric(lote[nrow(lote), 1]) + 1
  }
  k <- do.call(rbind, paginas)
  k <- k[as.numeric(k[, 7]) < as.numeric(Sys.time()) * 1000, ]   # solo velas cerradas
  tibble(fecha = as.POSIXct(as.numeric(k[, 1]) / 1000, origin = "1970-01-01", tz = "UTC"),
         open = as.numeric(k[, 2]), high = as.numeric(k[, 3]), low = as.numeric(k[, 4]),
         close = as.numeric(k[, 5]), volume = as.numeric(k[, 6])) |>
    write_csv("BTCUSDT_1d.csv")
}
# El CSV de respaldo de Drive empieza en 2020; el de Colab, en 2023. Se recorta
# para que las cifras coincidan con las de Python sea cual sea el que se suba.
velas <- read_csv("BTCUSDT_1d.csv") |>
  filter(fecha >= as.POSIXct("2023-01-01", tz = "UTC"))
glimpse(velas)

# 2. dplyr: filtrar y agrupar --------------------------------------------------
# Cierre de cada mes de 2024 y cuánto se movió el precio dentro del día, de media.
velas |>
  filter(format(fecha, "%Y") == "2024") |>
  mutate(mes = format(fecha, "%Y-%m")) |>
  group_by(mes) |>
  summarise(cierre = last(close),
            rango_medio = mean((high - low) / close))

# 3. Rendimientos y crecimiento de 1 000 dólares --------------------------------
velas <- velas |>
  mutate(rendimiento = log(close / lag(close)),
         capital = 1000 * exp(cumsum(coalesce(rendimiento, 0))))

velas |>
  summarise(volatilidad_anual = sd(rendimiento, na.rm = TRUE) * sqrt(365),
            peor_dia = min(close / lag(close) - 1, na.rm = TRUE),
            capital_final = last(capital))

# 4. ggplot2: dos gráficos -----------------------------------------------------
ggplot(velas, aes(fecha, capital)) +
  geom_line(colour = "#2c7fb8") +
  labs(title = "1 000 dólares en bitcoin desde enero de 2023", x = NULL, y = "dólares")

ggplot(filter(velas, !is.na(rendimiento)), aes(rendimiento)) +   # el primer día no tiene rendimiento
  geom_histogram(bins = 80, fill = "#2c7fb8", alpha = 0.7) +
  labs(title = "Rendimientos diarios: la mayoría pequeños, algunos enormes",
       x = "rendimiento diario (logarítmico)", y = "días")
