# Lo que comparten todos los cuadernos en R: librerías, periodo de estudio y
# cómo se cargan los precios de Binance. Cómo se construye, en la Unidad 1.
#
# Es la misma lógica que la celda de «Preparación» de los cuadernos de Python,
# con el mismo periodo: así las cifras de R y de Python se pueden comparar.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(lubridate)
})

ACTIVOS <- c("BTC", "ETH", "SOL")       # las monedas de la cartera de ejemplo del piloto
COLORES <- c(BTC = "#F7931A", ETH = "#627EEA", SOL = "#9945FF")
DESDE <- as.Date("2020-01-01")          # el piloto también mira desde 2020
HASTA <- as.Date("2026-09-26")          # fija las cifras del texto; NA = hasta ayer
ANUAL <- 365                            # las criptos cotizan todos los días del año
API <- "https://data-api.binance.vision/api/v3/klines"   # api.binance.com bloquea EE. UU.

theme_set(theme_minimal(base_size = 11))
options(dplyr.summarise.inform = FALSE, pillar.sigfig = 6)

# Velas diarias YA CERRADAS de SIMBOLO/USDT en Binance, 1000 por petición.
descargar <- function(simbolo, desde = DESDE) {
  inicio <- as.numeric(as.POSIXct(desde, tz = "UTC")) * 1000
  trozos <- list()
  repeat {
    url <- sprintf("%s?symbol=%sUSDT&interval=1d&startTime=%.0f&limit=1000", API, simbolo, inicio)
    lote <- jsonlite::fromJSON(url)
    if (length(lote) == 0) break
    trozos[[length(trozos) + 1]] <- lote[, 1:7, drop = FALSE]
    if (nrow(lote) < 1000) break
    inicio <- as.numeric(lote[nrow(lote), 1]) + 1     # la página siguiente, tras la última vela
  }
  m <- do.call(rbind, trozos)
  ahora_ms <- as.numeric(Sys.time()) * 1000
  tibble(
    fecha  = as.Date(as.POSIXct(as.numeric(m[, 1]) / 1000, origin = "1970-01-01", tz = "UTC")),
    open   = as.numeric(m[, 2]), high = as.numeric(m[, 3]), low = as.numeric(m[, 4]),
    close  = as.numeric(m[, 5]), volume = as.numeric(m[, 6]),
    cierre_ms = as.numeric(m[, 7])
  ) |>
    filter(cierre_ms < ahora_ms) |>                  # fuera la vela que no ha cerrado
    select(-cierre_ms) |>
    distinct(fecha, .keep_all = TRUE)
}

# Velas diarias del periodo de estudio: del CSV si ya está, si no de Binance (y lo guarda).
velas <- function(simbolo) {
  archivo <- paste0(simbolo, "USDT_1d.csv")
  fin <- if (is.na(HASTA)) Sys.Date() - 1 else HASTA
  df <- NULL
  if (file.exists(archivo)) {
    df <- read_csv(archivo, show_col_types = FALSE) |>
      mutate(fecha = as.Date(fecha, tz = "UTC"))
  }
  if (is.null(df) || max(df$fecha) < fin) {
    nuevo <- tryCatch(descargar(simbolo), error = function(e) {
      if (is.null(df)) stop("No pude descargar ", simbolo, " de Binance (", conditionMessage(e),
                            "). Sube ", archivo, " (datos de respaldo del curso) y repite.")
      message("Aviso: sin conexión con Binance; ", archivo, " llega solo al ", max(df$fecha), ".")
      NULL
    })
    if (!is.null(nuevo)) {
      df <- nuevo
      write_csv(mutate(df, fecha = format(as.POSIXct(fecha, tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")), archivo)
    }
  }
  filter(df, fecha >= DESDE, fecha <= fin)
}

# Cierres diarios, una columna por moneda: la tabla con la que trabaja el piloto.
cierres <- function(activos = ACTIVOS) {
  tablas <- lapply(activos, function(a) velas(a) |> select(fecha, !!a := close))
  Reduce(function(x, y) full_join(x, y, by = "fecha"), tablas) |> arrange(fecha)
}
