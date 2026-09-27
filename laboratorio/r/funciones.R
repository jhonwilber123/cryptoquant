# Funciones auxiliares del laboratorio en R.
#
# Los indicadores replican exactamente los de Python
# (cryptoquant/laboratorio/variables.py), para que ambas versiones puedan
# contrastarse cifra a cifra con comparar_con_python().

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# --- Datos --------------------------------------------------------------------

# Dominios de la API publica de Binance, por orden. api.binance.com responde
# 451 desde IP de EE. UU. (donde suele correr Posit Cloud); data-api.binance.vision
# sirve los mismos datos de mercado sin esa restriccion y sin clave.
DOMINIOS_BINANCE <- c("https://data-api.binance.vision", "https://api.binance.com")

primer_dominio_que_responde <- function() {
  for (d in DOMINIOS_BINANCE) {
    ok <- tryCatch({
      jsonlite::fromJSON(paste0(d, "/api/v3/ping")); TRUE
    }, error = function(e) FALSE)
    if (ok) return(d)
  }
  stop("Binance no responde desde este equipo: use el CSV de respaldo")
}

# Descarga velas de la API publica de Binance (sin clave), 1000 por peticion.
# Solo hace falta si no existe el CSV que exporta Python.
descargar_binance <- function(simbolo = "BTCUSDT", intervalo = "1h",
                              desde = "2022-01-01") {
  paso_ms <- c("1h" = 3600e3, "4h" = 14400e3, "1d" = 86400e3)[[intervalo]]
  cursor <- as.numeric(as.POSIXct(desde, tz = "UTC")) * 1000
  dominio <- primer_dominio_que_responde()
  trozos <- list()
  repeat {
    url <- sprintf(
      "%s/api/v3/klines?symbol=%s&interval=%s&startTime=%.0f&limit=1000",
      dominio, simbolo, intervalo, cursor)
    k <- jsonlite::fromJSON(url)
    if (length(k) == 0) break
    trozos[[length(trozos) + 1]] <- k
    if (nrow(k) < 1000) break
    cursor <- as.numeric(k[nrow(k), 1]) + paso_ms
    Sys.sleep(0.2)
  }
  m <- do.call(rbind, trozos)
  velas <- tibble(
    fecha  = as.POSIXct(as.numeric(m[, 1]) / 1000, origin = "1970-01-01", tz = "UTC"),
    open   = as.numeric(m[, 2]), high = as.numeric(m[, 3]),
    low    = as.numeric(m[, 4]), close = as.numeric(m[, 5]),
    volume = as.numeric(m[, 6]),
    cierre = as.numeric(m[, 7]) / 1000
  )
  # La vela en curso no ha cerrado: su "cierre" es el precio de este instante.
  velas |>
    filter(cierre + 120 <= as.numeric(Sys.time())) |>
    select(-cierre) |>
    distinct(fecha, .keep_all = TRUE)
}

leer_velas <- function(ruta) {
  read_csv(ruta, col_types = cols(fecha = col_datetime(), .default = col_double())) |>
    mutate(fecha = lubridate::with_tz(fecha, "UTC"))
}

# Velas horarias -> diarias UTC. El ultimo dia se descarta si esta incompleto.
a_diario <- function(velas) {
  paso <- as.numeric(difftime(velas$fecha[2], velas$fecha[1], units = "secs"))
  ultimo_fin <- max(velas$fecha) + paso
  diario <- velas |>
    mutate(dia = as.Date(fecha, tz = "UTC")) |>
    group_by(dia) |>
    summarise(open = first(open), high = max(high), low = min(low),
              close = last(close), volume = sum(volume), .groups = "drop")
  if (ultimo_fin < as.POSIXct(max(diario$dia) + 1, tz = "UTC")) {
    diario <- head(diario, -1)
  }
  diario
}

# --- Indicadores (identicos a pandas) -----------------------------------------

media_movil <- function(x, n) as.numeric(stats::filter(x, rep(1 / n, n), sides = 1))

desv_movil <- function(x, n) zoo::rollapplyr(x, n, sd, fill = NA)

# pandas ewm(span, adjust = FALSE): y[1] = x[1]; y[t] = a x[t] + (1 - a) y[t-1]
ema <- function(x, span) {
  a <- 2 / (span + 1)
  # Con init = x[1]: y[1] = a x[1] + (1 - a) x[1] = x[1], como pandas.
  as.numeric(stats::filter(a * x, 1 - a, method = "recursive", init = x[1]))
}

# pandas ewm(alpha, adjust = TRUE): media ponderada con pesos (1 - a)^i sobre
# las observaciones disponibles, ignorando los NA iniciales.
ewm_ajustada <- function(x, alpha) {
  y <- rep(NA_real_, length(x))
  num <- 0; den <- 0
  for (i in seq_along(x)) {
    if (is.na(x[i])) next
    num <- x[i] + (1 - alpha) * num
    den <- 1 + (1 - alpha) * den
    y[i] <- num / den
  }
  y
}

rsi <- function(close, n = 14) {
  d <- c(NA, diff(close))
  g <- ewm_ajustada(pmax(d, 0), 1 / n)
  p <- ewm_ajustada(pmax(-d, 0), 1 / n)
  r <- 100 - 100 / (1 + g / ifelse(p == 0, NA, p))
  r[is.na(r)] <- 50
  r[seq_len(n)] <- NA  # arranque
  r
}

preparar_variables <- function(diario, rezagos = c(1, 2, 3, 5, 10)) {
  v <- diario |>
    mutate(
      retorno_simple = close / lag(close) - 1,
      retorno        = log(close / lag(close)),
      sma_20 = media_movil(close, 20),
      sma_50 = media_movil(close, 50),
      dist_sma_20 = close / sma_20 - 1,
      dist_sma_50 = close / sma_50 - 1,
      rsi_14 = rsi(close, 14),
      macd       = ema(close, 12) - ema(close, 26),
      macd_senal = ema(macd, 9),
      macd_hist  = macd - macd_senal,
      bb_media = sma_20,
      bb_sup   = sma_20 + 2 * desv_movil(close, 20),
      bb_inf   = sma_20 - 2 * desv_movil(close, 20),
      bb_pct   = (close - bb_inf) / (bb_sup - bb_inf),
      bb_ancho = (bb_sup - bb_inf) / bb_media,
      vol_20    = desv_movil(retorno, 20),
      vol_ratio = vol_20 / desv_movil(retorno, 100),
      rango_rel = (high - low) / close,
      # Objetivo: lo unico que mira al futuro. Nunca entra como variable.
      retorno_futuro = log(lead(close) / close),
      sube = as.numeric(retorno_futuro > 0)
    )
  v[seq_len(34), c("macd", "macd_senal", "macd_hist")] <- NA  # 26 + 9 - 1 de arranque
  v <- v |> mutate(macd_rel = macd / close, macd_senal_rel = macd_senal / close,
                   macd_hist_rel = macd_hist / close)
  for (k in rezagos) v[[paste0("retorno_l", k)]] <- dplyr::lag(v$retorno, k)
  v
}

VARIABLES_MODELO <- c(
  "retorno", paste0("retorno_l", c(1, 2, 3, 5, 10)),
  "dist_sma_20", "dist_sma_50", "rsi_14",
  "macd_rel", "macd_senal_rel", "macd_hist_rel",
  "bb_pct", "bb_ancho", "vol_20", "vol_ratio", "rango_rel"
)

# --- Utilidades ----------------------------------------------------------------

# AUC por Mann-Whitney: probabilidad de que un dia que sube reciba mas
# puntuacion que uno que baja.
auc <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# TRUE en la vela en que `a` pasa por encima de `b` (serie o umbral fijo).
cruza_arriba <- function(a, b) {
  b_prev <- if (length(b) == 1) b else lag(b)
  (a > b) & (lag(a) <= b_prev)
}

# Diferencia maxima entre las variables de R y las de Python, por columna.
comparar_con_python <- function(v, ruta_python) {
  py <- read_csv(ruta_python, show_col_types = FALSE)
  py$dia <- as.Date(py$fecha)
  j <- inner_join(v, py, by = "dia", suffix = c("_r", "_py"))
  cols <- intersect(c(VARIABLES_MODELO, "sma_20", "macd", "bb_sup", "sube"), names(v))
  tibble(variable = cols,
         dif_maxima = sapply(cols, \(c) max(abs(j[[paste0(c, "_r")]] - j[[paste0(c, "_py")]]),
                                          na.rm = TRUE)))
}
