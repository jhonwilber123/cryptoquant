# Contraste independiente, en R, de cifras de la tesis.
#
# `python -m cryptoquant tesis` deja en reports/tesis/ las velas diarias de
# los ocho activos, las variables del laboratorio calculadas por Python y los
# VaR pronosticados con las posiciones del modelo. Este script:
#
#   1. recalcula las 21 variables desde las velas con funciones.R y mide la
#      diferencia maxima con las de Python, activo por activo;
#   2. recalcula desde cero las pruebas de Kupiec y Christoffersen sobre los
#      VaR pronosticados y las compara con las de Python.
#
# Si alguna difiere mas alla del redondeo, una de las dos implementaciones
# esta mal. Uso, desde laboratorio/r:  Rscript tesis.R

source("funciones.R")
suppressPackageStartupMessages(library(tidyr))

DIR <- file.path("..", "..", "reports", "tesis")
stopifnot("Ejecute antes: python -m cryptoquant tesis" =
            file.exists(file.path(DIR, "variables_tesis.csv")))

# --- 1. Variables ---------------------------------------------------------------
velas <- read_csv(file.path(DIR, "velas_tesis.csv"), show_col_types = FALSE)
py <- read_csv(file.path(DIR, "variables_tesis.csv"), show_col_types = FALSE) |>
  mutate(dia = as.Date(fecha, tz = "UTC"))
cols <- c(VARIABLES_MODELO, "sma_20", "macd", "bb_sup", "sube")

variables <- lapply(split(velas, velas$activo), function(v) {
  d <- v |> arrange(fecha) |> mutate(dia = as.Date(fecha, tz = "UTC"))
  j <- inner_join(preparar_variables(d), filter(py, activo == v$activo[1]),
                  by = "dia", suffix = c("_r", "_py"))
  tibble(activo = v$activo[1], variable = cols,
         dif_maxima = sapply(cols, \(c) max(abs(j[[paste0(c, "_r")]] - j[[paste0(c, "_py")]]),
                                         na.rm = TRUE)),
         # Escala: la dif. maxima relativa al mayor valor absoluto de la variable.
         escala = sapply(cols, \(c) max(abs(j[[paste0(c, "_py")]]), na.rm = TRUE)),
         nas_distintos = sapply(cols, \(c) sum(is.na(j[[paste0(c, "_r")]]) != is.na(j[[paste0(c, "_py")]]))),
         filas = nrow(j))
}) |> bind_rows() |>
  mutate(dif_relativa = dif_maxima / escala)
write_csv(variables, file.path(DIR, "contraste_r_variables.csv"))

# --- 2. Backtest del VaR ----------------------------------------------------------
xlogy <- function(x, y) ifelse(x == 0, 0, x * log(y))

kupiec <- function(exc, alpha) {
  n <- length(exc); x <- sum(exc); p <- 1 - alpha; ph <- x / n
  lr <- -2 * (xlogy(n - x, 1 - p) + xlogy(x, p) - xlogy(n - x, 1 - ph) - xlogy(x, ph))
  c(excepciones = x, p_kupiec = pchisq(lr, 1, lower.tail = FALSE), lr_kupiec = lr)
}

christoffersen <- function(exc) {
  a <- head(exc, -1); b <- tail(exc, -1)
  n00 <- sum(!a & !b); n01 <- sum(!a & b); n10 <- sum(a & !b); n11 <- sum(a & b)
  p0 <- if (n00 + n01) n01 / (n00 + n01) else 0
  p1 <- if (n10 + n11) n11 / (n10 + n11) else 0
  p  <- (n01 + n11) / max(1, n00 + n01 + n10 + n11)
  ll0 <- xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
  ll1 <- xlogy(n00, 1 - p0) + xlogy(n01, p0) + xlogy(n10, 1 - p1) + xlogy(n11, p1)
  lr <- max(0, -2 * (ll0 - ll1))
  c(p_independencia = pchisq(lr, 1, lower.tail = FALSE), lr_independencia = lr)
}

METODOS <- c("historico", "normal", "t", "garch_t")
r_var <- expand_grid(cartera = c("modelo", "equiponderada", "BTC"), alpha = c(0.95, 0.99)) |>
  purrr::pmap(function(cartera, alpha) {
    t <- read_csv(file.path(DIR, sprintf("var_%s_%.0f.csv", cartera, alpha * 100)),
                  show_col_types = FALSE) |>
      filter(exposicion >= 0.01)
    lapply(METODOS, function(m) {
      exc <- -t$retorno > t[[paste0("var_", m)]]
      k <- kupiec(exc, alpha); ch <- christoffersen(exc)
      tibble(cartera = cartera, alpha = alpha, metodo = m, n = length(exc),
             excepciones = k[["excepciones"]], p_kupiec = k[["p_kupiec"]],
             p_independencia = ch[["p_independencia"]],
             p_cc = pchisq(k[["lr_kupiec"]] + ch[["lr_independencia"]], 2, lower.tail = FALSE))
    }) |> bind_rows()
  }) |> bind_rows() |>
  mutate(aceptado = n >= 250 & p_kupiec > 0.05 & p_cc > 0.05)

py_var <- read_csv(file.path(DIR, "validacion_var.csv"), show_col_types = FALSE)
comparacion <- inner_join(r_var, py_var, by = c("cartera", "alpha", "metodo"), suffix = c("_r", "_py")) |>
  transmute(cartera, alpha, metodo,
            excepciones_r, excepciones_py,
            dif_p_kupiec = abs(p_kupiec_r - p_kupiec_py),
            dif_p_independencia = abs(p_independencia_r - p_independencia_py),
            dif_p_cc = abs(p_cc - p_cobertura_condicional),
            aceptado_r, aceptado_py)
write_csv(comparacion, file.path(DIR, "contraste_r_var.csv"))

# --- Resumen --------------------------------------------------------------------
cat(sprintf("R %s\n", paste(R.version$major, R.version$minor, sep = ".")))
cat(sprintf("Variables: %d activos x %d variables; dif. maxima relativa %.2e; NA distintos %d\n",
            n_distinct(variables$activo), length(cols), max(variables$dif_relativa, na.rm = TRUE),
            sum(variables$nas_distintos)))
cat(sprintf("VaR: %d combinaciones; excepciones distintas %d; dif. maxima de p-valores %.2e; veredictos distintos %d\n",
            nrow(comparacion), sum(comparacion$excepciones_r != comparacion$excepciones_py),
            max(comparacion$dif_p_kupiec, comparacion$dif_p_independencia, comparacion$dif_p_cc),
            sum(comparacion$aceptado_r != comparacion$aceptado_py)))

write_csv(tibble(componente = c("R", "dplyr", "readr", "zoo"),
                 version = c(paste(R.version$major, R.version$minor, sep = "."),
                             as.character(packageVersion("dplyr")), as.character(packageVersion("readr")),
                             as.character(packageVersion("zoo")))),
          file.path(DIR, "contraste_r_entorno.csv"))
