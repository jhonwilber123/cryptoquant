# Prueba de la instalación: R y RStudio.
# Pega todo en la consola de RStudio (abajo a la izquierda) y pulsa Enter.
paquetes <- c("dplyr", "readr", "ggplot2", "jsonlite")
faltan <- paquetes[!vapply(paquetes, requireNamespace, logical(1), quietly = TRUE)]
if (length(faltan) > 0) install.packages(faltan)
fallos <- 0
for (p in paquetes) {
  if (requireNamespace(p, quietly = TRUE)) {
    cat("[OK]", p, format(packageVersion(p)), "\n")
  } else {
    cat("[REVISAR] No se pudo instalar", p, "\n")
    fallos <- fallos + 1
  }
}
cat("[OK]", R.version.string, "\n")
velas <- tryCatch(
  jsonlite::fromJSON("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=90"),
  error = function(e) NULL)
if (is.null(velas)) {
  cat("[REVISAR] No se pudo descargar de Binance. ¿Hay internet? ¿Lo bloquea un antivirus?\n")
  fallos <- fallos + 1
} else {
  cierre <- as.numeric(velas[, 5])
  cat("[OK] Internet: último cierre de BTC en Binance,", format(tail(cierre, 1), big.mark = ","), "USDT\n")
  plot(cierre, type = "l", main = "BTC, últimos 90 días (R)", xlab = "día", ylab = "USDT")
}
if (fallos == 0) cat("\nTODO LISTO. Escribe en el chat de la clase:  R y RStudio OK\n")
