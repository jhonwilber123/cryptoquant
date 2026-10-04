#!/usr/bin/env bash
# Lo que scripts/desplegar.ps1 ejecuta en el servidor (Ubuntu 24.04). Uso:
#   bash servidor.sh preparar    Docker, cortafuegos, memoria de intercambio y parches automaticos
#   bash servidor.sh instalar    descomprime piloto.tgz y construye y arranca la app
#   bash servidor.sh respaldar   deja una copia de la cartera en respaldo.tgz
#   bash servidor.sh restaurar   pone cartera.tgz como cartera (antes guarda la que habia)
#   bash servidor.sh estado      contenedores y ultimas lineas del registro
set -euo pipefail

DIR=/opt/piloto
VOLUMEN=piloto_datos   # proyecto "piloto" + volumen "datos" de docker-compose.yml

compose() {
  docker compose --env-file "$DIR/piloto.env" -f "$DIR/app/deploy/docker-compose.yml" "$@"
}

case "${1:-}" in
  preparar)
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v docker >/dev/null 2>&1; then
      curl -fsSL https://get.docker.com | sh
    fi
    # Cortafuegos: solo SSH, HTTP y HTTPS. La app (8501) no se publica: va detras de Caddy.
    command -v ufw >/dev/null 2>&1 || apt-get install -y -qq ufw >/dev/null
    ufw allow OpenSSH >/dev/null
    ufw allow 80/tcp >/dev/null
    ufw allow 443 >/dev/null
    ufw --force enable >/dev/null
    # 1 GB de intercambio: el Droplet mas pequeño se queda corto al construir la imagen.
    if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
      if fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile; then
        grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
      else
        echo "Aviso: no se pudo crear la memoria de intercambio." >&2
      fi
    fi
    apt-get update -qq && apt-get install -y -qq unattended-upgrades >/dev/null
    mkdir -p "$DIR" && chmod 700 "$DIR"
    docker --version
    ;;
  instalar)
    test -f "$DIR/piloto.env" || { echo "Falta $DIR/piloto.env: ejecute desplegar.ps1 -Clave" >&2; exit 1; }
    rm -rf "$DIR/app.nuevo" && mkdir -p "$DIR/app.nuevo"
    tar -xzf "$DIR/piloto.tgz" -C "$DIR/app.nuevo"
    rm -rf "$DIR/app" && mv "$DIR/app.nuevo" "$DIR/app" && rm -f "$DIR/piloto.tgz"
    compose up -d --build --remove-orphans
    docker image prune -f >/dev/null
    compose ps
    ;;
  respaldar)
    docker run --rm -v "$VOLUMEN":/datos:ro -v "$DIR":/salida alpine \
      tar -czf /salida/respaldo.tgz -C /datos .
    ;;
  restaurar)
    docker run --rm -v "$VOLUMEN":/datos -v "$DIR":/entrada alpine sh -c \
      'tar -czf /entrada/antes-de-restaurar.tgz -C /datos . && tar -xzf /entrada/cartera.tgz -C /datos && chown -R 1000:1000 /datos'
    rm -f "$DIR/cartera.tgz"
    echo "La cartera anterior quedo en $DIR/antes-de-restaurar.tgz"
    ;;
  estado)
    compose ps
    compose logs --tail 20
    ;;
  *)
    echo "Uso: bash servidor.sh preparar|instalar|respaldar|restaurar|estado" >&2
    exit 2
    ;;
esac
