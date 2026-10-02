# Publicar el piloto en un servidor (Docker)

Para abrir el piloto desde cualquier sitio, también el móvil, sin dejar el PC
encendido. Va en un servidor propio, un *Droplet* de DigitalOcean, con Docker:
la app detrás de [Caddy](https://caddyserver.com), que pone el HTTPS, y con
contraseña. Todo se maneja desde Windows con [`scripts/desplegar.ps1`](../scripts/desplegar.ps1).

| Archivo | Para qué |
|---|---|
| `Dockerfile` | La imagen de la app: solo lo que usa, un usuario sin privilegios, la cartera en el volumen `/datos` y la contraseña obligatoria |
| `requirements-piloto.txt` | Sus librerías, con las versiones probadas en el PC |
| `docker-compose.yml` | La app y Caddy en el servidor |
| `docker-compose.local.yml` | Para probarla en el PC con Docker Desktop |
| `Caddyfile` | HTTPS automático (Let's Encrypt) y cabeceras de seguridad |
| `servidor.sh` | Lo que se ejecuta en el servidor: `preparar`, `instalar`, `respaldar`, `restaurar`, `estado` |

## Por qué un Droplet y no App Platform

App Platform no guarda archivos: cada despliegue o reinicio empieza con el
disco vacío, y el piloto perdería la cartera y su historial, que viven en
archivos. En un Droplet, la cartera está en un volumen de Docker que sobrevive
a cada actualización.

## Pasos

1. **Crear el Droplet**: Ubuntu 24.04, plan *Basic* de 1 GB y autenticación con
   la clave SSH de este equipo (`%USERPROFILE%\.ssh\id_ed25519.pub`).
2. **Prepararlo y publicar la app**, desde la raíz del proyecto:

   ```powershell
   .\scripts\desplegar.ps1 -Servidor 203.0.113.10 -Preparar
   ```

   Instala Docker, el cortafuegos (solo SSH, 80 y 443), 1 GB de intercambio y
   las actualizaciones automáticas. Pide la contraseña de la app (dos veces),
   construye la imagen en el servidor y espera a que responda.
3. **Abrir** `https://203-0-113-10.sslip.io` (la IP con guiones) y entrar con la
   contraseña. Con un dominio propio, se crea un registro A que apunte a la IP y
   se añade `-Dominio piloto.midominio.com`.

| Después | Orden |
|---|---|
| Llevar una versión nueva del código | `.\scripts\desplegar.ps1 -Servidor <IP>` |
| Subir la cartera de este equipo | `... -SubirCartera` (guarda antes la del servidor) |
| Copia de la cartera del servidor | `... -Respaldar` (a `data/piloto/respaldos_servidor/`) |
| Cambiar la contraseña | `... -Clave` |
| Ver los contenedores y el registro | `... -Estado` |

## Probarlo en el PC

Con Docker Desktop, desde la raíz del proyecto:

```powershell
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.local.yml up --build
```

La app queda en <http://127.0.0.1:8501>, sin Caddy ni contraseña y sin salir de
este equipo. Su cartera vive en el volumen de Docker, no en `data/piloto/`.

## Seguridad

- **Contraseña obligatoria.** La imagen no abre la app sin ella. El servidor solo
  guarda su derivada PBKDF2-SHA256 (`/opt/piloto/piloto.env`, legible solo por
  root), nunca la contraseña. Cada fallo espera 2 segundos y queda en el registro.
- **HTTPS** con certificado de Let's Encrypt, HSTS y cabeceras de seguridad. La
  app no se publica: solo Caddy escucha en los puertos 80 y 443.
- **Sin claves ni órdenes.** El servidor solo lee precios públicos de Binance,
  igual que en el PC.
- **La cartera está en el servidor.** DigitalOcean puede acceder a sus discos; con
  `-Respaldar` se guarda una copia en este equipo.

## Qué se probó (01-10-2026)

- La imagen se construye y los tests del piloto pasan dentro de ella (164).
- El despliegue completo, contra un servidor simulado (Docker dentro de Docker con
  SSH): HTTPS a través de Caddy, contraseña mala rechazada, guardar la cartera,
  una actualización que recrea la app sin perder la cartera, `-Respaldar` y `-Estado`.
- `preparar` en un Ubuntu 24.04 limpio.

Falta probarlo en un Droplet real: el certificado de Let's Encrypt solo se
obtiene con una dirección pública.
