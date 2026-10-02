<#
.SYNOPSIS
    Publica el piloto de riesgo en un servidor propio (un Droplet de DigitalOcean) con Docker.

.DESCRIPTION
    Todo va por SSH con la clave de este equipo. En el servidor, la app vive en
    /opt/piloto y la cartera en un volumen de Docker que sobrevive a cada
    despliegue. Guia paso a paso: deploy/LEEME.md.

    Sin opciones, lleva el codigo de esta copia al servidor y reconstruye la app.
    La primera vez pide la clave para entrar a la app.

    -Preparar      Antes del primer despliegue: Docker, cortafuegos e intercambio.
    -Clave         Cambia la clave de la app.
    -Dominio       La direccion publica. Por defecto, <ip-con-guiones>.sslip.io.
    -SubirCartera  Copia al servidor la cartera de este equipo (data/piloto), sin las velas.
    -Respaldar     Trae a data/piloto/respaldos_servidor una copia de la cartera del servidor.
    -Estado        Muestra los contenedores y las ultimas lineas del registro.
    -CarpetaCartera  Otra carpeta para -SubirCartera y -Respaldar (por ejemplo, una demo).

.EXAMPLE
    .\scripts\desplegar.ps1 -Servidor 203.0.113.10 -Preparar
.EXAMPLE
    .\scripts\desplegar.ps1 -Servidor 203.0.113.10
.EXAMPLE
    .\scripts\desplegar.ps1 -Servidor 203.0.113.10 -Dominio piloto.midominio.com
.EXAMPLE
    .\scripts\desplegar.ps1 -Servidor 203.0.113.10 -Respaldar
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Servidor,
    [string]$Usuario = "root",
    [int]$PuertoSsh = 22,
    [string]$Dominio,
    [switch]$Preparar,
    [switch]$Clave,
    [switch]$SubirCartera,
    [switch]$Respaldar,
    [switch]$Estado,
    [string]$CarpetaCartera,
    # La derivada ya hecha (python -m cryptoquant piloto --crear-clave), sin preguntar la clave.
    [string]$ClaveDerivada,
    # Opciones extra para ssh y scp, por ejemplo @("-i", "C:\ruta\otra_clave").
    [string[]]$OpcionesSsh = @(),
    [string]$Python = "C:\Users\ADMIN\.venvs\cryptoquant\Scripts\python.exe"
)

# Los programas externos (ssh, scp, tar) avisan por stderr; con "Stop", PowerShell
# 5.1 cortaria el script. Sus fallos se miran en $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$raiz = Split-Path $PSScriptRoot -Parent
if (-not $CarpetaCartera) { $CarpetaCartera = Join-Path $raiz "data\piloto" }
$destino = "$Usuario@$Servidor"
$remoto = "/opt/piloto"
$sshOpc = @("-p", "$PuertoSsh", "-o", "StrictHostKeyChecking=accept-new") + $OpcionesSsh
$scpOpc = @("-P", "$PuertoSsh", "-o", "StrictHostKeyChecking=accept-new", "-q") + $OpcionesSsh

function Invoke-Remoto([string]$Orden) {
    & ssh @sshOpc $destino $Orden
    if ($LASTEXITCODE -ne 0) { throw "El servidor devolvio el codigo $LASTEXITCODE en: $Orden" }
}

function Send-Archivo([string]$Local, [string]$Remoto) {
    & scp @scpOpc $Local "${destino}:$Remoto"
    if ($LASTEXITCODE -ne 0) { throw "No se pudo copiar $Local al servidor." }
}

function Receive-Archivo([string]$Remoto, [string]$Local) {
    & scp @scpOpc "${destino}:$Remoto" $Local
    if ($LASTEXITCODE -ne 0) { throw "No se pudo traer $Remoto del servidor." }
}

function Write-ConLF([string]$Ruta, [string]$Texto) {
    # bash y Docker, en el servidor, necesitan LF; git deja CRLF en Windows.
    [IO.File]::WriteAllText($Ruta, ($Texto -replace "`r`n", "`n"), (New-Object Text.UTF8Encoding $false))
}

$tmp = Join-Path $env:TEMP ("piloto-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmp -ErrorAction Stop | Out-Null
try {
    # El script del servidor viaja siempre: asi esta al dia con esta copia.
    $sh = Join-Path $tmp "servidor.sh"
    Write-ConLF $sh ([IO.File]::ReadAllText((Join-Path $raiz "deploy\servidor.sh")))
    Invoke-Remoto "mkdir -p $remoto && chmod 700 $remoto"
    Send-Archivo $sh "$remoto/servidor.sh"

    if ($Estado) {
        Invoke-Remoto "bash $remoto/servidor.sh estado"
        return
    }

    if ($Respaldar) {
        Invoke-Remoto "bash $remoto/servidor.sh respaldar"
        $dir = Join-Path $CarpetaCartera "respaldos_servidor"
        New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
        $archivo = Join-Path $dir ("cartera-servidor-{0:yyyyMMdd-HHmmss}.tgz" -f (Get-Date))
        Receive-Archivo "$remoto/respaldo.tgz" $archivo
        Write-Output "Copia de la cartera del servidor en $archivo"
        return
    }

    if ($Preparar) {
        Write-Output "Preparando el servidor: Docker, cortafuegos e intercambio (unos minutos)..."
        Invoke-Remoto "bash $remoto/servidor.sh preparar"
    }

    # Direccion y clave, en /opt/piloto/piloto.env (solo root puede leerlo).
    $actual = @{}
    foreach ($linea in (& ssh @sshOpc $destino "cat $remoto/piloto.env 2>/dev/null || true")) {
        if ($linea -match '^(\w+)=(.*)$') { $actual[$Matches[1]] = $Matches[2].Trim() }
    }
    if (-not $Dominio) {
        if ($actual["PILOTO_DOMINIO"]) { $Dominio = $actual["PILOTO_DOMINIO"] }
        elseif ($Servidor -match '^\d{1,3}(\.\d{1,3}){3}$') { $Dominio = ($Servidor -replace '\.', '-') + ".sslip.io" }
        else { $Dominio = $Servidor }
    }
    $derivada = $actual["PILOTO_CLAVE_HASH"]
    if ($ClaveDerivada) {
        $derivada = $ClaveDerivada.Trim()
    } elseif ($Clave -or -not $derivada) {
        Write-Output "Elija la clave para entrar a la app. No sale de este equipo: el servidor solo recibe su derivada."
        Push-Location $raiz
        try { $derivada = & $Python -m cryptoquant piloto --crear-clave | Select-Object -Last 1 }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "No se creo la clave." }
    }
    if ($derivada -notmatch '^pbkdf2_sha256:\d+:[0-9a-f]+:[0-9a-f]+$') { throw "La derivada de la clave no es valida." }
    if ($Dominio -ne $actual["PILOTO_DOMINIO"] -or $derivada -ne $actual["PILOTO_CLAVE_HASH"]) {
        $archivoEnv = Join-Path $tmp "piloto.env"
        Write-ConLF $archivoEnv "PILOTO_DOMINIO=$Dominio`nPILOTO_CLAVE_HASH=$derivada`n"
        Send-Archivo $archivoEnv "$remoto/piloto.env"
        Invoke-Remoto "chmod 600 $remoto/piloto.env"
    }

    # El codigo de esta copia: el paquete y deploy/ (este, con LF).
    $etapa = Join-Path $tmp "etapa"
    New-Item -ItemType Directory -Force -Path (Join-Path $etapa "deploy") -ErrorAction Stop | Out-Null
    foreach ($f in Get-ChildItem (Join-Path $raiz "deploy") -File) {
        Write-ConLF (Join-Path $etapa "deploy\$($f.Name)") ([IO.File]::ReadAllText($f.FullName))
    }
    $tgz = Join-Path $tmp "piloto.tgz"
    & tar -czf $tgz --exclude=__pycache__ --exclude=*.pyc -C $raiz cryptoquant -C $etapa deploy
    if ($LASTEXITCODE -ne 0) { throw "No se pudo empaquetar el codigo." }
    Send-Archivo $tgz "$remoto/piloto.tgz"
    Write-Output "Construyendo y arrancando la app en el servidor (la primera vez, unos minutos)..."
    Invoke-Remoto "bash $remoto/servidor.sh instalar"

    if ($SubirCartera) {
        $archivos = @("cartera.json", "fotos.jsonl") | Where-Object { Test-Path (Join-Path $CarpetaCartera $_) }
        if (-not $archivos) { throw "No hay ninguna cartera guardada en $CarpetaCartera." }
        $cartera = Join-Path $tmp "cartera.tgz"
        & tar -czf $cartera -C $CarpetaCartera @archivos
        if ($LASTEXITCODE -ne 0) { throw "No se pudo empaquetar la cartera." }
        Send-Archivo $cartera "$remoto/cartera.tgz"
        Invoke-Remoto "bash $remoto/servidor.sh restaurar"
    }

    $url = "https://$Dominio"
    if ($Dominio -eq "localhost") {
        Write-Output "Listo (prueba local): $url"
        return
    }
    # La primera vez, Caddy tarda unos segundos en obtener el certificado.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $responde = $false
    foreach ($intento in 1..24) {
        try {
            $r = Invoke-WebRequest "$url/_stcore/health" -UseBasicParsing -TimeoutSec 10
            if ($r.Content -match "ok") { $responde = $true; break }
        } catch { Start-Sleep -Seconds 5 }
    }
    if ($responde) {
        Write-Output "Listo: abra $url y entre con su clave."
    } else {
        Write-Output "La app esta en marcha, pero $url aun no responde con HTTPS."
        Write-Output "Revise el registro: .\scripts\desplegar.ps1 -Servidor $Servidor -Estado"
    }
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
