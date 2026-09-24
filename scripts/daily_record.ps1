<#
.SYNOPSIS
    Registra la decision diaria del forward test. Pensado para ejecutarse
    desatendido desde el Programador de tareas de Windows.

.DESCRIPTION
    Se ejecuta varias veces al dia a proposito. El diario rechaza fechas
    duplicadas, asi que solo la primera ejecucion posterior al cierre de la
    barra diaria (00:00 UTC) escribe algo; las demas salen sin hacer nada.

    Ese diseno hace la automatizacion tolerante a fallos: si el equipo estaba
    apagado a la hora prevista, el siguiente intento del dia recupera la
    anotacion. Lo que NO hace es rellenar dias pasados -- un backfill seria un
    backtest disfrazado, no evidencia prospectiva.
#>
[CmdletBinding()]
param(
    [string]$ProjectDir,
    [string]$Python = "C:\Users\ADMIN\.venvs\cryptoquant\Scripts\python.exe",
    [int]$MaxLogLines = 2000
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot NO esta disponible al evaluar los valores por defecto de
# param() cuando se invoca con `powershell.exe -File`, que es justo como lo
# llama el Programador de tareas. Resolverlo aqui, ya en el cuerpo del script,
# es lo que hace que funcione tanto lanzado a mano como desatendido.
if (-not $ProjectDir) {
    if ($PSScriptRoot) {
        $scriptDir = $PSScriptRoot
    } else {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    }
    $ProjectDir = Split-Path -Parent $scriptDir
}
$logDir = Join-Path $ProjectDir "data\forward"
$logFile = Join-Path $logDir "record.log"

function Write-Log {
    param([string]$Level, [string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss'Z'")
    $line = "$stamp [$Level] $Message"
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Output $line
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# --- Comprobaciones previas ------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Log "ERROR" "no se encuentra el interprete en $Python"
    exit 2
}
if (-not (Test-Path (Join-Path $ProjectDir "cryptoquant"))) {
    Write-Log "ERROR" "no se encuentra el paquete en $ProjectDir"
    exit 2
}
if (Test-Path (Join-Path $ProjectDir "COPIA_DE_TRABAJO.txt")) {
    Write-Log "ERROR" "esta carpeta es la copia de trabajo: la tarea debe apuntar a la que recolecta"
    exit 2
}

# --- Ejecucion -------------------------------------------------------------
Push-Location $ProjectDir
try {
    Write-Log "INFO" "iniciando registro diario"

    # stderr se une a stdout para que los avisos queden en el log.
    $output = & $Python -W "ignore::UserWarning" -m cryptoquant forward record --if-new 2>&1
    $code = $LASTEXITCODE

    foreach ($line in $output) {
        $text = "$line".TrimEnd()
        if ($text) { Add-Content -Path $logFile -Value "    $text" -Encoding utf8 }
    }

    if ($code -eq 0) {
        $already = $output | Where-Object { "$_" -match "Ya existe la anotacion" }
        if ($already) {
            Write-Log "OK" "sin cambios: la barra de hoy ya estaba registrada"
        } else {
            $seq = ($output | Where-Object { "$_" -match "Anotacion #(\d+)" } |
                    ForEach-Object { $Matches[1] } | Select-Object -First 1)
            Write-Log "OK" "anotacion registrada (#$seq)"
        }
    } else {
        # Salida distinta de cero: red caida, datos sinteticos, cadena rota...
        # No se reintenta aqui; la siguiente ejecucion programada lo hara.
        Write-Log "FALLO" "el comando devolvio codigo $code"
    }
}
catch {
    Write-Log "EXCEPCION" $_.Exception.Message
    $code = 3
}
finally {
    # El panel se regenera SIEMPRE, tambien si el registro fallo: es justo
    # cuando mas importa que muestre el problema. Va con nivel PANEL, que el
    # panel ignora al leer este log; si no, un "regenerado" taparia un FALLO.
    # Nada de lo que pase aqui cambia el codigo de salida del registro.
    try {
        $ErrorActionPreference = "Continue"
        $panel = & $Python -W "ignore::UserWarning" -m cryptoquant dashboard 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "PANEL" "regenerado"
        } else {
            Write-Log "PANEL" "no se pudo regenerar (codigo $LASTEXITCODE)"
            foreach ($line in $panel) {
                $text = "$line".TrimEnd()
                if ($text) { Add-Content -Path $logFile -Value "    $text" -Encoding utf8 }
            }
        }
    }
    catch {
        Write-Log "PANEL" "no se pudo regenerar: $($_.Exception.Message)"
    }

    # Si el USB con la copia de trabajo esta conectado, se le traen los datos
    # nuevos. Nivel COPIA, que el panel tambien ignora; nunca cambia el codigo
    # de salida del registro.
    try {
        $sync = Join-Path $ProjectDir "scripts\sincronizar.ps1"
        if (Test-Path $sync) {
            $ErrorActionPreference = "Continue"
            $res = & $sync -Datos -Recolector $ProjectDir -Python $Python 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Log "COPIA" "copia de trabajo actualizada"
            } else {
                Write-Log "COPIA" ("sin copia: " + "$($res | Select-Object -Last 1)")
            }
        }
    }
    catch {
        Write-Log "COPIA" "sin copia: $($_.Exception.Message)"
    }
    Pop-Location
}

# --- Rotacion del log ------------------------------------------------------
if (Test-Path $logFile) {
    $lines = @(Get-Content $logFile)
    if ($lines.Count -gt $MaxLogLines) {
        $keep = $lines[-$MaxLogLines..-1]
        Set-Content -Path $logFile -Value $keep -Encoding utf8
    }
}

exit $code
