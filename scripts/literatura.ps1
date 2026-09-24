<#
.SYNOPSIS
    Recolecta libros y antecedentes (investigacion\recolectar.py), a mano o
    como tarea programada semanal.

.DESCRIPTION
    Sin parametros, ejecuta la recoleccion completa (metadatos y PDFs) y deja
    el resultado en investigacion\busquedas\tarea.log.

    La tarea se instala en el PC que recolecta, no en la copia de trabajo del
    USB. Corre cada lunes y, con -Primera, una vez mas en la fecha indicada.
    Si el equipo esta apagado a esa hora, corre al encenderlo.

    Por que semanal y no diaria: OpenAlex da a cada IP un presupuesto gratuito
    diario, y la literatura no cambia de un dia para otro. Las consultas que
    ya se hicieron ese dia salen de la cache y no gastan presupuesto.

.EXAMPLE
    .\scripts\literatura.ps1                                   # ejecutar ahora
    .\scripts\literatura.ps1 -Install -Primera "2026-09-24 08:30"
    .\scripts\literatura.ps1 -Status
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status,
    [string]$Primera,
    [string]$TaskName = "CryptoQuant-Literatura",
    [string]$Python = "C:\Users\ADMIN\.venvs\cryptoquant\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
if ($PSScriptRoot) { $scriptDir = $PSScriptRoot } else { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition }
$projectDir = Split-Path -Parent $scriptDir
$logFile = Join-Path $projectDir "investigacion\busquedas\tarea.log"

function Write-Log([string]$Level, [string]$Message) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss'Z'")
    $dir = Split-Path -Parent $logFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Add-Content -Path $logFile -Value "$stamp [$Level] $Message" -Encoding utf8
    Write-Output "$stamp [$Level] $Message"
}

# --------------------------------------------------------------------------
if ($Install) {
    if (Test-Path (Join-Path $projectDir "COPIA_DE_TRABAJO.txt")) {
        Write-Output "Esta carpeta es la copia de trabajo: la tarea se instala en el PC que recolecta."
        exit 1
    }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptDir\literatura.ps1`"" `
        -WorkingDirectory $projectDir
    $triggers = @(New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:00")
    if ($Primera) { $triggers += New-ScheduledTaskTrigger -Once -At ([datetime]$Primera) }
    # Con una conexion lenta, descargar los PDFs lleva horas.
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 8) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
        -Principal $principal -Description "Recolecta libros y antecedentes para la investigacion de cryptoquant." | Out-Null
    Write-Output "Tarea '$TaskName' instalada: cada lunes a las 09:00$(if ($Primera) { " y una vez el $Primera" })."
    Write-Output "Proxima ejecucion: $((Get-ScheduledTaskInfo -TaskName $TaskName).NextRunTime)"
    exit 0
}
if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Tarea '$TaskName' eliminada. La carpeta investigacion no se toca."
    exit 0
}
if ($Status) {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) { Write-Output "La tarea '$TaskName' no esta instalada."; exit 1 }
    $i = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Output "Estado: $($t.State) | ultima: $($i.LastRunTime) (resultado $($i.LastTaskResult)) | proxima: $($i.NextRunTime)"
    if (Test-Path $logFile) { Get-Content $logFile -Tail 5 }
    exit 0
}

# --------------------------------------------------------------------------
Push-Location $projectDir
$code = 1
try {
    Write-Log "INFO" "recoleccion iniciada"
    $ErrorActionPreference = "Continue"
    $env:PYTHONIOENCODING = "utf-8"
    $out = & $Python -W "ignore::UserWarning" "investigacion\recolectar.py" 2>&1
    $code = $LASTEXITCODE
    $resumen = $out | Where-Object { "$_" -match "^PRISMA" } | Select-Object -Last 1
    if ($code -eq 0) { Write-Log "OK" "$resumen" }
    else {
        Write-Log "FALLO" "codigo $code"
        $out | Select-Object -Last 15 | ForEach-Object { Add-Content -Path $logFile -Value "    $_" -Encoding utf8 }
    }
}
catch {
    Write-Log "EXCEPCION" $_.Exception.Message
    $code = 3
}
finally {
    Pop-Location
}
exit $code
