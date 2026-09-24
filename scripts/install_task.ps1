<#
.SYNOPSIS
    Instala, consulta o elimina la tarea programada del forward test.

.DESCRIPTION
    Registra una tarea en el Programador de tareas de Windows que ejecuta
    `daily_record.ps1` cada 6 horas. La barra diaria cierra a las 00:00 UTC y
    el diario rechaza fechas duplicadas, asi que en la practica se anota una
    vez al dia y las demas ejecuciones no hacen nada.

    Se repite cada 6 horas a proposito, no por redundancia inutil: si el equipo
    estaba apagado o sin red a la hora prevista, el siguiente intento recupera
    la anotacion del dia. Los dias perdidos NO se rellenan despues -- eso seria
    un backtest disfrazado.

.PARAMETER Install
    Registra la tarea (o la reemplaza si ya existe).

.PARAMETER Uninstall
    Elimina la tarea. No toca el diario ni el pre-registro.

.PARAMETER Status
    Muestra el estado, la ultima ejecucion y la proxima.

.PARAMETER RunNow
    Lanza la tarea inmediatamente, sin esperar al disparador.

.EXAMPLE
    .\scripts\install_task.ps1 -Install
    .\scripts\install_task.ps1 -Status
#>
[CmdletBinding(DefaultParameterSetName = "Status")]
param(
    [Parameter(ParameterSetName = "Install")][switch]$Install,
    [Parameter(ParameterSetName = "Uninstall")][switch]$Uninstall,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "RunNow")][switch]$RunNow,
    [string]$TaskName = "CryptoQuant-ForwardRecord",
    [string]$StartTime = "01:00",
    [int]$RepeatHours = 6
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $PSScriptRoot "daily_record.ps1"

function Get-Task {
    try { return Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop }
    catch { return $null }
}

# --------------------------------------------------------------------------
if ($Install) {
    if (Test-Path (Join-Path $projectDir "COPIA_DE_TRABAJO.txt")) {
        Write-Output "Esta carpeta es la copia de trabajo: la tarea no se instala aqui."
        Write-Output "Instalela desde la que recolecta: C:\Users\ADMIN\Proyectos\Blockchain y criptoactivos"
        exit 1
    }
    if (-not (Test-Path $scriptPath)) {
        Write-Error "no se encuentra $scriptPath"; exit 1
    }

    # Ventana oculta: con la visible, cerrarla por error cortaba la anotacion a
    # medias (0xC000013A el 2026-09-21).
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`"" `
        -WorkingDirectory $projectDir

    # Disparador diario, repetido cada N horas durante todo el dia.
    $trigger = New-ScheduledTaskTrigger -Daily -At $StartTime
    $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At $StartTime `
        -RepetitionInterval (New-TimeSpan -Hours $RepeatHours) `
        -RepetitionDuration (New-TimeSpan -Days 1)).Repetition

    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
        -MultipleInstances IgnoreNew

    # LogonType Interactive: no pide contrasena ni permisos de administrador.
    # A cambio, la tarea solo corre con la sesion iniciada; por eso los
    # reintentos cada 6 horas y StartWhenAvailable.
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Limited

    if (Get-Task) {
        Write-Output "La tarea ya existe; se reemplaza."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal `
        -Description "Registra la decision diaria del forward test de cryptoquant." | Out-Null

    Write-Output ""
    Write-Output "Tarea '$TaskName' instalada."
    Write-Output "  Script    : $scriptPath"
    Write-Output "  Proyecto  : $projectDir"
    Write-Output "  Horario   : diario a las $StartTime, repitiendo cada $RepeatHours h"
    Write-Output "  Ejecuta   : con la sesion de $env:USERNAME iniciada"
    Write-Output ""
    Write-Output "Comprobar : .\scripts\install_task.ps1 -Status"
    Write-Output "Probar    : .\scripts\install_task.ps1 -RunNow"
    Write-Output "Quitar    : .\scripts\install_task.ps1 -Uninstall"
    exit 0
}

# --------------------------------------------------------------------------
if ($Uninstall) {
    if (-not (Get-Task)) { Write-Output "No hay ninguna tarea '$TaskName'."; exit 0 }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Tarea '$TaskName' eliminada."
    Write-Output "El diario y el pre-registro siguen intactos."
    exit 0
}

# --------------------------------------------------------------------------
if ($RunNow) {
    if (-not (Get-Task)) { Write-Error "No hay ninguna tarea '$TaskName'."; exit 1 }
    Start-ScheduledTask -TaskName $TaskName
    Write-Output "Lanzada. El resultado tarda un par de minutos en aparecer en:"
    Write-Output "  $projectDir\data\forward\record.log"
    exit 0
}

# --------------------------------------------------------------------------
# Status (por defecto)
$task = Get-Task
if (-not $task) {
    Write-Output "La tarea '$TaskName' NO esta instalada."
    Write-Output "Instalar con: .\scripts\install_task.ps1 -Install"
    exit 1
}

$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Output "Tarea '$TaskName'"
Write-Output ("-" * 60)
Write-Output "  Estado          : $($task.State)"
Write-Output "  Ultima ejecucion: $($info.LastTaskResult) el $($info.LastRunTime)"
Write-Output "  Proxima         : $($info.NextRunTime)"
Write-Output "  Ejecuciones     : $($info.NumberOfMissedRuns) perdidas"

$log = Join-Path $projectDir "data\forward\record.log"
if (Test-Path $log) {
    Write-Output ""
    Write-Output "Ultimas lineas del log"
    Write-Output ("-" * 60)
    Get-Content $log -Tail 10 | ForEach-Object { Write-Output "  $_" }
} else {
    Write-Output ""
    Write-Output "  (aun no hay log: la tarea no se ha ejecutado nunca)"
}
exit 0
