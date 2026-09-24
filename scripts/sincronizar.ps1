<#
.SYNOPSIS
    Sincroniza las dos copias del proyecto.

.DESCRIPTION
    El proyecto vive en dos sitios con papeles distintos:

      Recolector (PC, C:)        La tarea programada anota alli cada dia en el
                                 diario sellado. Es la unica copia que escribe
                                 en el experimento.
      Copia de trabajo (USB, D:) Donde se trabaja con VS Code. Lleva el fichero
                                 COPIA_DE_TRABAJO.txt, que le impide escribir en
                                 el diario y instalar la tarea.

    -Datos   Trae a la copia de trabajo lo recolectado: diario, pre-registro,
             enmiendas, log, cache de precios y el panel. Siempre del PC al USB,
             nunca al reves. La tarea programada lo hace sola al final de cada
             pasada si el USB esta conectado.

    -Codigo  Lleva al PC el codigo cambiado en la copia de trabajo. Antes pasa
             los tests aqui; despues, alli, pasa los tests y recalcula todas las
             decisiones del diario. Si algo falla, restaura el codigo anterior:
             un codigo que ya no reproduce las decisiones registradas exige
             antes una enmienda al protocolo.

.EXAMPLE
    .\scripts\sincronizar.ps1 -Datos
    .\scripts\sincronizar.ps1 -Codigo
#>
[CmdletBinding()]
param(
    [switch]$Datos,
    [switch]$Codigo,
    [string]$Recolector = "C:\Users\ADMIN\Proyectos\Blockchain y criptoactivos",
    [string]$Copia = "D:\Blockchain y criptoactivos",
    [string]$Python = "C:\Users\ADMIN\.venvs\cryptoquant\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$marker = "COPIA_DE_TRABAJO.txt"
$codeDirs = @("cryptoquant", "tests", "scripts")
$codeFiles = @("config.yaml", "requirements.txt", "README.md", ".gitignore")

function Copy-Tree([string]$From, [string]$To, [string[]]$Extra = @()) {
    # /MIR deja el destino identico al origen. /FFT tolera la resolucion de
    # 2 s de las fechas en FAT32, el sistema de archivos del USB.
    $rc = @($From, $To, "/MIR", "/FFT", "/R:2", "/W:2", "/NP", "/NFL", "/NDL", "/NJH", "/NJS") + $Extra
    & robocopy @rc | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy fallo copiando $From (codigo $LASTEXITCODE)" }
}

function Invoke-Py([string]$Dir, [string[]]$PyArgs) {
    # Los avisos de Python salen por stderr; con "Stop", PowerShell 5.1 los
    # convertiria en un error que corta el script.
    Push-Location $Dir
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Python @PyArgs 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
        Pop-Location
    }
    return @{ Code = $code; Out = @($out | ForEach-Object { "$_" }) }
}

function Copy-Code([string]$From, [string]$To) {
    foreach ($d in $codeDirs) { Copy-Tree (Join-Path $From $d) (Join-Path $To $d) @("/XD", "__pycache__") }
    foreach ($f in $codeFiles) {
        $src = Join-Path $From $f
        if (Test-Path $src) { Copy-Item $src (Join-Path $To $f) -Force }
    }
}

# --- Comprobaciones: cada carpeta tiene que ser lo que dice ser ------------
if (-not ($Datos -or $Codigo)) {
    Write-Output "Indique -Datos o -Codigo. Ayuda: Get-Help .\scripts\sincronizar.ps1 -Detailed"
    exit 1
}
if (-not (Test-Path (Join-Path $Recolector "cryptoquant"))) {
    Write-Output "No se encuentra el recolector en $Recolector"
    exit 1
}
if (Test-Path (Join-Path $Recolector $marker)) {
    Write-Output "ERROR: $Recolector lleva $marker. El recolector no puede ser una copia de trabajo."
    exit 1
}
if (-not (Test-Path $Copia)) {
    Write-Output "La copia de trabajo no esta disponible ($Copia): el USB no esta conectado."
    exit 1
}
if (-not (Test-Path (Join-Path $Copia $marker))) {
    Write-Output "ERROR: $Copia no lleva $marker. Sin el marcador esa carpeta podria escribir"
    Write-Output "su propio diario y bifurcar el experimento, asi que no se sincroniza."
    exit 1
}

# --- Datos: del PC al USB -------------------------------------------------
if ($Datos) {
    Copy-Tree (Join-Path $Recolector "data\forward") (Join-Path $Copia "data\forward")
    Copy-Tree (Join-Path $Recolector "data\cache") (Join-Path $Copia "data\cache")
    $panel = Join-Path $Recolector "reports\dashboard.html"
    if (Test-Path $panel) { Copy-Item $panel (Join-Path $Copia "reports\dashboard.html") -Force }
    # Lo que recolecta la tarea de literatura. /XO: solo lo mas reciente, sin
    # borrar nada, asi no pisa lo que se este editando en el USB.
    $inv = Join-Path $Recolector "investigacion"
    if (Test-Path $inv) {
        & robocopy $inv (Join-Path $Copia "investigacion") /E /XO /FFT /R:2 /W:2 /NP /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy fallo copiando investigacion (codigo $LASTEXITCODE)" }
    }
    Write-Output "Datos traidos a la copia de trabajo: diario, log, cache de precios, panel e investigacion."
}

# --- Codigo: del USB al PC, con red de seguridad --------------------------
if ($Codigo) {
    Write-Output "1/4 Tests en la copia de trabajo..."
    $t = Invoke-Py $Copia @("-m", "pytest", "-q", "-p", "no:cacheprovider")
    Write-Output ("    " + ($t.Out | Select-Object -Last 1))
    if ($t.Code -ne 0) {
        Write-Output "Los tests fallan en la copia de trabajo: no se lleva nada al PC."
        exit 1
    }

    $backup = Join-Path $env:TEMP ("cryptoquant-respaldo-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    Write-Output "2/4 Respaldo del codigo actual del PC en $backup"
    New-Item -ItemType Directory -Path $backup | Out-Null
    Copy-Code $Recolector $backup

    Write-Output "3/4 Copiando el codigo al PC..."
    Copy-Code $Copia $Recolector
    # La carpeta de investigacion viaja tambien: sin borrar nada en el PC y
    # solo lo mas reciente (/XO), para no pisar lo que la tarea recolecto alli.
    $inv = Join-Path $Copia "investigacion"
    if (Test-Path $inv) {
        & robocopy $inv (Join-Path $Recolector "investigacion") /E /XO /FFT /R:2 /W:2 /NP /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy fallo copiando investigacion (codigo $LASTEXITCODE)" }
    }

    Write-Output "4/4 Tests y reproduccion de todas las decisiones del diario en el PC..."
    $t = Invoke-Py $Recolector @("-m", "pytest", "-q", "-p", "no:cacheprovider")
    Write-Output ("    tests: " + ($t.Out | Select-Object -Last 1))
    $r = Invoke-Py $Recolector @("-W", "ignore::UserWarning", "-m", "cryptoquant", "forward", "reproduce", "--last", "0")
    $r.Out | Select-Object -Last 3 | ForEach-Object { Write-Output "    $_" }

    if ($t.Code -ne 0 -or $r.Code -ne 0) {
        Write-Output ""
        Write-Output "Algo falla en el PC: se restaura el codigo anterior."
        Copy-Code $backup $Recolector
        if ($r.Code -ne 0) {
            Write-Output "Alguna decision registrada ya no se reproduce con el codigo nuevo."
            Write-Output "Si el cambio es intencionado, necesita antes una enmienda al protocolo."
        }
        exit 1
    }
    Write-Output ""
    Write-Output "Codigo llevado al PC. La tarea programada lo usara en su proxima pasada."
}
exit 0
