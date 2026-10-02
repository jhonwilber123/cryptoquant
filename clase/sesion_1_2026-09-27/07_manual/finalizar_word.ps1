<#
.SYNOPSIS
    Termina el manual con Word: actualiza el indice y los campos, lo guarda y exporta el PDF.
    Lo llama construir_word.py.
#>
param(
    [Parameter(Mandatory)][string]$Entrada,
    [Parameter(Mandatory)][string]$Salida
)
$ErrorActionPreference = "Stop"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $d = $w.Documents.Open($Entrada, $false, $false, $false)
    foreach ($toc in $d.TablesOfContents) { $toc.Update() }
    $d.Fields.Update() | Out-Null
    foreach ($toc in $d.TablesOfContents) { $toc.Update() }
    $d.SaveAs2($Salida, 16)
    $d.ExportAsFixedFormat(($Salida -replace '\.docx$', '.pdf'), 17)
    "Paginas: $($d.ComputeStatistics(2))"
    $d.Close($false)
}
finally {
    $w.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($w)
}
