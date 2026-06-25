param(
    [int]$Samples = 120,
    [switch]$AutoCapture,
    [double]$Interval = 0.2
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonPath)) {
    $PythonPath = "python"
}

$Letters = @("L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
$DatasetPath = Join-Path $ProjectRoot "dataset\gesture_dataset.csv"

function Get-LabelCount {
    param([string]$Label)

    if (-not (Test-Path $DatasetPath)) {
        return 0
    }

    $Rows = Import-Csv $DatasetPath
    return @($Rows | Where-Object { $_.label -eq $Label }).Count
}

foreach ($Letter in $Letters) {
    $ExistingCount = Get-LabelCount -Label $Letter
    if ($ExistingCount -ge $Samples) {
        Write-Host "$Letter sudah punya $ExistingCount sample, skip."
        continue
    }

    Write-Host "Mulai collect huruf $Letter ($Samples sample)."
    $CollectArgs = @("collect_dataset.py", "--label", $Letter, "--samples", $Samples.ToString())

    if ($AutoCapture) {
        $IntervalText = $Interval.ToString([Globalization.CultureInfo]::InvariantCulture)
        $CollectArgs += @("--auto", "--interval", $IntervalText)
    }

    & $PythonPath @CollectArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Collect huruf $Letter gagal. Proses L-Z dihentikan."
        exit $LASTEXITCODE
    }

    $NewCount = Get-LabelCount -Label $Letter
    if ($NewCount -lt $Samples) {
        Write-Host "Huruf $Letter belum lengkap ($NewCount/$Samples). Proses L-Z dihentikan."
        exit 1
    }
}

Write-Host "Collect L-Z selesai. Training ulang model..."
& $PythonPath "train_model.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training gagal."
    exit $LASTEXITCODE
}

Write-Host "Training selesai. Jalankan aplikasi dengan: .\.venv\Scripts\python.exe app.py"
