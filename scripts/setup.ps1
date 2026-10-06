# Окружение для четырёх задач. TensorFlow 2.21 на Windows требует Python 3.13.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

py -3.13 -m venv .venv
$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install torch torchvision
}
& $python -m pip install -r requirements.txt
Write-Host "Готово. Дальше: .\.venv\Scripts\python.exe -m ml_tasks demo"
