# FinPilot AI — environment bootstrap (Windows PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 [-Python "C:\Path\To\python.exe"]
param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtualenv at $venv"
    & $Python -m venv $venv
}

$vp = Join-Path $venv "Scripts\python.exe"
Write-Host "Upgrading pip and installing requirements..."
& $vp -m pip install --upgrade pip
& $vp -m pip install -r (Join-Path $root "requirements.txt")

Write-Host "Verifying imports..."
& $vp -c "import streamlit, pandas, numpy, plotly, scipy, dotenv, pytest; print('core deps OK')"
Write-Host "Done. Launch with:  .venv\Scripts\streamlit run app.py"
