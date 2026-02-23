# install.ps1 – PainSense AI one-click setup for Windows + GTX 1650
# Run from inside the painsense_ai\ directory:
#   powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  PainSense AI – Installation Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1 – Python virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment …" -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment already exists." -ForegroundColor Green
}

# Step 2 – Activate venv
Write-Host "[2/4] Activating virtual environment …" -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Step 3 – PyTorch with CUDA 11.8 (GTX 1650 supports CUDA 11.x)
Write-Host "[3/4] Installing PyTorch (CUDA 11.8) …" -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

# Step 4 – Remaining dependencies
Write-Host "[4/4] Installing remaining dependencies …" -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Launch dashboard:" -ForegroundColor White
Write-Host "    python main.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  CLI analysis:" -ForegroundColor White
Write-Host "    python main.py --video path\to\clip.mp4" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Green
