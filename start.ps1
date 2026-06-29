# start.ps1 – One-click launcher for the GBERT Detector service
# Usage: .\start.ps1

# Kill anything on port 8000
$proc = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
if ($proc) {
    Stop-Process -Id $proc -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped previous process on port 8000 (PID $proc)" -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}

# Kill anything on port 8080 (static UI server)
$proc2 = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue |
         Select-Object -First 1 -ExpandProperty OwningProcess
if ($proc2) {
    Stop-Process -Id $proc2 -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped previous process on port 8080 (PID $proc2)" -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "=== Starting GBERT Detector ===" -ForegroundColor Cyan
Write-Host "  API  -> http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  UI   -> http://127.0.0.1:8080" -ForegroundColor Green
Write-Host "  Docs -> http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host ""

# Start static file server for the UI in background
Start-Process -NoNewWindow -FilePath ".\venv\Scripts\python.exe" `
    -ArgumentList "-m http.server 8080 --directory ."

# Open browser
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:8080"

# Start API server (blocking – logs appear here)
Write-Host "Starting API server (model loading takes ~10s)..." -ForegroundColor Yellow
.\venv\Scripts\python.exe server.py
