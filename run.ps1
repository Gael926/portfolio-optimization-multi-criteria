$port = 8501
$maxPort = 8510

Write-Host "--- Finance App Docker Runner ---" -ForegroundColor Cyan

while ($port -le $maxPort) {
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($null -eq $connection) {
        break
    }
    Write-Host "Port $port is already in use, trying next one..." -ForegroundColor Yellow
    $port++
}

if ($port -gt $maxPort) {
    Write-Host "Error: No free ports found between 8501 and $maxPort. Please stop some containers." -ForegroundColor Red
    exit 1
}

Write-Host "Starting application on http://localhost:$port" -ForegroundColor Green
$env:HOST_PORT = $port
docker compose up --build
