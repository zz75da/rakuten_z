#!/usr/bin/env pwsh
# ============================================================================
# verify_stack.ps1 - Vérifie pipeline DVC + Docker + (Kubernetes si présent)
# Lance dans le dossier du projet : pwsh ./verify_stack.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$root = "C:\Users\zobir\DScientest\rakuten_mlops_services"
Set-Location $root

function Write-Section($t) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $t -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Test-Tcp($svc, $hostname, $port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $iar = $c.BeginConnect($hostname, $port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(2000, $false)
        if ($ok -and $c.Connected) { $c.Close(); return $true } else { $c.Close(); return $false }
    } catch { return $false }
}

function Test-Http($name, $url) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Host ("  [OK]   {0,-15} {1}  (HTTP {2})" -f $name, $url, $r.StatusCode) -ForegroundColor Green
        return $true
    } catch {
        Write-Host ("  [DOWN] {0,-15} {1}  ({2})" -f $name, $url, $_.Exception.Message.Split("`n")[0]) -ForegroundColor Red
        return $false
    }
}

# ----------------------------------------------------------------------------
# 1. Docker daemon
# ----------------------------------------------------------------------------
Write-Section "1. DOCKER DAEMON"
$dockerOk = $false
try {
    $info = docker info --format '{{.ServerVersion}}' 2>$null
    if ($LASTEXITCODE -eq 0 -and $info) {
        Write-Host "  [OK] Docker daemon en ligne (version $info)" -ForegroundColor Green
        $dockerOk = $true
    } else {
        Write-Host "  [DOWN] Docker daemon inaccessible. Lance Docker Desktop." -ForegroundColor Red
    }
} catch {
    Write-Host "  [DOWN] Commande 'docker' introuvable." -ForegroundColor Red
}

# ----------------------------------------------------------------------------
# 2. Conteneurs docker-compose
# ----------------------------------------------------------------------------
Write-Section "2. CONTENEURS docker-compose"
$expected = @("postgres","minio","airflow","airflow-init","gate-api",
              "train-api","predict-api","prometheus","grafana","streamlit",
              "pushgateway","alertmanager")

if ($dockerOk) {
    $running = docker ps --format "{{.Names}}|{{.Status}}" 2>$null
    $running_names = ($running | ForEach-Object { ($_ -split '\|')[0] })
    foreach ($svc in $expected) {
        $line = $running | Where-Object { $_ -like "$svc|*" } | Select-Object -First 1
        if ($line) {
            $st = ($line -split '\|')[1]
            if ($st -match 'unhealthy') {
                Write-Host ("  [WARN] {0,-15} {1}" -f $svc, $st) -ForegroundColor Yellow
            } elseif ($st -match 'healthy|Up') {
                Write-Host ("  [OK]   {0,-15} {1}" -f $svc, $st) -ForegroundColor Green
            } else {
                Write-Host ("  [???]  {0,-15} {1}" -f $svc, $st) -ForegroundColor Yellow
            }
        } else {
            Write-Host ("  [DOWN] {0,-15} (absent)" -f $svc) -ForegroundColor Red
        }
    }
}

# ----------------------------------------------------------------------------
# 3. Endpoints HTTP exposés
# ----------------------------------------------------------------------------
Write-Section "3. ENDPOINTS HTTP"
Test-Http "airflow"      "http://localhost:8080/login/"   | Out-Null
Test-Http "mlflow-dagshub" "https://dagshub.com/zz75da/rakuten_z.mlflow" | Out-Null
Test-Http "gate-api"     "http://localhost:5004/health"   | Out-Null
Test-Http "train-api"    "http://localhost:5002/health"   | Out-Null
Test-Http "predict-api"  "http://localhost:5003/health"   | Out-Null
Test-Http "prometheus"   "http://localhost:9090/-/healthy"| Out-Null
Test-Http "grafana"      "http://localhost:3000/api/health"| Out-Null
Test-Http "streamlit"    "http://localhost:8501/"         | Out-Null
Test-Http "minio"        "http://localhost:9001/"         | Out-Null
Test-Http "pushgateway"  "http://localhost:9091/"         | Out-Null
Test-Http "alertmanager" "http://localhost:9093/"         | Out-Null

# ----------------------------------------------------------------------------
# 4. Pipeline DVC
# ----------------------------------------------------------------------------
Write-Section "4. PIPELINE DVC"
try {
    $dvcVer = dvc --version 2>$null
    Write-Host "  [OK] DVC $dvcVer" -ForegroundColor Green
    Write-Host "  - dvc status (workspace) :" -ForegroundColor Yellow
    dvc status 2>&1 | ForEach-Object { Write-Host "      $_" }
    Write-Host "  - dvc status -c (remote) :" -ForegroundColor Yellow
    dvc status -c 2>&1 | ForEach-Object { Write-Host "      $_" }
} catch {
    Write-Host "  [DOWN] DVC non installé ou non configuré" -ForegroundColor Red
}

# ----------------------------------------------------------------------------
# 5. Kubernetes
# ----------------------------------------------------------------------------
Write-Section "5. KUBERNETES"
$hasManifests = (Get-ChildItem -Recurse -Include *.yaml,*.yml -ErrorAction SilentlyContinue |
                  Select-String -Pattern '^kind:\s*(Deployment|StatefulSet|Service|Pod|Ingress)\b' -List).Count
if ($hasManifests -eq 0) {
    Write-Host "  [INFO] Aucun manifest Kubernetes trouve dans le projet." -ForegroundColor Yellow
    Write-Host "         (pas de dossier k8s/, helm/, charts/ ni de manifest)" -ForegroundColor Yellow
}
try {
    $kctx = kubectl config current-context 2>$null
    if ($LASTEXITCODE -eq 0 -and $kctx) {
        Write-Host "  [OK] kubectl context : $kctx" -ForegroundColor Green
        kubectl cluster-info 2>&1 | Select-Object -First 3 | ForEach-Object { Write-Host "      $_" }
        Write-Host "  - Pods (default) :" -ForegroundColor Yellow
        kubectl get pods 2>&1 | ForEach-Object { Write-Host "      $_" }
    } else {
        Write-Host "  [DOWN] Aucun cluster Kubernetes actif (kubectl sans contexte)." -ForegroundColor Red
    }
} catch {
    Write-Host "  [DOWN] kubectl non installe." -ForegroundColor Red
}

Write-Section "FIN"
Write-Host "Astuce : pour demarrer la stack -> docker compose --env-file .env up -d" -ForegroundColor Cyan
Write-Host "         pour voir les logs       -> docker compose logs -f <service>" -ForegroundColor Cyan
