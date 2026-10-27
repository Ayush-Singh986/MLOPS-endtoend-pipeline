# =========================
# CONFIG
# =========================
$DEPLOYMENT = "breast-cancer-api"
$CONTAINER = "breast-cancer-api"
$IMAGE = "yourdockerhubusername/breast-cancer-api:latest"
$SERVICE_URL = "http://localhost:5000"

Write-Host "=============================="
Write-Host "Starting Kubernetes Deployment"
Write-Host "Image: $IMAGE"
Write-Host "=============================="

# =========================
# STEP 1: Update deployment
# =========================
kubectl set image deployment/$DEPLOYMENT `
    $CONTAINER=$IMAGE

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to set image. Rolling back..."
    kubectl rollout undo deployment/$DEPLOYMENT
    exit 1
}

# =========================
# STEP 2: Wait for rollout
# =========================
kubectl rollout status deployment/$DEPLOYMENT --timeout=120s

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Rollout failed. Rolling back..."
    kubectl rollout undo deployment/$DEPLOYMENT
    kubectl rollout status deployment/$DEPLOYMENT
    exit 1
}

# =========================
# STEP 3: Smoke test API
# =========================
Write-Host "Running smoke tests..."

Start-Sleep -Seconds 10

try {
    $health = Invoke-RestMethod -Uri "$SERVICE_URL/" -Method GET
    $metrics = Invoke-RestMethod -Uri "$SERVICE_URL/metrics" -Method GET

    if (-not $health) {
        throw "Health check failed"
    }

    Write-Host "✅ Health check passed"
}
catch {
    Write-Host "❌ Smoke test failed. Rolling back deployment..."

    kubectl rollout undo deployment/$DEPLOYMENT
    kubectl rollout status deployment/$DEPLOYMENT

    exit 1
}

# =========================
# SUCCESS
# =========================
Write-Host "=============================="
Write-Host "✅ Deployment Successful"
Write-Host "=============================="