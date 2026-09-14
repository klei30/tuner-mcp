# Deploy an already-built candidate, preserving external state and a rollback image.
$ErrorActionPreference = 'Stop'
$tunerProject = Split-Path $PSScriptRoot -Parent
$tunerBackup = 'tuner-mcp:backup-' + (Get-Date -Format 'yyyyMMddHHmmss')
$tunerOldKey = $env:TINKER_API_KEY
$tunerOldToken = $env:TUNER_AUTH_TOKEN
$tunerReplaced = $false
try {
    $tunerUserKey = [Environment]::GetEnvironmentVariable('TINKER_API_KEY', 'User')
    if ($tunerUserKey) { $env:TINKER_API_KEY = $tunerUserKey }
    $tunerHeaders = & "$PSScriptRoot/codex_headers.ps1" | ConvertFrom-Json
    $env:TUNER_AUTH_TOKEN = $tunerHeaders.Authorization.Substring(7)
    if (-not $env:TINKER_API_KEY) { throw 'TINKER_API_KEY is not configured.' }
    docker image inspect tuner-mcp:candidate --format '{{.Id}}'
    if ($LASTEXITCODE -ne 0) { throw 'Build tuner-mcp:candidate first.' }
    @'
import json, sqlite3
db = sqlite3.connect('file:/var/lib/tuner/runs/control.sqlite3?mode=ro', uri=True)
runs = [json.loads(row[0]) for row in db.execute("SELECT payload FROM records WHERE id LIKE 'run_%'")]
active = [r['run_id'] for r in runs if r.get('status') not in {'completed','failed','stopped','interrupted'}]
print(json.dumps({'stored_runs': len(runs), 'active_runs': active}))
raise SystemExit(bool(active))
'@ | docker exec -i tuner-mcp-http /app/.venv/bin/python -
    if ($LASTEXITCODE -ne 0) { throw 'Deployment requires no active runs.' }
    docker exec tuner-test-redis redis-cli ping
    if ($LASTEXITCODE -ne 0) { throw 'Redis must be running.' }
    $tunerCurrentImage = docker inspect tuner-mcp-http --format '{{.Image}}'
    if ($LASTEXITCODE -ne 0 -or -not ($tunerCurrentImage -join '').Trim()) {
        throw 'Could not resolve the current Tuner image.'
    }
    docker tag ($tunerCurrentImage -join '').Trim() $tunerBackup
    if ($LASTEXITCODE -ne 0) { throw 'Could not preserve the rollback image.' }
    docker tag tuner-mcp:candidate tuner-mcp:local
    if ($LASTEXITCODE -ne 0) { throw 'Image tagging failed.' }
    docker compose --project-directory $tunerProject stop tuner
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the old server.' }
    $tunerReplaced = $true
    docker compose --project-directory $tunerProject up -d --no-build --force-recreate tuner
    if ($LASTEXITCODE -ne 0) { throw 'Updated server failed to start.' }
    Write-Output "Deployed candidate. Rollback image: $tunerBackup. Run scripts/doctor.ps1 after startup."
} catch {
    if ($tunerReplaced) {
        docker tag $tunerBackup tuner-mcp:local
        docker compose --project-directory $tunerProject up -d --no-build --force-recreate tuner
    }
    throw
} finally {
    $env:TINKER_API_KEY = $tunerOldKey
    $env:TUNER_AUTH_TOKEN = $tunerOldToken
}
