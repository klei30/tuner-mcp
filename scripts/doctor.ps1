$ErrorActionPreference = 'Stop'
$tunerProject = Split-Path $PSScriptRoot -Parent
$tunerDockerVersion = docker info --format '{{.ServerVersion}}'
if ($LASTEXITCODE -ne 0 -or -not ($tunerDockerVersion -join '').Trim()) {
    throw 'Docker engine is not ready or could not be reached.'
}
Write-Output $tunerDockerVersion
docker inspect --format '{{.Name}} running={{.State.Running}} image={{.Image}}' tuner-mcp-http tuner-test-redis
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Tuner and Redis containers.' }
docker exec tuner-test-redis redis-cli ping
if ($LASTEXITCODE -ne 0) { throw 'Redis is not responding.' }
$tunerPrevious = $env:TUNER_MCP_AUTH_TOKEN
try {
    $tunerAuthHeaders = & "$PSScriptRoot/codex_headers.ps1" | ConvertFrom-Json
    $env:TUNER_MCP_AUTH_TOKEN = $tunerAuthHeaders.Authorization.Substring(7)
    & "$tunerProject/.venv/Scripts/python.exe" "$PSScriptRoot/check_http.py"
    if ($LASTEXITCODE -ne 0) { throw 'Authenticated MCP protocol check failed.' }
} finally {
    $env:TUNER_MCP_AUTH_TOKEN = $tunerPrevious
}
