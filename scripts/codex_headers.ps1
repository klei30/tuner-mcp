# Called by Codex's http_headers_helper. Output is a credential, never log it.
$ErrorActionPreference = 'Stop'
$tunerBearer = [Environment]::GetEnvironmentVariable('TUNER_MCP_AUTH_TOKEN', 'User')
if ([string]::IsNullOrWhiteSpace($tunerBearer)) {
    $tunerBearer = [Environment]::GetEnvironmentVariable('TUNER_MCP_AUTH_TOKEN', 'Process')
}
if ([string]::IsNullOrWhiteSpace($tunerBearer)) {
    throw 'TUNER_MCP_AUTH_TOKEN is missing from the Windows user environment.'
}
@{ Authorization = "Bearer $tunerBearer" } | ConvertTo-Json -Compress
