# Security policy

## Supported versions

Security fixes are applied to the latest release and the default branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
put credentials, exploit details, private datasets, checkpoint URLs, session
traces, or customer data in a public issue.

Include the affected version or commit, deployment mode, impact, reproduction
steps, and any suggested mitigation. Remove or rotate any credential that may
have been exposed before sending the report.

## Credential and data boundaries

Tuner MCP reads the Tinker API key and HTTP bearer token from environment-backed
configuration. They must never be committed to the repository or included in
logs. Dataset paths are restricted to configured roots, HTTP mode requires
authentication, and artifact reads are bounded and confined to recorded runs.

Only expose the HTTP endpoint on a trusted host or network. Review plans before
starting paid work, use dedicated service credentials, and apply the smallest
permissions and retention period suitable for the deployment.
