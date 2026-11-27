[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Ticket)
$ErrorActionPreference='Stop'
$plan=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'outputs/export-acceptance-20261127/plan.json') -Raw | ConvertFrom-Json
$runtimePython=[string]$plan.runtime.python
if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) { throw 'Admitted Python runtime missing' }
& $runtimePython -I -B (Join-Path $PSScriptRoot 'tools/export_acceptance/dll_diagnostic.py') --run-ticket $Ticket
exit $LASTEXITCODE
