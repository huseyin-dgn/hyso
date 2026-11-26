[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(ParameterSetName = 'Status')][Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][switch]$Status,
    [Parameter(ParameterSetName = 'Stop')][switch]$Stop,
    [Parameter(ParameterSetName = 'Prepare')][Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][switch]$PrepareOnly,
    [Parameter(ParameterSetName = 'Support')][switch]$SupportScope,
    [Parameter(ParameterSetName = 'Cuda', Mandatory = $true)][switch]$CudaScope,
    [Parameter(ParameterSetName = 'User', Mandatory = $true)][switch]$UserAcceptance,
    [Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][switch]$Resume,
    [Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][string[]]$Revalidate
)
$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$runtimeFile = Join-Path $repoRoot 'outputs/release-acceptance-20261125/source-runtime.json'
$runtimeInfo = Get-Content -LiteralPath $runtimeFile -Raw -Encoding UTF8 | ConvertFrom-Json
$campaignPython = [string]$runtimeInfo.python
if (-not (Test-Path -LiteralPath $campaignPython -PathType Leaf)) {
    # Management remains usable when a previous temporary CPU environment is gone.
    # The controller will report the missing CPU runtime; it never substitutes GPU tests.
    $campaignPython = (Get-Command python -ErrorAction Stop).Source
}
$controllerPath = Join-Path $repoRoot 'tools/final_acceptance/campaign_controller_main.py'
if ($SupportScope) { $controllerPath = Join-Path $repoRoot 'tools/final_acceptance/verify_support_scope.py' }
if ($CudaScope) { $controllerPath = Join-Path $repoRoot 'tools/cuda_acceptance/runner.py' }
if ($UserAcceptance) { $controllerPath = Join-Path $repoRoot 'tools/user_acceptance/runner.py' }
$controllerArgs = @('-I', '-B', $controllerPath)
if ($Status) { $controllerArgs += '--status' }
elseif ($Stop) { $controllerArgs += '--stop' }
elseif ($PrepareOnly) { $controllerArgs += '--prepare-only' }
if ($Resume) { $controllerArgs += '--resume' }
foreach ($case in $Revalidate) { $controllerArgs += @('--revalidate', $case) }
& $campaignPython @controllerArgs
exit $LASTEXITCODE
