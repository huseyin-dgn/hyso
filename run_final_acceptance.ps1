[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(ParameterSetName = 'Status')][Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][Parameter(ParameterSetName = 'Product')][Parameter(ParameterSetName = 'Export')][Parameter(ParameterSetName = 'ExportFinal')][switch]$Status,
    [Parameter(ParameterSetName = 'Stop')][switch]$Stop,
    [Parameter(ParameterSetName = 'Prepare')][Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][Parameter(ParameterSetName = 'Product')][Parameter(ParameterSetName = 'Export')][Parameter(ParameterSetName = 'ExportFinal')][switch]$PrepareOnly,
    [Parameter(ParameterSetName = 'Support')][switch]$SupportScope,
    [Parameter(ParameterSetName = 'Cuda', Mandatory = $true)][switch]$CudaScope,
    [Parameter(ParameterSetName = 'User', Mandatory = $true)][switch]$UserAcceptance,
    [Parameter(ParameterSetName = 'Product', Mandatory = $true)][switch]$ProductExperience,
    [Parameter(ParameterSetName = 'Export', Mandatory = $true)][switch]$ExportAcceptance,
    [Parameter(ParameterSetName = 'ExportFinal', Mandatory = $true)][switch]$ExportFinalAcceptance,
    [Parameter(ParameterSetName = 'ExportFinal')][switch]$CpuOnly,
    [Parameter(ParameterSetName = 'ExportFinal')][string]$RuntimeEnvironment = 'hyso_trt',
    [Parameter(ParameterSetName = 'ExportFinal')][string]$RuntimePython,
    [Parameter(ParameterSetName = 'ExportFinal')][string]$AcceptanceRoot,
    [Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][Parameter(ParameterSetName = 'Product')][Parameter(ParameterSetName = 'Export')][Parameter(ParameterSetName = 'ExportFinal')][switch]$Resume,
    [Parameter(ParameterSetName = 'Cuda')][Parameter(ParameterSetName = 'User')][Parameter(ParameterSetName = 'Product')][Parameter(ParameterSetName = 'Export')][Parameter(ParameterSetName = 'ExportFinal')][string[]]$Revalidate
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
if ($ProductExperience) { $controllerPath = Join-Path $repoRoot 'tools/product_acceptance/runner.py' }
if ($ExportAcceptance) { $controllerPath = Join-Path $repoRoot 'tools/export_acceptance/runner.py' }
if ($ExportFinalAcceptance) { $controllerPath = Join-Path $repoRoot 'tools/export_acceptance/runner.py' }
$controllerArgs = @('-I', '-B', $controllerPath)
if ($ExportFinalAcceptance) {
    $controllerArgs += '--final'
    if ($CpuOnly) { $controllerArgs += '--cpu-only' }
    else { $controllerArgs += @('--conda-env', $RuntimeEnvironment) }
    if ($RuntimePython) { $controllerArgs += @('--runtime-python', $RuntimePython) }
    if ($AcceptanceRoot) { $controllerArgs += @('--campaign-root', $AcceptanceRoot) }
}
if ($Status) { $controllerArgs += '--status' }
elseif ($Stop) { $controllerArgs += '--stop' }
elseif ($PrepareOnly) { $controllerArgs += '--prepare-only' }
if ($Resume) { $controllerArgs += '--resume' }
foreach ($case in $Revalidate) { $controllerArgs += @('--revalidate', $case) }
& $campaignPython @controllerArgs
exit $LASTEXITCODE
