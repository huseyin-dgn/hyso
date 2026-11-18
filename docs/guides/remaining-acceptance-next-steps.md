# Remaining acceptance commands — prepared, not executed

Run in PowerShell at the repository root. Use a new directory for every manifest
and output. These commands never reuse a consumed manifest for execution.

```powershell
$py = 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe'
$entry = Join-Path $PWD 'tools\run_full_acceptance.py'
$tag = [guid]::NewGuid().ToString('N')
$root = Join-Path $env:LOCALAPPDATA "hyso-next-$tag"
```

The lowest-cost read-only P13 action is repeatable recovery into a new directory:

```powershell
$old = Join-Path $env:LOCALAPPDATA 'hyso-p13-native-20261103-174604-e36fd535\run\manifest.json'
& $py -B -m tools.full_acceptance.p13_capture_recover --manifest $old --approved-sha256 a07f887125cce11f5e851d7f781f83fee44da7224554b3741eaba828dee8fabb --output "$root-p13-recovery"
if ($LASTEXITCODE -ne 0) { throw 'Recovery audit failed' }
```

A new P13 capture is necessary to obtain a clean controller terminal. Prepare
only, with the pinned existing tool plan; this grants no GPU authorization:

```powershell
$plan = Join-Path $env:LOCALAPPDATA 'hyso-p13-native-20261103-174604-e36fd535\p13-nsys-plan.json'
$prepared = "$root-p13-prepared"
& $py -B $entry --preflight --continue-independent --p13-nsys-plan $plan --output $prepared
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
```

Only after separately authorizing **this new capture**, the operator can run the
following block. It requests the actual authorization text; none is fabricated:

```powershell
$approval = Read-Host 'Enter explicit approval for this new P13 GPU capture; leave empty to stop'
if ([string]::IsNullOrWhiteSpace($approval)) { throw 'No native authorization' }
$preparedManifest = Join-Path $prepared 'manifest.json'
$preparedSha = (Get-FileHash -LiteralPath $preparedManifest -Algorithm SHA256).Hash.ToLowerInvariant()
$run = "$root-p13-run"
& $py -B $entry --manifest $preparedManifest --approved-sha256 $preparedSha --authorize '06.p13_capture' --approval-text $approval --output $run
if ($LASTEXITCODE -ne 0) { throw 'Authorization failed' }
$manifest = Join-Path $run 'manifest.json'
$runSha = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
& $py -B $entry --manifest $manifest --approved-sha256 $runSha --execute
if ($LASTEXITCODE -ne 2 -and $LASTEXITCODE -ne 0) { throw 'Execution failed; preserve evidence and stop' }
# Exit 2 can mean unapproved unrelated stages. Finalization still requires a
# clean inactive controller, preservation PASS and no finalization errors.
& $py -B -m tools.full_acceptance.p13_capture_finalize --manifest $manifest --approved-sha256 $runSha --output "$root-p13-finalized"
if ($LASTEXITCODE -ne 0) { throw 'Finalization failed; no layer evidence may be claimed' }
$layerPrepared = "$root-layer-prepared"
& $py -B $entry --preflight --continue-independent --p13-evidence "$root-p13-finalized\p13-evidence-configuration.json" --output $layerPrepared
if ($LASTEXITCODE -ne 0) { throw 'Layer preparation failed' }
# Separate explicit approval of the new CPU calibration/layer acceptance:
$layerApproval = Read-Host 'Enter explicit approval for NEW 06.layer; leave empty to stop'
if ([string]::IsNullOrWhiteSpace($layerApproval)) { throw 'No layer authorization' }
$layerManifest = Join-Path $layerPrepared 'manifest.json'
$layerSha = (Get-FileHash -LiteralPath $layerManifest -Algorithm SHA256).Hash.ToLowerInvariant()
$layerRun = "$root-layer-run"
& $py -B $entry --manifest $layerManifest --approved-sha256 $layerSha --authorize '06.layer' --approval-text $layerApproval --output $layerRun
if ($LASTEXITCODE -ne 0) { throw 'Layer authorization failed' }
$layerManifest = Join-Path $layerRun 'manifest.json'
$layerSha = (Get-FileHash -LiteralPath $layerManifest -Algorithm SHA256).Hash.ToLowerInvariant()
& $py -B $entry --manifest $layerManifest --approved-sha256 $layerSha --execute
```

P30 already has local weights/video. Prepare its new contract without opening
either a video decoder or a camera:

```powershell
$config = Join-Path $env:LOCALAPPDATA "hyso-p30-assets\p30-contract-$tag.json"
& $py -B tools/p30_mobilenet_prepare.py --offline --contract-only --config $config
if ($LASTEXITCODE -ne 0) { throw 'Contract preparation failed' }
$videoPrepared = "$root-video-prepared"
& $py -B $entry --preflight --continue-independent --camera-config $config --output $videoPrepared
if ($LASTEXITCODE -ne 0) { throw 'Video preflight failed' }
```

Authorize only `08.video` for recorded-video acceptance with the exact new
manifest/SHA and actual approval text, using the block below. For **separate
physical camera acceptance**, first verify the intended device and task/label
semantics, then create a different config and new preflight:

```powershell
$cameraIdentity = Read-Host 'Actual verified physical camera identity/source (not video provenance)'
if ([string]::IsNullOrWhiteSpace($cameraIdentity)) { throw 'Physical camera prerequisite missing' }
$cameraConfig = Join-Path $env:LOCALAPPDATA "hyso-p30-assets\p30-camera-$tag.json"
& $py -B tools/p30_mobilenet_prepare.py --offline --contract-only --camera-index 0 --camera-provenance $cameraIdentity --config $cameraConfig
if ($LASTEXITCODE -ne 0) { throw 'Camera contract preparation failed' }
$cameraPrepared = "$root-camera-prepared"
& $py -B $entry --preflight --continue-independent --camera-config $cameraConfig --output $cameraPrepared
if ($LASTEXITCODE -ne 0) { throw 'Camera preflight failed' }
```

Execute this authorization block separately for the chosen source. Camera
execution needs its own explicit authorization. Neither block starts P18/P31/P32:

```powershell
# For the camera, change these TWO values to $cameraPrepared and '08.camera'.
$selectedPrepared = $videoPrepared
$selectedStage = '08.video'
$approval = Read-Host "Enter explicit approval for NEW $selectedStage; leave empty to stop"
if ([string]::IsNullOrWhiteSpace($approval)) { throw 'No native authorization' }
$manifest = Join-Path $selectedPrepared 'manifest.json'
$sha = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
$selectedRun = "$selectedPrepared-authorized"
& $py -B $entry --manifest $manifest --approved-sha256 $sha --authorize $selectedStage --approval-text $approval --output $selectedRun
if ($LASTEXITCODE -ne 0) { throw 'Authorization failed' }
$manifest = Join-Path $selectedRun 'manifest.json'
$sha = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
& $py -B $entry --manifest $manifest --approved-sha256 $sha --execute
```

Both successful source scopes remain necessary for P30. To satisfy the current
same-matrix promotion gate in one new run, use the camera-ready config and
explicitly approve `08.video,08.camera` together. Separate runs do not automatically
promote either run's matrix; retain both receipts for an explicit scope review.

P23 has no approved hook-enabled runner. The following command **only validates**
fresh operator-collected telemetry under the existing policy:

```powershell
$telemetryPath = Read-Host 'Absolute path to freshly collected P23 preflight evidence JSON'
$forensicRunId = Read-Host 'Fresh forensic run ID'
$forensicManifestSha = Read-Host 'Exact new forensic manifest SHA256'
& $py -B tools/forensics/p23_preflight.py $telemetryPath --run-id $forensicRunId --manifest-sha256 $forensicManifestSha
```

Do not enable production CUDA hooks, rerun the old control or consumed forensic
manifest, synthesize telemetry, or treat an eligible preflight as permission.
See `p23-forensic-plan.md` for the separate runner/review/authorization blocker.
