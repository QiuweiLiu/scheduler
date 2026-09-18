# Phase 21 — S_* task-context pipeline repair, pilot (run on the Windows/GPU machine).
#
# Restores ONLY benchmark task metadata (domain / official_task_type / sub_category)
# that the frozen J3:seed11 predictor was trained with but never received at
# scheduler-inference time. The model, checkpoint and weights are NOT touched.
#
# Produces a paired masked/fixed pilot on 16 pre-registered videos and audits
# calibration. No scheduler run is performed here.

$ErrorActionPreference = "Stop"
$PY    = "D:\anaconda\envs\scheduler\python.exe"
$ROOT  = "F:\scheduler"
Set-Location $ROOT

$CFG   = "experiments\EXP-20260911_p9d_j_predictor_acceptance\config.json"
$REG   = "data\manifests\videomme_600_source_v2.jsonl"
$VOC   = "results\processed\j_series_dataset_v1\j_train.jsonl.gz"
$RUNS  = "results\raw\r7_trace_full_20260817"
$SPLIT = "results\processed\r7_workload_20260817\split_manifest_train_validation.jsonl"
$TMPL  = "results\processed\r7_workload_v03_no_run_container\job_templates_r7_v03.jsonl"

$OUT   = "outputs\phase21_taskctx_fix"
$PILOT = "$OUT\pilot_videos.txt"

Write-Host "== [0/5] inputs =="
foreach ($p in @($CFG, $REG, $VOC, $RUNS, $SPLIT, $TMPL, $PILOT)) {
  if (-not (Test-Path $p)) { throw "missing input: $p" }
}
New-Item -ItemType Directory -Force -Path `
  $OUT, "$OUT\anchors_old_masked", "$OUT\anchors_fixed", `
  "$OUT\artifacts_old_masked_full5", "$OUT\artifacts_fixed_full5", "$OUT\audit" | Out-Null

# 1) masked control: same builder, registry fields forced to "unknown"
Write-Host "== [1/5] masked control anchors =="
& $PY scripts\build_sstar_predictor_anchors.py `
  --runs-root $RUNS --manifest $SPLIT --vocab-source $VOC `
  --task-registry $REG --video-allowlist $PILOT --mask-registry-task-context `
  --output "$OUT\anchors_old_masked\features_sstar.jsonl.gz"
if ($LASTEXITCODE -ne 0) { throw "masked anchor build failed" }

# 2) repaired anchors (same builder, registry context restored)
Write-Host "== [2/5] repaired anchors =="
& $PY scripts\build_sstar_predictor_anchors.py `
  --runs-root $RUNS --manifest $SPLIT --vocab-source $VOC `
  --task-registry $REG --video-allowlist $PILOT `
  --output "$OUT\anchors_fixed\features_sstar.jsonl.gz"
if ($LASTEXITCODE -ne 0) { throw "repaired anchor build failed (see the task-context gate output above)" }

# 3) predictor packs: min-steps 5 so no runtime slot is hidden by the length head
Write-Host "== [3/5] packs (frozen J3:seed11 checkpoint) =="
& $PY scripts\pack_j_predictor_artifacts.py --config $CFG `
  --anchors-file "$OUT\anchors_old_masked\features_sstar.jsonl.gz" `
  --output-root "$OUT\artifacts_old_masked_full5" --min-steps 5 --emit-distributions
if ($LASTEXITCODE -ne 0) { throw "masked pack failed" }
& $PY scripts\pack_j_predictor_artifacts.py --config $CFG `
  --anchors-file "$OUT\anchors_fixed\features_sstar.jsonl.gz" `
  --output-root "$OUT\artifacts_fixed_full5" --min-steps 5 --emit-distributions
if ($LASTEXITCODE -ne 0) { throw "repaired pack failed" }

# 4) checkpoint must still be the frozen candidate
Write-Host "== [4/5] checkpoint identity =="
$EXPECTED = "0ee8ded4f92553853026ee24a3c320f21d524f9d2c60de841091430d14949c77"
foreach ($d in @("$OUT\artifacts_old_masked_full5", "$OUT\artifacts_fixed_full5")) {
  $manifest = Get-ChildItem -Path $d -Recurse -Filter *manifest.json | Select-Object -First 1
  if (-not $manifest) { throw "no manifest under $d" }
  $sha = (Get-Content $manifest.FullName -Raw | ConvertFrom-Json).checkpoint_sha256
  if ($sha -ne $EXPECTED) { throw "checkpoint changed in $d : $sha" }
  Write-Host "   $d -> checkpoint OK"
}

# 5) paired calibration audit (this is the pilot gate)
Write-Host "== [5/5] masked vs fixed calibration =="
& $PY scripts\analysis\analysis_phase21_taskctx_calibration.py `
  --fixed "$OUT\artifacts_fixed_full5\prediction_artifacts" `
  --masked "$OUT\artifacts_old_masked_full5\prediction_artifacts" `
  --templates $TMPL --video-allowlist $PILOT `
  --out "$OUT\audit\calibration.json"
if ($LASTEXITCODE -ne 0) { throw "calibration audit failed" }

Write-Host ""
Write-Host "PILOT DONE. Send back these files:"
Write-Host "  $OUT\audit\calibration.json"
Write-Host "  $OUT\anchors_fixed\coverage_report.json"
Write-Host "  $OUT\anchors_old_masked\coverage_report.json"
Write-Host ""
Write-Host "Pre-registered pilot gates:"
Write-Host "  pipeline: registry join 100%, unknown_rate(domain/official_task_type/sub_category) = 0,"
Write-Host "            no all-unknown rows, checkpoint SHA unchanged, anchor ids/prefix hashes unchanged"
Write-Host "  calibration: R50 in [0.70, 1.30]; p50 coverage in [0.45, 0.65];"
Write-Host "            p90 coverage in [0.85, 0.97]; p95 coverage in [0.92, 0.995];"
Write-Host "            mean pinball down >= 10% vs masked; next-role/family not worse by > 2pp"
