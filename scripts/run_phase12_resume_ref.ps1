<#
.SYNOPSIS
    SatQuery AI - Phase 12 REFERENCE_MAPS RETRY runner. Reference_Maps ONLY.

.DESCRIPTION
    BigEarthNet-S2 and BigEarthNet-S1 are BOTH already complete and independently
    validated:

      BigEarthNet-S2 : 336,000 files / 4,607,680,000 B / 28,000 patch dirs /
                       0 size mismatches / stream bytes exactly 63,251,710,377 /
                       extractor rc 0
      BigEarthNet-S1 : 56,000 files / 3,249,344,000 B / 28,000 patch dirs /
                       0 size mismatches / stream bytes exactly 54,439,153,171 /
                       extractor rc 0

    Both trees are FROZEN and READ-ONLY. This script NEVER schedules them and
    never writes inside them. It reads them only to verify them.

    Only Reference_Maps is extracted. Its previous attempt was TRUNCATED in
    flight by the remote endpoint:

      curl: (18) end of response with 24062821 bytes missing
      -> ReadError: empty header, extractor rc 2
      -> 26,394 of 28,000 files written, 258,328,480 of 282,391,301 stream bytes

    The truncation landed on a clean member boundary, but that does NOT make the
    stage resumable: the archive is a SINGLE NON-SEEKABLE zstd frame, the
    extractor has NO resume/offset/range/checkpoint capability of any kind, and
    dirs_never_seen==0 / stream_bytes_read==expected are whole-run invariants a
    resumed run could never satisfy. Reference_Maps therefore ALWAYS restarts
    from byte 0, and any previous partial tree must be removed by the operator
    BEFORE this script runs.

      STEP 0/6  FROZEN GUARD      fingerprint + verify S2 AND S1 (read-only)
      STEP 1/6  PREFLIGHT         files, dirs-file line count, free disk,
                                  Reference_Maps tree must be absent/empty
      STEP 2/6  MANIFEST IDENTITY manifest hash + config hash + composition
      STEP 3/6  ZENODO PREFLIGHT  DNS + connectivity, BEFORE any extraction
      STEP 4/6  CONFIRMATION
      STEP 5/6  RUN STAGE         ref (Reference_Maps) ONLY, from byte 0
      STEP 6/6  FINAL INTEGRITY   re-fingerprint S2, verify s2 + s1 + ref
      SUMMARY                     run_record.json + summary.txt + integrity.json

    The hardened extractor (.scratch/p12_stream_extract.py) is invoked
    UNCHANGED, with the same arguments, exit codes (0/2/3/4/5) and validation
    semantics as the original launcher. Its curl arguments and retry semantics
    are untouched.

.SAFETY
    Creates only: this run's log directory, the Reference_Maps output tree, and
    its stage log/report.
    DELETES NOTHING and contains no cleanup logic of any kind. There is not a
    single Remove-Item, rd, del or equivalent in this file.
    If the Reference_Maps tree is present and non-empty the script REFUSES to
    start (exit 6) and prints the exact path that must be removed by hand.
    Never writes to Phase 9 artifacts, configs/base.yaml, levir_change_v001, the
    selection manifest, the dirs files, the extraction contract,
    BigEarthNet-S2, or BigEarthNet-S1. Partial output from a failed stage is left
    in place.

.PARAMETER RepoRoot
    Repository root. Default: parent of this script's directory.

.PARAMETER OutRoot
    Extraction target. Default: <repo>\data\bigearthnet_v2\reben

.PARAMETER LogRoot
    Log directory. Default: <repo>\logs\phase12_extraction\resume_ref_<UTC stamp>

.PARAMETER S2GuardLogRoot
    Prior run directory whose s2_report.json validated BigEarthNet-S2.
    Default: auto-detected (newest run whose s2_report.json is complete,
    reports 336,000 files and stream bytes 63,251,710,377).

.PARAMETER S1ReportLogRoot
    Prior run directory whose s1_report.json validated BigEarthNet-S1.
    Default: auto-detected (newest run whose s1_report.json is complete,
    reports 56,000 files and stream bytes 54,439,153,171). NOTE this is a
    DIFFERENT directory from S2GuardLogRoot: the run that completed S2 holds a
    FAILED s1_report.json, so S1 must be attributed to its own successful run.

.PARAMETER CheckOnly
    Run STEP 0..3 and print everything, then exit WITHOUT extracting.
    Costs no bandwidth beyond one HEAD probe.

.PARAMETER SkipConfirm
    Skip the interactive confirmation prompt.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_phase12_resume_ref.ps1 -CheckOnly

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_phase12_resume_ref.ps1

.NOTES
    Exit codes
      0  Reference_Maps validated complete; S2 + S1 + ref all PASS
      1  preflight / manifest / firewall failure
      2  Reference_Maps stage failed (partial output preserved, NOT deleted)
      3  final integrity verification failed
      4  frozen BigEarthNet-S2 or BigEarthNet-S1 guard failed
      5  Zenodo unreachable
      6  Reference_Maps tree is present and non-empty: remove it by hand first
#>
[CmdletBinding()]
param(
    [string] $RepoRoot,
    [string] $OutRoot,
    [string] $LogRoot,
    [string] $S2GuardLogRoot,
    [string] $S1ReportLogRoot,
    [switch] $CheckOnly,
    [switch] $SkipConfirm
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
if (-not $RepoRoot) {
    if ($PSScriptRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
    else { $RepoRoot = (Get-Location).Path }
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

if (-not $OutRoot) { $OutRoot = Join-Path $RepoRoot 'data\bigearthnet_v2\reben' }
$Extractor    = Join-Path $RepoRoot '.scratch\p12_stream_extract.py'
$DirsDir      = Join-Path $RepoRoot '.scratch\p12_extract'
$Pyz          = Join-Path $RepoRoot '.scratch\pyz'
$VenvPy       = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Manifest     = Join-Path $RepoRoot 'artifacts\phase12_selection\selection_manifest_seed10.jsonl'
$RunsDir      = Join-Path $RepoRoot 'logs\phase12_extraction'

$PreflightVerifier = Join-Path $RepoRoot 'scripts\p12_preflight_verify.py'
$IntegrityVerifier = Join-Path $RepoRoot 'scripts\p12_integrity_verify.py'
$FingerprintTool   = Join-Path $RepoRoot '.scratch\p12_tree_fingerprint.py'

if (-not $LogRoot) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $LogRoot = Join-Path $RunsDir "resume_ref_$stamp"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$RunLog        = Join-Path $LogRoot 'run.log'
$PreflightJson = Join-Path $LogRoot 'preflight.json'
$IntegrityJson = Join-Path $LogRoot 'integrity.json'
$RunRecord     = Join-Path $LogRoot 'run_record.json'
$SummaryTxt    = Join-Path $LogRoot 'summary.txt'

$script:RunLog     = $RunLog
$script:StartedUtc = (Get-Date).ToUniversalTime()

# zstandard lives in .scratch/pyz and is injected via PYTHONPATH (the project
# venv's site-packages is deliberately untouched). Set once, inherited by the
# extractor exactly as the original launcher does.
$env:PYTHONPATH = $Pyz

# ----------------------------------------------------------------------------
# curl executable
# ----------------------------------------------------------------------------
# IMPORTANT: in Windows PowerShell 5.1 `curl` is an ALIAS for Invoke-WebRequest,
# so `& curl ...` never reaches the real curl - it would raise a parameter
# binding error and the preflight would wrongly report "zenodo unreachable".
# The extractor launches curl via subprocess.Popen(["curl", ...]), which
# resolves curl.exe on PATH. Resolve that SAME application here, with
# -CommandType Application so an alias can never be returned, so the preflight
# probe exercises the very binary the extraction will use.
$CurlExe = $null
foreach ($cand in @('curl.exe', 'curl')) {
    $cmd = Get-Command $cand -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $CurlExe = $cmd.Source; break }
}

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = '[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        'OK'    { Write-Host $line -ForegroundColor Green }
        default { Write-Host $line }
    }
    Add-Content -Path $script:RunLog -Value $line -Encoding UTF8
}

function Write-Rule {
    param([string]$Char = '=', [int]$Width = 78)
    $line = $Char * $Width
    Write-Host $line
    Add-Content -Path $script:RunLog -Value $line -Encoding UTF8
}

function Invoke-PythonFile {
    <#  Runs a repo Python helper and captures its combined output.
        Returns @{ExitCode; Output}. Creates nothing, deletes nothing. #>
    param([string]$ScriptPath, [string[]]$Arguments, [string]$LogFile)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $all = @()
    $rc  = -1
    try {
        $all = @(& $VenvPy $ScriptPath @Arguments 2>&1)
        $rc = 0
        if (Test-Path -Path variable:LASTEXITCODE) { $rc = $LASTEXITCODE }
    } catch {
        $rc = -1
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($LogFile) { $all | Out-File -FilePath $LogFile -Encoding UTF8 }
    return @{ ExitCode = $rc; Output = ($all | Out-String) }
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json) } catch { return $null }
}

# ----------------------------------------------------------------------------
# Frozen S2 constants (from the validated run; NOT re-derived here)
# ----------------------------------------------------------------------------
$S2Label        = 'BigEarthNet-S2'
$S2ExpectedFiles= 336000
$S2WrittenBytes = 4607680000
$S2PatchDirs    = 28000
$S2StreamBytes  = 63251710377
$S2Tree         = Join-Path $OutRoot 'BigEarthNet-S2'

# ----------------------------------------------------------------------------
# Frozen S1 constants (from the validated run; NOT re-derived here)
# ----------------------------------------------------------------------------
$S1Label        = 'BigEarthNet-S1'
$S1ExpectedFiles= 56000
$S1WrittenBytes = 3249344000
$S1PatchDirs    = 28000
$S1StreamBytes  = 54439153171
$S1Tree         = Join-Path $OutRoot 'BigEarthNet-S1'

# Firewall constants that must not drift
$ExpectedManifestHash = 'fb4d8b4b024a17a842d351e9cfdbb3904648f1883406a5953a41449b375a7843'
$ExpectedConfigHash   = '78f1e3700da15aa1'

# ----------------------------------------------------------------------------
# Stage table - ref ONLY. There is deliberately no way to select s1 or s2.
# Byte counts are the archives' exact Content-Length values.
# ----------------------------------------------------------------------------
$StageDefs = [ordered]@{
    ref = @{
        Label          = 'Reference_Maps'
        Url            = 'https://zenodo.org/api/records/10891137/files/Reference_Maps.tar.zst/content'
        DirsFile       = (Join-Path $DirsDir 'ref_dirs.txt')
        ExpectedFiles  = 28000
        ExpectedBytes  = 282391301
        EstDiskGB      = 0.08
    }
}

$Ordered = @('ref')
if ($Ordered -contains 's2') {
    Write-Host 'INTERNAL ERROR: s2 must never be scheduled by the Reference_Maps retry runner.' -ForegroundColor Red
    exit 1
}
if ($Ordered -contains 's1') {
    Write-Host 'INTERNAL ERROR: s1 must never be scheduled by the Reference_Maps retry runner.' -ForegroundColor Red
    exit 1
}
if ($Ordered.Count -ne 1 -or $Ordered[0] -ne 'ref') {
    Write-Host 'INTERNAL ERROR: ref must be the one and only scheduled stage.' -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
Write-Rule
Write-Host 'SatQuery AI - Phase 12 REFERENCE_MAPS RETRY runner (Reference_Maps only)' -ForegroundColor Cyan
Write-Host 'BigEarthNet-S2 and BigEarthNet-S1 are FROZEN and will NOT be re-extracted.' -ForegroundColor Yellow
Write-Rule
Write-Log "repo root    : $RepoRoot"
Write-Log "output root  : $OutRoot"
Write-Log "log root     : $LogRoot"
Write-Log "stages       : $($Ordered -join ', ')   (s2 and s1 SKIPPED - both already validated)"
Write-Log "check only   : $([bool]$CheckOnly)"
Write-Log "started (UTC): $($script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
Write-Log "PYTHONPATH   : $env:PYTHONPATH"
if ($CurlExe) { Write-Log "curl         : $CurlExe" } else { Write-Log 'curl         : NOT FOUND on PATH (looked for curl.exe)' 'WARN' }
Write-Host ''

# ============================================================================
# STEP 0 - FROZEN GUARD  (S2 + S1; read-only; runs before anything else)
# ============================================================================
Write-Rule '-'
Write-Host 'STEP 0/6  FROZEN S2 + S1 GUARD (read-only)' -ForegroundColor Cyan
Write-Rule '-'

# --- 0a. locate the prior run whose s2_report.json validated BigEarthNet-S2
if ($S2GuardLogRoot) {
    if (-not (Test-Path -LiteralPath $S2GuardLogRoot)) {
        Write-Log "S2 GUARD FAILED: -S2GuardLogRoot does not exist: $S2GuardLogRoot" 'ERROR'
        exit 4
    }
} else {
    Write-Log 'auto-detecting the run that validated BigEarthNet-S2...'
    $cands = @(Get-ChildItem -LiteralPath $RunsDir -Directory -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -like 'run_*' } | Sort-Object Name -Descending)
    foreach ($cd in $cands) {
        $rp = Join-Path $cd.FullName 's2_report.json'
        if (-not (Test-Path -LiteralPath $rp)) { continue }
        $tmp = Read-JsonFile -Path $rp
        if (-not $tmp) { continue }
        if ($tmp.complete -eq $true -and
            $tmp.files_extracted -eq $S2ExpectedFiles -and
            $tmp.size_mismatches -eq 0 -and
            $tmp.stream_bytes_read -eq $S2StreamBytes) {
            $S2GuardLogRoot = $cd.FullName
            break
        }
    }
}

if (-not $S2GuardLogRoot) {
    Write-Log 'S2 GUARD FAILED: no run directory holds a complete, matching s2_report.json.' 'ERROR'
    Write-Log 'Refusing to run: the frozen BigEarthNet-S2 tree cannot be attributed to a validated run.' 'ERROR'
    Write-Log 'Nothing was downloaded and nothing was extracted.' 'ERROR'
    exit 4
}

$S2SourceReport = Join-Path $S2GuardLogRoot 's2_report.json'
$S2SourceSha    = (Get-FileHash -LiteralPath $S2SourceReport -Algorithm SHA256).Hash.ToLower()
Write-Log "s2 source run : $S2GuardLogRoot"
Write-Log "s2 report     : $S2SourceReport"
Write-Log "s2 report sha : $S2SourceSha"

# --- 0a2. locate the prior run whose s1_report.json validated BigEarthNet-S1.
#          This is deliberately a SEPARATE search: the run that completed S2
#          (run_20260919T162727Z) also contains an s1_report.json, but that one
#          is the FAILED S1 attempt (curl 6, 0 files). Attributing S1 to it
#          would make the integrity verifier compare a 56,000-file tree against
#          a 0-file report. The auto-detect below requires a complete report.
if ($S1ReportLogRoot) {
    if (-not (Test-Path -LiteralPath $S1ReportLogRoot)) {
        Write-Log "S1 GUARD FAILED: -S1ReportLogRoot does not exist: $S1ReportLogRoot" 'ERROR'
        exit 4
    }
} else {
    Write-Log 'auto-detecting the run that validated BigEarthNet-S1...'
    $cands1 = @(Get-ChildItem -LiteralPath $RunsDir -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending)
    foreach ($cd in $cands1) {
        $rp = Join-Path $cd.FullName 's1_report.json'
        if (-not (Test-Path -LiteralPath $rp)) { continue }
        $tmp = Read-JsonFile -Path $rp
        if (-not $tmp) { continue }
        if ($tmp.complete -eq $true -and
            $tmp.files_extracted -eq $S1ExpectedFiles -and
            $tmp.size_mismatches -eq 0 -and
            $tmp.stream_bytes_read -eq $S1StreamBytes) {
            $S1ReportLogRoot = $cd.FullName
            break
        }
    }
}

if (-not $S1ReportLogRoot) {
    Write-Log 'S1 GUARD FAILED: no run directory holds a complete, matching s1_report.json.' 'ERROR'
    Write-Log 'Refusing to run: the frozen BigEarthNet-S1 tree cannot be attributed to a validated run.' 'ERROR'
    Write-Log 'Nothing was downloaded and nothing was extracted.' 'ERROR'
    exit 4
}

$S1SourceReport = Join-Path $S1ReportLogRoot 's1_report.json'
$S1SourceSha    = (Get-FileHash -LiteralPath $S1SourceReport -Algorithm SHA256).Hash.ToLower()
Write-Log "s1 source run : $S1ReportLogRoot"
Write-Log "s1 report     : $S1SourceReport"
Write-Log "s1 report sha : $S1SourceSha"

# --- 0b. fingerprint the S2 tree BEFORE anything else touches the disk
Write-Log 'fingerprinting BigEarthNet-S2 (before)...'
$s2FpBeforePath = Join-Path $LogRoot 's2_fingerprint_before.json'
$fpB = Invoke-PythonFile -ScriptPath $FingerprintTool `
        -Arguments @('--root', $S2Tree, '--out', $s2FpBeforePath, '--label', $S2Label) `
        -LogFile (Join-Path $LogRoot 's2_fingerprint_before_stdout.txt')
Write-Log $fpB.Output.Trim()
$s2FpBefore = Read-JsonFile -Path $s2FpBeforePath
if (-not $s2FpBefore) {
    Write-Log 'S2 GUARD FAILED: could not fingerprint BigEarthNet-S2.' 'ERROR'
    exit 4
}

# --- 0b2. fingerprint the S1 tree BEFORE anything else touches the disk
Write-Log 'fingerprinting BigEarthNet-S1 (before)...'
$s1FpBeforePath = Join-Path $LogRoot 's1_fingerprint_before.json'
$fp1B = Invoke-PythonFile -ScriptPath $FingerprintTool `
        -Arguments @('--root', $S1Tree, '--out', $s1FpBeforePath, '--label', $S1Label) `
        -LogFile (Join-Path $LogRoot 's1_fingerprint_before_stdout.txt')
Write-Log $fp1B.Output.Trim()
$s1FpBefore = Read-JsonFile -Path $s1FpBeforePath
if (-not $s1FpBefore) {
    Write-Log 'S1 GUARD FAILED: could not fingerprint BigEarthNet-S1.' 'ERROR'
    exit 4
}

# --- 0c. independent on-disk verification of S2 against the validated report
Write-Log 'running independent on-disk verification of BigEarthNet-S2...'
$s2GuardJson = Join-Path $LogRoot 's2_guard.json'
$g = Invoke-PythonFile -ScriptPath $IntegrityVerifier `
        -Arguments @('--repo', $RepoRoot, '--out-root', $OutRoot, '--log-root', $S2GuardLogRoot, `
                     '--stages', 's2', '--out-json', $s2GuardJson) `
        -LogFile (Join-Path $LogRoot 's2_guard_stdout.txt')
Write-Log $g.Output.Trim()
$guardData = Read-JsonFile -Path $s2GuardJson

# --- 0c2. independent on-disk verification of S1 against ITS validated report
Write-Log 'running independent on-disk verification of BigEarthNet-S1...'
$s1GuardJson = Join-Path $LogRoot 's1_guard.json'
$g1 = Invoke-PythonFile -ScriptPath $IntegrityVerifier `
        -Arguments @('--repo', $RepoRoot, '--out-root', $OutRoot, '--log-root', $S1ReportLogRoot, `
                     '--stages', 's1', '--out-json', $s1GuardJson) `
        -LogFile (Join-Path $LogRoot 's1_guard_stdout.txt')
Write-Log $g1.Output.Trim()
$guard1Data = Read-JsonFile -Path $s1GuardJson

# --- 0d. hard gate
$s2Ok = $false
$s2GuardNotes = @()
if (-not $guardData) {
    $s2GuardNotes += 's2_guard.json not readable'
} elseif ($guardData.ok -ne $true) {
    foreach ($f in @($guardData.failures)) { $s2GuardNotes += "$f" }
} elseif ($guardData.trees.PSObject.Properties.Name -notcontains 's2') {
    $s2GuardNotes += 'guard report has no s2 tree entry'
} else {
    $t = $guardData.trees.s2
    if ($t.present -ne $true)                                        { $s2GuardNotes += 'tree not present' }
    if ($t.files -ne $S2ExpectedFiles)                               { $s2GuardNotes += "files $($t.files) != $S2ExpectedFiles" }
    if ($t.bytes -ne $S2WrittenBytes)                                { $s2GuardNotes += "bytes $($t.bytes) != $S2WrittenBytes" }
    if ($t.patch_dirs -ne $S2PatchDirs)                              { $s2GuardNotes += "patch_dirs $($t.patch_dirs) != $S2PatchDirs" }
    if ($t.bad_patch_dirs -ne 0)                                     { $s2GuardNotes += "bad_patch_dirs $($t.bad_patch_dirs)" }
    if ($t.zero_byte -ne 0)                                          { $s2GuardNotes += "zero_byte $($t.zero_byte)" }
    if ($s2FpBefore.files -ne $S2ExpectedFiles)                      { $s2GuardNotes += "fingerprint files $($s2FpBefore.files) != $S2ExpectedFiles" }
    if ($s2FpBefore.bytes -ne $S2WrittenBytes)                       { $s2GuardNotes += "fingerprint bytes $($s2FpBefore.bytes) != $S2WrittenBytes" }
    if ($s2GuardNotes.Count -eq 0) { $s2Ok = $true }
}

if (-not $s2Ok) {
    Write-Log 'S2 GUARD FAILED - the frozen BigEarthNet-S2 tree is NOT in the validated state:' 'ERROR'
    foreach ($n in $s2GuardNotes) { Write-Log "  - $n" 'ERROR' }
    Write-Log 'Refusing to run. Nothing was downloaded and nothing was extracted.' 'ERROR'
    Write-Log 'Do NOT delete the S2 tree. Review the reports before acting.' 'ERROR'
    exit 4
}
Write-Log ("S2 GUARD PASSED: {0:N0} files, {1:N0} bytes, {2:N0} patch dirs, 0 bad, 0 zero-byte" -f `
    $s2FpBefore.files, $s2FpBefore.bytes, $guardData.trees.s2.patch_dirs) 'OK'
Write-Log ("S2 fingerprint (before): {0}" -f $s2FpBefore.fingerprint_sha256)
Write-Log ("S2 newest file mtime    : {0}" -f $s2FpBefore.newest_mtime_utc)

# --- 0d2. hard gate for S1
$s1Ok = $false
$s1GuardNotes = @()
if (-not $guard1Data) {
    $s1GuardNotes += 's1_guard.json not readable'
} elseif ($guard1Data.ok -ne $true) {
    foreach ($f in @($guard1Data.failures)) { $s1GuardNotes += "$f" }
} elseif ($guard1Data.trees.PSObject.Properties.Name -notcontains 's1') {
    $s1GuardNotes += 'guard report has no s1 tree entry'
} else {
    $t = $guard1Data.trees.s1
    if ($t.present -ne $true)                                        { $s1GuardNotes += 'tree not present' }
    if ($t.files -ne $S1ExpectedFiles)                               { $s1GuardNotes += "files $($t.files) != $S1ExpectedFiles" }
    if ($t.bytes -ne $S1WrittenBytes)                                { $s1GuardNotes += "bytes $($t.bytes) != $S1WrittenBytes" }
    if ($t.patch_dirs -ne $S1PatchDirs)                              { $s1GuardNotes += "patch_dirs $($t.patch_dirs) != $S1PatchDirs" }
    if ($t.bad_patch_dirs -ne 0)                                     { $s1GuardNotes += "bad_patch_dirs $($t.bad_patch_dirs)" }
    if ($t.zero_byte -ne 0)                                          { $s1GuardNotes += "zero_byte $($t.zero_byte)" }
    if ($s1FpBefore.files -ne $S1ExpectedFiles)                      { $s1GuardNotes += "fingerprint files $($s1FpBefore.files) != $S1ExpectedFiles" }
    if ($s1FpBefore.bytes -ne $S1WrittenBytes)                       { $s1GuardNotes += "fingerprint bytes $($s1FpBefore.bytes) != $S1WrittenBytes" }
    if ($s1GuardNotes.Count -eq 0) { $s1Ok = $true }
}

if (-not $s1Ok) {
    Write-Log 'S1 GUARD FAILED - the frozen BigEarthNet-S1 tree is NOT in the validated state:' 'ERROR'
    foreach ($n in $s1GuardNotes) { Write-Log "  - $n" 'ERROR' }
    Write-Log 'Refusing to run. Nothing was downloaded and nothing was extracted.' 'ERROR'
    Write-Log 'Do NOT delete the S1 tree. Review the reports before acting.' 'ERROR'
    exit 4
}
Write-Log ("S1 GUARD PASSED: {0:N0} files, {1:N0} bytes, {2:N0} patch dirs, 0 bad, 0 zero-byte" -f `
    $s1FpBefore.files, $s1FpBefore.bytes, $guard1Data.trees.s1.patch_dirs) 'OK'
Write-Log ("S1 fingerprint (before): {0}" -f $s1FpBefore.fingerprint_sha256)
Write-Log ("S1 newest file mtime    : {0}" -f $s1FpBefore.newest_mtime_utc)
Write-Host ''

# ============================================================================
# STEP 1 - PREFLIGHT  (fail loudly BEFORE anything is downloaded)
# ============================================================================
Write-Rule '-'
Write-Host 'STEP 1/6  PREFLIGHT' -ForegroundColor Cyan
Write-Rule '-'

$preflightFailures = @()

foreach ($f in @($VenvPy, $Extractor, $Manifest, $FingerprintTool, $IntegrityVerifier, $PreflightVerifier)) {
    if (-not (Test-Path -LiteralPath $f)) { $preflightFailures += "missing required file: $f" }
}
foreach ($k in @('ref')) {
    $d = $StageDefs[$k].DirsFile
    if (-not (Test-Path -LiteralPath $d)) { $preflightFailures += "missing dirs file: $d" }
}
if (-not (Test-Path -LiteralPath $Pyz)) { $preflightFailures += "missing zstandard target dir: $Pyz" }
if (-not $CurlExe) { $preflightFailures += "curl executable not found on PATH (looked for curl.exe); the extractor cannot stream without it" }

if ($preflightFailures.Count -gt 0) {
    Write-Log 'PREFLIGHT FAILED - required files are missing. Nothing was downloaded.' 'ERROR'
    foreach ($p in $preflightFailures) { Write-Log "  - $p" 'ERROR' }
    exit 1
}

Write-Log 'dirs-file line counts:'
foreach ($k in @('ref')) {
    $d    = $StageDefs[$k].DirsFile
    $n    = @(Get-Content -LiteralPath $d | Where-Object { $_.Trim() -ne '' }).Count
    $want = 28000
    $okStr = 'OK'
    if ($n -ne $want) { $okStr = "MISMATCH (want $want)"; $preflightFailures += "$($StageDefs[$k].Label) dirs file has $n lines, want $want" }
    Write-Log ("  {0,-18} {1,8} lines  {2}" -f $StageDefs[$k].Label, $n, $okStr)
}

# ---------------------------------------------------------------------------
# --- 1b. Reference_Maps OUTPUT TREE GATE  (exit 6 - fail closed, never clean)
# ---------------------------------------------------------------------------
# Reference_Maps always restarts from byte 0: the archive is a single
# non-seekable zstd frame and the extractor has no resume capability. If a
# previous partial tree is still on disk the extractor would overwrite those
# paths in place, producing a tree whose provenance mixes two runs. This script
# therefore REFUSES to start and asks the operator to remove the tree by hand.
# It contains no deletion logic of any kind - by design.
$refTree    = Join-Path $OutRoot $StageDefs['ref'].Label
$refPresent = Test-Path -LiteralPath $refTree
$refFiles   = 0
$refBytes   = 0
if ($refPresent) {
    $rf = @(Get-ChildItem -LiteralPath $refTree -Recurse -File -ErrorAction SilentlyContinue)
    $refFiles = $rf.Count
    $refBytes = ($rf | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $refBytes) { $refBytes = 0 }
}

Write-Log ("Reference_Maps tree : {0}" -f $refTree)
if (-not $refPresent) {
    Write-Log '  not present - clean slate, Reference_Maps will start from byte 0' 'OK'
} elseif ($refFiles -eq 0) {
    Write-Log '  present but EMPTY (0 files) - acceptable, will start from byte 0' 'OK'
} else {
    Write-Log ("  PRESENT AND NON-EMPTY: {0:N0} files / {1:N0} bytes" -f $refFiles, $refBytes) 'ERROR'
    Write-Log '' 'ERROR'
    Write-Log 'REFUSING TO START. This is deliberate: a byte-0 retry must not write into a' 'ERROR'
    Write-Log 'partial tree left by a previous run, because the resulting tree would mix two' 'ERROR'
    Write-Log 'runs and dirs_never_seen==0 would stop being meaningful evidence.' 'ERROR'
    Write-Log '' 'ERROR'
    Write-Log 'This script will NOT delete it for you. Remove it explicitly, by hand:' 'ERROR'
    Write-Log '' 'ERROR'
    Write-Log ("    Remove-Item -LiteralPath '{0}' -Recurse -Force" -f $refTree) 'ERROR'
    Write-Log '' 'ERROR'
    Write-Log 'Then re-run this script.' 'ERROR'
    Write-Log 'Nothing was downloaded and nothing was extracted.' 'ERROR'
    exit 6
}

# --- free disk
$drive = (Get-Item -LiteralPath $RepoRoot).PSDrive.Name
$freeMB = [math]::Round((Get-PSDrive -Name $drive).Free / 1MB, 0)
Write-Log ("free disk on {0}: {1:N0} MB" -f $drive, $freeMB)
if ($freeMB -lt 10000) { $preflightFailures += "only $freeMB MB free; want >= 10,000 MB" }

if ($preflightFailures.Count -gt 0) {
    Write-Log 'PREFLIGHT FAILED. Nothing was downloaded.' 'ERROR'
    foreach ($p in $preflightFailures) { Write-Log "  - $p" 'ERROR' }
    exit 1
}

# ============================================================================
# STEP 2 - IDENTITY, COMPOSITION, COST  (re-asserted, not assumed)
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 2/6  MANIFEST IDENTITY AND COMPOSITION' -ForegroundColor Cyan
Write-Rule '-'

Write-Log 'running deep preflight (manifest hash, composition, path-set equality, env)...'
$pf = Invoke-PythonFile -ScriptPath $PreflightVerifier `
        -Arguments @('--repo', $RepoRoot, '--out-json', $PreflightJson) `
        -LogFile (Join-Path $LogRoot 'preflight_stdout.txt')
Write-Log $pf.Output.Trim()

$pfData = Read-JsonFile -Path $PreflightJson
if ($pf.ExitCode -ne 0 -or -not $pfData -or $pfData.ok -ne $true) {
    Write-Log 'PREFLIGHT FAILED - see preflight.json. Nothing was downloaded.' 'ERROR'
    exit 1
}

$m = $pfData.manifest
$c = $pfData.composition
$totalBytes = 0
foreach ($k in $Ordered) { $totalBytes += $StageDefs[$k].ExpectedBytes }
$totalFiles = 0
foreach ($k in $Ordered) { $totalFiles += $StageDefs[$k].ExpectedFiles }
$estDiskGB = 0.0
foreach ($k in $Ordered) { $estDiskGB += $StageDefs[$k].EstDiskGB }

Write-Log ("manifest name     : {0}" -f $m.name)
Write-Log ("manifest hash     : {0}" -f $m.declared_hash)
Write-Log ("hash verified     : {0}  (recomputed {1})" -f $m.hash_ok, $m.recomputed_hash)
Write-Log ("records           : {0:N0}" -f $m.count)
Write-Log ("splits            : train {0:N0} / val {1:N0} / test {2:N0}" -f $m.splits.train, $m.splits.val, $m.splits.test)
Write-Log ("T2 scene blocks   : {0}   (impure blocks: {1})" -f $m.scenes, $m.impure_blocks)
Write-Log ("label policy      : {0}" -f $m.label_policy)
Write-Log ("config hash       : {0}" -f $m.config_hash)

# --- firewall assertions: the manifest and config must be the frozen ones
$firewallFailures = @()
if ($m.declared_hash -ne $ExpectedManifestHash) { $firewallFailures += "manifest hash drifted: $($m.declared_hash)" }
if ($m.recomputed_hash -ne $ExpectedManifestHash) { $firewallFailures += "recomputed manifest hash drifted: $($m.recomputed_hash)" }
if ($m.config_hash -ne $ExpectedConfigHash) { $firewallFailures += "config hash drifted: $($m.config_hash)" }
if ($m.count -ne 28000) { $firewallFailures += "manifest record count drifted: $($m.count)" }
if ($firewallFailures.Count -gt 0) {
    Write-Log 'FIREWALL ASSERTION FAILED - refusing to run:' 'ERROR'
    foreach ($f in $firewallFailures) { Write-Log "  - $f" 'ERROR' }
    exit 1
}
Write-Log 'firewall assertions: manifest hash + config hash + record count all match the frozen values' 'OK'
Write-Host ''

Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host '  PATCH COMPOSITION OF THE 28,000-PATCH SLICE' -ForegroundColor Yellow
Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host ("    clean (no snow/cloud/shadow) : {0,7:N0}" -f $c.selected_from_clean) -ForegroundColor Yellow
Write-Host ("    snow / cloud / shadow        : {0,7:N0}" -f $c.selected_from_snow_cloud) -ForegroundColor Yellow
Write-Host ("    total                        : {0,7:N0}" -f $m.count) -ForegroundColor Yellow
Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host '  This is the existing, verified manifest. It has NOT been altered.' -ForegroundColor Yellow
Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host ''

Add-Content -Path $script:RunLog -Value ("composition: clean={0} snow_cloud={1} total={2}" -f $c.selected_from_clean, $c.selected_from_snow_cloud, $m.count) -Encoding UTF8

Write-Host 'STAGE PLAN (resume)' -ForegroundColor Cyan
Write-Log ("  {0,-18} {1,10} {2,18} {3,10}" -f 'stage', 'files', 'stream bytes', 'est. disk')
foreach ($k in $Ordered) {
    $d = $StageDefs[$k]
    Write-Log ("  {0,-18} {1,10:N0} {2,18:N0} {3,9:N2} GB" -f $d.Label, $d.ExpectedFiles, $d.ExpectedBytes, $d.EstDiskGB)
}
Write-Log ("  {0,-18} {1,10:N0} {2,18:N0} {3,9:N2} GB" -f 'TOTAL', $totalFiles, $totalBytes, $estDiskGB)
Write-Log ("  {0,-18} {1,10} {2,18} {3,10}" -f 'BigEarthNet-S2', 'SKIPPED', 'already extracted', 'frozen')
Write-Log ("  {0,-18} {1,10} {2,18} {3,10}" -f 'BigEarthNet-S1', 'SKIPPED', 'already extracted', 'frozen')
Write-Host ''

$bytesMB = $totalBytes / 1MB
$writeHours = ($totalFiles * 21.2 / 1000.0) / 3600.0
Write-Host 'EXPECTED DURATION  (measured rates: 1.16-2.38 MB/s; 21.2 ms/file create cost)' -ForegroundColor Cyan
Write-Log ("  per-file create overhead (measured) : {0:N2} h for {1:N0} files" -f $writeHours, $totalFiles)
$proj = @(
    @{ n = 'slowest measured  (1.16 MB/s)'; r = 1.16  },
    @{ n = 'mean curl         (1.21 MB/s)'; r = 1.214 },
    @{ n = 'mean pipeline     (1.80 MB/s)'; r = 1.799 },
    @{ n = 'fastest measured  (2.38 MB/s)'; r = 2.379 }
)
foreach ($p in $proj) {
    $h = ($bytesMB / $p.r) / 3600.0
    Write-Log ("  {0}  ->  stream {1,6:N1} h  +  writes {2:N1} h  =  {3,6:N1} h" -f $p.n, $h, $writeHours, ($h + $writeHours))
}
Write-Log '  Reference_Maps is a small archive (~269 MB): minutes, not hours, when the link holds.'
Write-Host ''
Write-Log ("EXPECTED DISK: {0:N2} GB extracted, ~{1:N2} GB peak.  Free now: {2:N0} MB." -f $estDiskGB, ($estDiskGB * 1.15), $freeMB)
Write-Log 'The archives are STREAMED and are never written to disk.'
Write-Host ''

Write-Log ("environment: Python {0}  |  zstandard {1} (libzstd {2})" -f $pfData.env.python, $pfData.env.zstandard, $pfData.env.libzstd) 'OK'
Write-Log ("extractor  : {0}   (invoked UNCHANGED)" -f $pfData.extractor.path)

# ============================================================================
# STEP 3 - ZENODO CONNECTIVITY / DNS PREFLIGHT  (before ANY extraction output)
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 3/6  ZENODO CONNECTIVITY / DNS PREFLIGHT' -ForegroundColor Cyan
Write-Rule '-'

function Test-ZenodoReachable {
    <#  Lightweight DNS + HTTP reachability probe. Transfers NO archive body:
        it issues a HEAD request only. Returns a hashtable. Never throws. #>
    param([string]$Url)

    $res = [ordered]@{
        host          = 'zenodo.org'
        url           = $Url
        curl_exe      = $CurlExe
        dns_ok        = $false
        dns_addresses = @()
        dns_error     = $null
        curl_rc       = $null
        http_status   = $null
        head_lines    = @()
        transport_err = $null
        ok            = $false
    }

    try {
        $addrs = [System.Net.Dns]::GetHostAddresses('zenodo.org')
        if ($addrs -and @($addrs).Count -gt 0) {
            $res.dns_ok = $true
            # "$_" invokes ToString() on the IPAddress, avoiding any reliance on
            # the IPAddressToString type-adapter property.
            $res.dns_addresses = @($addrs | ForEach-Object { "$_" })
        } else {
            $res.dns_error = 'resolver returned no addresses'
        }
    } catch {
        $res.dns_error = $_.Exception.Message
    }
    if (-not $res.dns_ok) { return $res }

    if (-not $CurlExe) {
        $res.transport_err = 'curl executable not found on PATH (looked for curl.exe)'
        return $res
    }

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $lines = @()
    $rc = -1
    try {
        $lines = @(& $CurlExe -sS -I -L --connect-timeout 30 --max-time 90 $Url 2>&1)
        if (Test-Path -Path variable:LASTEXITCODE) { $rc = $LASTEXITCODE }
    } catch {
        $res.transport_err = $_.Exception.Message
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $res.curl_rc = $rc
    # keep the raw response lines so the caller can read content-length; this is
    # header text only - no archive body is ever transferred by this probe.
    $res.head_lines = @($lines | ForEach-Object { [string]$_ })

    $status = $null
    foreach ($ln in $lines) {
        $s = [string]$ln
        $mm = [regex]::Match($s, '^HTTP/\d(?:\.\d)?\s+(\d{3})')
        if ($mm.Success) { $status = [int]$mm.Groups[1].Value }
    }
    if ($null -ne $status) { $res.http_status = $status }

    # Reachable == DNS resolved AND curl completed without a transport error.
    # A 4xx/5xx status still proves the host is reachable; it is reported but
    # does not hard-fail, because the extractor's own curl retries those.
    if ($res.dns_ok -and $rc -eq 0) { $res.ok = $true }
    return $res
}

$netProbeUrl = $StageDefs['ref'].Url
Write-Log "probing: $netProbeUrl  (HEAD only - no archive body transferred)"
$np = Test-ZenodoReachable -Url $netProbeUrl

if ($np.dns_ok) {
    Write-Log ("DNS          : OK   {0}" -f ($np.dns_addresses -join ', '))
} else {
    Write-Log ("DNS          : FAILED  {0}" -f $np.dns_error) 'ERROR'
}
Write-Log ("curl rc      : {0}" -f $np.curl_rc)
if ($null -ne $np.http_status) {
    if ($np.http_status -lt 400) {
        Write-Log ("HTTP status  : {0}" -f $np.http_status) 'OK'
    } elseif ($np.http_status -eq 405) {
        Write-Log ("HTTP status  : 405 (HEAD not allowed) - host is reachable") 'WARN'
    } else {
        Write-Log ("HTTP status  : {0} - host is reachable but the endpoint answered with an error; the extractor's curl retries transient statuses" -f $np.http_status) 'WARN'
    }
} else {
    Write-Log 'HTTP status  : not parsed from the response' 'WARN'
}
if ($np.transport_err) { Write-Log ("transport    : {0}" -f $np.transport_err) 'ERROR' }

if (-not $np.ok) {
    Write-Log 'ZENODO PREFLIGHT FAILED - zenodo.org cannot be resolved/reached.' 'ERROR'
    if (-not $np.dns_ok) { Write-Log ("  DNS error  : {0}" -f $np.dns_error) 'ERROR' }
    if ($np.curl_rc -ne 0) { Write-Log ("  curl rc    : {0}  (6=DNS, 7=connect, 28=timeout, 35=TLS)" -f $np.curl_rc) 'ERROR' }
    if ($np.transport_err) { Write-Log ("  transport  : {0}" -f $np.transport_err) 'ERROR' }
    Write-Log 'A DNS/connectivity failure is exactly what aborted the previous S1 attempt.' 'ERROR'
    Write-Log 'Aborting BEFORE creating any extraction output. Nothing was extracted.' 'ERROR'
    Write-Log 'Nothing was deleted. Re-run when DNS/connectivity is restored.' 'ERROR'
    exit 5
}
Write-Log 'ZENODO PREFLIGHT PASSED - DNS resolved and the endpoint responded.' 'OK'

# --- 3b. the archive identity must not have drifted. A truncated or replaced
#         artifact would make a byte-0 retry pointless or silently wrong.
#         Content-Length is read from the same HEAD response; the probe still
#         transfers no archive body.
$npLen = $null
if ($np.head_lines) {
    foreach ($ln in @($np.head_lines)) {
        $s = [string]$ln
        $mm = [regex]::Match($s, '^(?i)content-length:\s*(\d+)')
        if ($mm.Success) { $npLen = [long]$mm.Groups[1].Value }
    }
}
$refExpectedBytes = $StageDefs['ref'].ExpectedBytes
if ($null -eq $npLen) {
    Write-Log 'content-length not parsed from the HEAD response - cannot confirm the archive is unchanged.' 'WARN'
} elseif ($npLen -ne $refExpectedBytes) {
    Write-Log ("ARCHIVE IDENTITY FAILED: Reference_Maps content-length is {0:N0}, expected {1:N0}." -f $npLen, $refExpectedBytes) 'ERROR'
    Write-Log 'The remote artifact differs from the frozen constant. Refusing to extract.' 'ERROR'
    Write-Log 'Nothing was downloaded and nothing was extracted.' 'ERROR'
    exit 1
} else {
    Write-Log ("Reference_Maps content-length: {0:N0} (matches the frozen constant)" -f $npLen) 'OK'
}

if ($CheckOnly) {
    Write-Host ''
    Write-Rule
    Write-Log 'CHECK-ONLY MODE: S2 guard + preflight + Zenodo preflight all passed.' 'OK'
    Write-Log 'Nothing was downloaded, nothing was extracted.' 'OK'
    Write-Rule
    exit 0
}

# ============================================================================
# STEP 4 - CONFIRMATION
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 4/6  CONFIRMATION' -ForegroundColor Cyan
Write-Rule '-'
Write-Host 'BigEarthNet-S2 and BigEarthNet-S1 are complete and validated. They will NOT be touched.' -ForegroundColor Green
Write-Host ("You are about to stream {0:N0} bytes (~{1:N2} GB) and write {2:N0} files (~{3:N2} GB)." -f $totalBytes, ($totalBytes / 1GB), $totalFiles, $estDiskGB)
Write-Host 'Reference_Maps will start from BYTE 0. The previous partial tree is gone.'
Write-Host ("This includes {0:N0} snow/cloud/shadow patches (existing verified manifest)." -f $c.selected_from_snow_cloud) -ForegroundColor Yellow
Write-Host 'The stage is NOT resumable: if it aborts, it restarts from byte 0.'
Write-Host 'The previous attempt died at 91.48% on a remote truncation (curl 18).'
Write-Host ''

if (-not $SkipConfirm) {
    $answer = Read-Host "Type RETRY-REF (all caps) to begin, anything else to abort"
    if ($answer -cne 'RETRY-REF') {
        Write-Log 'Aborted by user at the confirmation prompt. Nothing was downloaded.' 'WARN'
        exit 0
    }
} else {
    Write-Log 'SkipConfirm set: proceeding without an interactive prompt.' 'WARN'
}

# ============================================================================
# STEP 5 - RUN THE STAGE  (ref only; s1 and s2 are never scheduled)
# ============================================================================
$stageResults = [ordered]@{}
$aggregate = 0

# Re-assert connectivity immediately before the first stage starts. DNS can
# change between the preflight and now, and this is the literal gate the run
# needs: no extraction output exists until the first stage writes a file.
Write-Log 're-checking Zenodo connectivity immediately before the first stage...'
$np2 = Test-ZenodoReachable -Url $netProbeUrl
if (-not $np2.ok) {
    Write-Log 'ZENODO RE-CHECK FAILED - aborting before any extraction output was created.' 'ERROR'
    Write-Log ("  DNS ok={0} curl rc={1} error={2}" -f $np2.dns_ok, $np2.curl_rc, $np2.dns_error) 'ERROR'
    exit 5
}
Write-Log ("Zenodo re-check passed (dns ok, curl rc={0}, http={1})" -f $np2.curl_rc, $np2.http_status) 'OK'

foreach ($k in $Ordered) {
    $d = $StageDefs[$k]
    $stageLog    = Join-Path $LogRoot ("{0}.log" -f $k)
    $stageReport = Join-Path $LogRoot ("{0}_report.json" -f $k)

    Write-Host ''
    Write-Rule
    Write-Host ("STAGE {0}  ->  {1}" -f $k.ToUpper(), $d.Label) -ForegroundColor Cyan
    Write-Rule
    Write-Log ("dirs file : {0}" -f $d.DirsFile)
    Write-Log ("output    : {0}" -f $OutRoot)
    Write-Log ("log       : {0}" -f $stageLog)
    Write-Log ("report    : {0}" -f $stageReport)
    Write-Log ("expected  : {0:N0} files, {1:N0} bytes" -f $d.ExpectedFiles, $d.ExpectedBytes)

    $stageArgs = @(
        $Extractor,
        '--url',            $d.Url,
        '--dirs-file',      $d.DirsFile,
        '--out',            $OutRoot,
        '--report',         $stageReport,
        '--archive',        $d.Label,
        '--expected-files', "$($d.ExpectedFiles)",
        '--expected-bytes', "$($d.ExpectedBytes)"
    )
    # No resume/offset/range argument exists or is passed: the extractor always
    # reads its archive from byte 0. The previous Reference_Maps stream was
    # truncated mid-flight and the frame is non-seekable, so byte 0 is the only
    # correct start. The output tree was confirmed absent/empty in STEP 1.

    $cmdLine = ('"{0}" {1}' -f $VenvPy, (($stageArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '))
    Write-Log ("command   : {0}" -f $cmdLine)
    Add-Content -Path $stageLog -Value ("# stage={0} archive={1}" -f $k, $d.Label) -Encoding UTF8
    Add-Content -Path $stageLog -Value ("# command={0}" -f $cmdLine) -Encoding UTF8
    Add-Content -Path $stageLog -Value ("# started={0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
    Add-Content -Path $stageLog -Value '' -Encoding UTF8

    $t0 = Get-Date
    Write-Log 'streaming... (live progress below; the extractor prints every 20 s)' 'OK'

    # Stream live to console AND to the stage log. Tee-Object is deliberately
    # NOT used: under Windows PowerShell 5.1 it writes through Out-File, whose
    # default encoding is Unicode (UTF-16LE), which would mix two encodings in
    # one stage log and blind the network-warning collector below.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $rc = -1
    $stageAbort = $null
    try {
        & $VenvPy @stageArgs 2>&1 | ForEach-Object {
            Add-Content -Path $stageLog -Value (($_ | Out-String).TrimEnd()) -Encoding UTF8
            $_
        }
        $rc = 0
        if (Test-Path -Path variable:LASTEXITCODE) { $rc = $LASTEXITCODE }
    } catch {
        $stageAbort = '{0}: {1}' -f $_.Exception.GetType().Name, $_.Exception.Message
        if (Test-Path -Path variable:LASTEXITCODE) { $rc = $LASTEXITCODE }
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $elapsed = (Get-Date) - $t0

    $finishLine = ("# finished={0} rc={1} wall={2:N0}s" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $rc, $elapsed.TotalSeconds)
    if ($stageAbort) { $finishLine = $finishLine + ("  launcher_abort={0}" -f $stageAbort) }
    Add-Content -Path $stageLog -Value $finishLine -Encoding UTF8
    Write-Log ("stage {0} exit code: {1}   (wall {2:N0} s = {3:N2} h)" -f $k, $rc, $elapsed.TotalSeconds, $elapsed.TotalHours)

    # --- collect network / retry noise from the stage log
    $netLines = @()
    if (Test-Path -LiteralPath $stageLog) {
        $netLines = @(Get-Content -LiteralPath $stageLog -Encoding UTF8 | Where-Object { $_ -match 'curl:|429|504|retry|ReadError|unexpected end|Connection|timed out' })
    }
    $netCount = $netLines.Count
    if ($netCount -gt 0) {
        Write-Log ("network/retry lines in stage log: {0}" -f $netCount) 'WARN'
        $netLines | Select-Object -First 10 | ForEach-Object { Write-Log ("    " + $_) 'WARN' }
    } else {
        Write-Log 'network/retry lines in stage log: 0'
    }

    # --- validate the stage report against the extractor's own invariants
    $rep = Read-JsonFile -Path $stageReport

    $checks = @()
    $stageOk = $true
    if (-not $rep) {
        $checks += 'FAIL  report json not readable'
        $stageOk = $false
    } else {
        $t = @(
            @{ n = 'files_extracted == expected'; ok = ($rep.files_extracted -eq $d.ExpectedFiles); d = "$($rep.files_extracted) vs $($d.ExpectedFiles)" },
            @{ n = 'size_mismatches == 0';        ok = ($rep.size_mismatches -eq 0);           d = "$($rep.size_mismatches)" },
            @{ n = 'dirs_never_seen == 0';        ok = ($rep.dirs_never_seen -eq 0);           d = "$($rep.dirs_never_seen)" },
            @{ n = 'stream_bytes_read == expected'; ok = ($rep.stream_bytes_read -eq $d.ExpectedBytes); d = "$($rep.stream_bytes_read) vs $($d.ExpectedBytes)" },
            @{ n = 'bytes_ok';                    ok = ($rep.bytes_ok -eq $true);              d = "$($rep.bytes_ok)" },
            @{ n = 'complete';                    ok = ($rep.complete -eq $true);              d = "$($rep.complete)" },
            @{ n = 'no abort_reason';             ok = ($null -eq $rep.abort_reason);          d = "$($rep.abort_reason)" }
        )
        foreach ($chk in $t) {
            $mark = 'PASS'
            if (-not $chk.ok) { $mark = 'FAIL'; $stageOk = $false }
            $checks += ("{0}  {1}  [{2}]" -f $mark, $chk.n, $chk.d)
        }
        Write-Log 'validation:'
        foreach ($chk in $checks) { Write-Log ("  " + $chk) }
    }

    $stageResults[$k] = [ordered]@{
        label            = $d.Label
        extractor_rc     = $rc
        wall_seconds     = [math]::Round($elapsed.TotalSeconds, 1)
        validated        = $stageOk
        network_warnings = $netCount
        report           = $stageReport
        log              = $stageLog
        files_extracted  = $(if ($rep) { $rep.files_extracted } else { $null })
        bytes_written    = $(if ($rep) { $rep.bytes_written } else { $null })
        stream_bytes_read= $(if ($rep) { $rep.stream_bytes_read } else { $null })
        effective_MBps   = $(if ($rep) { $rep.effective_MBps } else { $null })
        abort_reason     = $(if ($rep) { $rep.abort_reason } else { $null })
    }

    if ($rc -ne 0 -or -not $stageOk) {
        $aggregate = 2
        Write-Log ("stage {0} did NOT validate. NOT starting any later stage." -f $k) 'ERROR'
        Write-Log 'PARTIAL OUTPUT HAS BEEN LEFT IN PLACE AND MUST NOT BE DELETED.' 'ERROR'
        Write-Log 'The underlying stream is a single non-seekable zstd frame: this stage is NOT resumable.' 'ERROR'
        break
    }
    Write-Log ("stage {0} VALIDATED" -f $k) 'OK'
}

# ============================================================================
# STEP 6 - FROZEN S2 RE-FINGERPRINT + FINAL INTEGRITY VERIFICATION
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 6/6  FROZEN S2 RE-CHECK + FINAL INTEGRITY VERIFICATION' -ForegroundColor Cyan
Write-Rule '-'

# --- 6a. fingerprint S2 again and prove it was not modified
Write-Log 'fingerprinting BigEarthNet-S2 (after)...'
$s2FpAfterPath = Join-Path $LogRoot 's2_fingerprint_after.json'
$fpA = Invoke-PythonFile -ScriptPath $FingerprintTool `
        -Arguments @('--root', $S2Tree, '--out', $s2FpAfterPath, '--label', $S2Label) `
        -LogFile (Join-Path $LogRoot 's2_fingerprint_after_stdout.txt')
Write-Log $fpA.Output.Trim()
$s2FpAfter = Read-JsonFile -Path $s2FpAfterPath

$s2Unchanged = $false
$s2DiffNotes = @()
if (-not $s2FpAfter) {
    $s2DiffNotes += 'after-fingerprint not readable'
} else {
    if ($s2FpAfter.fingerprint_sha256 -ne $s2FpBefore.fingerprint_sha256) { $s2DiffNotes += 'metadata fingerprint changed' }
    if ($s2FpAfter.files -ne $s2FpBefore.files) { $s2DiffNotes += "files $($s2FpBefore.files) -> $($s2FpAfter.files)" }
    if ($s2FpAfter.bytes -ne $s2FpBefore.bytes) { $s2DiffNotes += "bytes $($s2FpBefore.bytes) -> $($s2FpAfter.bytes)" }
    if ($s2FpAfter.zero_byte -ne 0) { $s2DiffNotes += "zero_byte $($s2FpAfter.zero_byte)" }
    if ($s2DiffNotes.Count -eq 0) { $s2Unchanged = $true }
}
if ($s2Unchanged) {
    Write-Log ("BigEarthNet-S2 UNCHANGED - before/after fingerprint identical: {0}" -f $s2FpAfter.fingerprint_sha256) 'OK'
} else {
    Write-Log 'BigEarthNet-S2 DIFFERS FROM ITS PRE-RUN STATE:' 'ERROR'
    foreach ($n in $s2DiffNotes) { Write-Log "  - $n" 'ERROR' }
}

# --- 6a2. fingerprint S1 again and prove it was not modified
Write-Log 'fingerprinting BigEarthNet-S1 (after)...'
$s1FpAfterPath = Join-Path $LogRoot 's1_fingerprint_after.json'
$fp1A = Invoke-PythonFile -ScriptPath $FingerprintTool `
        -Arguments @('--root', $S1Tree, '--out', $s1FpAfterPath, '--label', $S1Label) `
        -LogFile (Join-Path $LogRoot 's1_fingerprint_after_stdout.txt')
Write-Log $fp1A.Output.Trim()
$s1FpAfter = Read-JsonFile -Path $s1FpAfterPath

$s1Unchanged = $false
$s1DiffNotes = @()
if (-not $s1FpAfter) {
    $s1DiffNotes += 'after-fingerprint not readable'
} else {
    if ($s1FpAfter.fingerprint_sha256 -ne $s1FpBefore.fingerprint_sha256) { $s1DiffNotes += 'metadata fingerprint changed' }
    if ($s1FpAfter.files -ne $s1FpBefore.files) { $s1DiffNotes += "files $($s1FpBefore.files) -> $($s1FpAfter.files)" }
    if ($s1FpAfter.bytes -ne $s1FpBefore.bytes) { $s1DiffNotes += "bytes $($s1FpBefore.bytes) -> $($s1FpAfter.bytes)" }
    if ($s1FpAfter.zero_byte -ne 0) { $s1DiffNotes += "zero_byte $($s1FpAfter.zero_byte)" }
    if ($s1DiffNotes.Count -eq 0) { $s1Unchanged = $true }
}
if ($s1Unchanged) {
    Write-Log ("BigEarthNet-S1 UNCHANGED - before/after fingerprint identical: {0}" -f $s1FpAfter.fingerprint_sha256) 'OK'
} else {
    Write-Log 'BigEarthNet-S1 DIFFERS FROM ITS PRE-RUN STATE:' 'ERROR'
    foreach ($n in $s1DiffNotes) { Write-Log "  - $n" 'ERROR' }
}

# --- 6b. integrity for ref, cross-checked against THIS run's fresh report
$integrityRefJson = Join-Path $LogRoot 'integrity_ref.json'
$ig = Invoke-PythonFile -ScriptPath $IntegrityVerifier `
        -Arguments @('--repo', $RepoRoot, '--out-root', $OutRoot, '--log-root', $LogRoot, `
                     '--stages', 'ref', '--out-json', $integrityRefJson) `
        -LogFile (Join-Path $LogRoot 'integrity_stdout.txt')
Write-Log $ig.Output.Trim()
$refData = Read-JsonFile -Path $integrityRefJson

# --- 6c. merge the three independently-verified trees into one integrity.json.
#         No single run directory holds all three reports (the S2 run's
#         s1_report.json is the FAILED S1 attempt), so each tree is verified
#         against the report of the run that actually produced it:
#           s2  <- S2GuardLogRoot   (the run that completed S2)
#           s1  <- S1ReportLogRoot  (the run that completed S1)
#           ref <- THIS run directory
$mergedTrees = [ordered]@{}
$treeSources = [ordered]@{ s2 = $guardData; s1 = $guard1Data; ref = $refData }
foreach ($k in @('s2','s1','ref')) {
    $src = $treeSources[$k]
    if ($src -and ($src.trees.PSObject.Properties.Name -contains $k)) {
        $mergedTrees[$k] = $src.trees.$k
    }
}
$mergedFailures = @()
if ($guardData  -and $guardData.failures)  { $mergedFailures += @($guardData.failures) }
if ($guard1Data -and $guard1Data.failures) { $mergedFailures += @($guard1Data.failures) }
if ($refData    -and $refData.failures)    { $mergedFailures += @($refData.failures) }

$guardOk = $false
if ($guardData -and $guardData.ok -eq $true) { $guardOk = $true }
$guard1Ok = $false
if ($guard1Data -and $guard1Data.ok -eq $true) { $guard1Ok = $true }
$refOk = $false
if ($refData -and $refData.ok -eq $true) { $refOk = $true }
$integrityOk = ($guardOk -and $guard1Ok -and $refOk -and $s2Unchanged -and $s1Unchanged)

$mergedIntegrity = [ordered]@{
    ok              = $integrityOk
    failures        = $mergedFailures
    out_root        = $OutRoot
    stages          = @('s2','s1','ref')
    s2_source_run   = $S2GuardLogRoot
    s2_source_sha256= $S2SourceSha
    s2_unchanged    = $s2Unchanged
    s1_source_run   = $S1ReportLogRoot
    s1_source_sha256= $S1SourceSha
    s1_unchanged    = $s1Unchanged
    raw_reports     = [ordered]@{
        s2_guard  = $s2GuardJson
        s1_guard  = $s1GuardJson
        ref       = $integrityRefJson
    }
    trees           = $mergedTrees
}
$mergedIntegrity | ConvertTo-Json -Depth 8 | Out-File -FilePath $IntegrityJson -Encoding UTF8

foreach ($k in @('s2','s1','ref')) {
    if ($mergedTrees.Contains($k)) {
        $t = $mergedTrees[$k]
        Write-Log ("{0,-18} files {1,9:N0} / {2,9:N0}   bytes {3,14:N0}   patch dirs {4,7:N0}   bad {5}" -f `
            $t.tree, $t.files, $t.expected_files, $t.bytes, $t.patch_dirs, $t.bad_patch_dirs)
    }
}
if ($integrityOk) {
    Write-Log 'FINAL INTEGRITY VERIFICATION PASSED (s2 frozen + s1 + ref)' 'OK'
} else {
    if ($aggregate -eq 0) { $aggregate = 3 }
    Write-Log 'FINAL INTEGRITY VERIFICATION FAILED' 'ERROR'
    foreach ($f in $mergedFailures) { Write-Log "  - $f" 'ERROR' }
}

# ============================================================================
# SUMMARY
# ============================================================================
$endedUtc = (Get-Date).ToUniversalTime()

Write-Host ''
Write-Rule
Write-Host 'REFERENCE_MAPS RETRY SUMMARY' -ForegroundColor Cyan
Write-Rule

Write-Log 'PER-STAGE RESULT'
Write-Log ("  {0,-18} {1,-10}" -f $S2Label, 'FROZEN') 'OK'
Write-Log ("  {0,-18} {1,-10}" -f $S1Label, 'FROZEN') 'OK'
foreach ($k in $Ordered) {
    if (-not $stageResults.Contains($k)) {
        Write-Log ("  {0,-18} {1,-10}" -f $StageDefs[$k].Label, 'NOT RUN') 'ERROR'
        continue
    }
    $r = $stageResults[$k]
    $verdict = 'FAIL'
    if ($r.validated -and $r.extractor_rc -eq 0) { $verdict = 'COMPLETED' }
    $line = ("  {0,-18} {1,-10} rc={2}  wall={3:N2} h  files={4}  net-warnings={5}" -f `
        $r.label, $verdict, $r.extractor_rc, ($r.wall_seconds / 3600.0), $r.files_extracted, $r.network_warnings)
    if ($verdict -eq 'COMPLETED') { Write-Log $line 'OK' } else { Write-Log $line 'ERROR' }
}

$allOk = $true
foreach ($k in $Ordered) {
    if (-not $stageResults.Contains($k)) { $allOk = $false; continue }
    if (-not ($stageResults[$k].validated -and $stageResults[$k].extractor_rc -eq 0)) { $allOk = $false }
}

$overall = 'FAILED'
if ($allOk -and $integrityOk) { $overall = 'PASSED' }
elseif (-not $allOk) { $overall = 'INCOMPLETE (partial output preserved)' }
else { $overall = 'COMPLETED BUT INTEGRITY CHECK FAILED' }

Write-Host ''
if ($overall -eq 'PASSED') { Write-Log ("WHOLE RETRY: {0}" -f $overall) 'OK' }
else { Write-Log ("WHOLE RETRY: {0}" -f $overall) 'ERROR' }

Write-Log ("BigEarthNet-S2 frozen & unmodified : {0}" -f $s2Unchanged)
Write-Log ("BigEarthNet-S1 frozen & unmodified : {0}" -f $s1Unchanged)
Write-Log ("started (UTC) : {0}" -f $script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))
Write-Log ("ended   (UTC) : {0}" -f $endedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))
Write-Log ("elapsed       : {0:N2} h" -f (($endedUtc - $script:StartedUtc).TotalHours))
Write-Log ("log directory : {0}" -f $LogRoot)

if ($aggregate -ne 0) {
    Write-Log 'PARTIAL OUTPUT HAS BEEN LEFT IN PLACE. DO NOT DELETE IT WITHOUT REVIEWING THE REPORT FIRST.' 'ERROR'
    Write-Log 'The affected stage is NOT resumable (single non-seekable zstd frame).' 'ERROR'
}

# --- machine-readable run record
$record = [ordered]@{
    script              = $MyInvocation.MyCommand.Path
    mode                = 'RESUME_REF'
    started_utc         = $script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ')
    ended_utc           = $endedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ')
    elapsed_hours       = [math]::Round(($endedUtc - $script:StartedUtc).TotalHours, 3)
    repo_root           = $RepoRoot
    out_root            = $OutRoot
    log_root            = $LogRoot
    check_only          = [bool]$CheckOnly
    stages_requested    = $Ordered
    stages_skipped      = @('s2','s1')
    s2_frozen           = [ordered]@{
        label           = $S2Label
        tree            = $S2Tree
        source_run      = $S2GuardLogRoot
        source_report   = $S2SourceReport
        source_report_sha256 = $S2SourceSha
        expected_files  = $S2ExpectedFiles
        expected_bytes_written = $S2WrittenBytes
        expected_stream_bytes  = $S2StreamBytes
        expected_patch_dirs    = $S2PatchDirs
        guard_ok        = $guardOk
        fingerprint_before = $s2FpBefore.fingerprint_sha256
        fingerprint_after  = $(if ($s2FpAfter) { $s2FpAfter.fingerprint_sha256 } else { $null })
        unchanged       = $s2Unchanged
    }
    s1_frozen           = [ordered]@{
        label           = $S1Label
        tree            = $S1Tree
        source_run      = $S1ReportLogRoot
        source_report   = $S1SourceReport
        source_report_sha256 = $S1SourceSha
        expected_files  = $S1ExpectedFiles
        expected_bytes_written = $S1WrittenBytes
        expected_stream_bytes  = $S1StreamBytes
        expected_patch_dirs    = $S1PatchDirs
        guard_ok        = $guard1Ok
        fingerprint_before = $s1FpBefore.fingerprint_sha256
        fingerprint_after  = $(if ($s1FpAfter) { $s1FpAfter.fingerprint_sha256 } else { $null })
        unchanged       = $s1Unchanged
    }
    ref_tree_before_run = [ordered]@{
        path            = $refTree
        present         = $refPresent
        files           = $refFiles
        bytes           = $refBytes
    }
    zenodo_preflight    = [ordered]@{
        url             = $np.url
        dns_ok          = $np.dns_ok
        dns_addresses   = $np.dns_addresses
        dns_error       = $np.dns_error
        curl_rc         = $np.curl_rc
        http_status     = $np.http_status
        ok              = $np.ok
    }
    python_version      = $pfData.env.python
    python_executable   = $pfData.env.executable
    zstandard_version   = $pfData.env.zstandard
    libzstd_version     = $pfData.env.libzstd
    extractor           = $pfData.extractor.path
    manifest            = $pfData.manifest
    composition         = $pfData.composition
    dirs_check          = $pfData.dirs
    plan                = $pfData.plan
    free_disk_mb        = $freeMB
    expected_bytes      = $totalBytes
    expected_files      = $totalFiles
    stage_results       = $stageResults
    integrity_ok        = $integrityOk
    s2_unchanged        = $s2Unchanged
    s1_unchanged        = $s1Unchanged
    overall             = $overall
    launcher_exit_code  = $aggregate
}
$record | ConvertTo-Json -Depth 8 | Out-File -FilePath $RunRecord -Encoding UTF8

# --- human summary
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("Phase 12 REFERENCE_MAPS RETRY summary (ref only; S2 and S1 not re-run)")
[void]$sb.AppendLine("overall           : $overall")
[void]$sb.AppendLine("started (UTC)     : $($script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))")
[void]$sb.AppendLine("ended   (UTC)     : $($endedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))")
[void]$sb.AppendLine("elapsed           : $([math]::Round(($endedUtc - $script:StartedUtc).TotalHours,2)) h")
[void]$sb.AppendLine("manifest hash     : $($m.declared_hash)")
[void]$sb.AppendLine("composition       : clean=$($c.selected_from_clean) snow_cloud=$($c.selected_from_snow_cloud) total=$($m.count)")
[void]$sb.AppendLine("log directory     : $LogRoot")
[void]$sb.AppendLine("")
[void]$sb.AppendLine(("{0,-18} {1,-10} files={2}" -f $S2Label, 'FROZEN', $S2ExpectedFiles))
[void]$sb.AppendLine(("{0,-18} {1,-10} files={2}" -f $S1Label, 'FROZEN', $S1ExpectedFiles))
foreach ($k in $Ordered) {
    if (-not $stageResults.Contains($k)) {
        [void]$sb.AppendLine(("{0,-18} {1,-10}" -f $StageDefs[$k].Label, 'NOT RUN'))
        continue
    }
    $r = $stageResults[$k]
    $v = 'FAIL'; if ($r.validated -and $r.extractor_rc -eq 0) { $v = 'COMPLETED' }
    [void]$sb.AppendLine(("{0,-18} {1,-10} rc={2} wall={3:N2}h files={4}" -f $r.label, $v, $r.extractor_rc, ($r.wall_seconds/3600.0), $r.files_extracted))
}
[void]$sb.AppendLine("")
[void]$sb.AppendLine("BigEarthNet-S2 unchanged : $s2Unchanged")
[void]$sb.AppendLine("BigEarthNet-S1 unchanged : $s1Unchanged")
[void]$sb.AppendLine("integrity check   : $(if($integrityOk){'PASSED'}else{'FAILED'})")
$sb.ToString() | Out-File -FilePath $SummaryTxt -Encoding UTF8

Write-Host ''
Write-Rule
Write-Log ("Files to send back: {0}" -f $RunRecord)
Write-Log ("                    {0}" -f $SummaryTxt)
Write-Log ("                    {0}" -f $IntegrityJson)
Write-Log ("                    {0}\ref_report.json" -f $LogRoot)
Write-Rule

exit $aggregate
