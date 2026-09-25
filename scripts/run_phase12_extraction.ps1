<#
.SYNOPSIS
    SatQuery AI - Phase 12 streaming extraction runner (standalone, user-run).

.DESCRIPTION
    Streams the three BigEarthNet v2.0 (reBEN) archives over HTTP, decompresses in
    flight, and writes ONLY the 28,000 selected patches. NO ARCHIVE IS EVER WRITTEN
    TO DISK.

    This launcher CALLS THE EXISTING HARDENED EXTRACTOR
    (.scratch/p12_stream_extract.py). It does not reimplement, wrap, or weaken it.
    All of that extractor's guards (overrun detection, per-member size check,
    exit codes 2/3/4/5) remain fully in force.

    Stages:
      s2   BigEarthNet-S2       336,000 files
      s1   BigEarthNet-S1        56,000 files
      ref  Reference_Maps        28,000 files
      total                      420,000 files

.SAFETY
    Read-only apart from the extraction target. DELETES NOTHING. Never writes to
    Phase 9 artifacts, configs/base.yaml, levir_change_v001, or the selection
    manifest. Partial output is never cleaned up automatically. The underlying
    stream is NOT resumable, and this script does not pretend otherwise.

.PARAMETER RepoRoot
    Repository root. Default: parent of this script's directory.

.PARAMETER OutRoot
    Extraction target. Default: <repo>\data\bigearthnet_v2\reben

.PARAMETER LogRoot
    Log directory. Default: <repo>\logs\phase12_extraction\run_<UTC timestamp>

.PARAMETER Stages
    Comma-separated subset of: s2,s1,ref. Default: all three.

.PARAMETER CheckOnly
    Run the full preflight, print everything, then exit WITHOUT extracting.
    Use this first - it costs no bandwidth.

.PARAMETER SkipConfirm
    Skip the interactive confirmation prompt.
    USING THIS MEANS YOU ACCEPT THE MANIFEST COMPOSITION PRINTED ABOVE THE PROMPT,
    INCLUDING THE 3,268 SNOW/CLOUD/SHADOW PATCHES.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_phase12_extraction.ps1 -CheckOnly

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_phase12_extraction.ps1
#>
[CmdletBinding()]
param(
    [string]   $RepoRoot,
    [string]   $OutRoot,
    [string]   $LogRoot,
    [string[]] $Stages = @('s2','s1','ref'),
    [switch]   $CheckOnly,
    [switch]   $SkipConfirm
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
$Extractor = Join-Path $RepoRoot '.scratch\p12_stream_extract.py'
$Driver    = Join-Path $RepoRoot '.scratch\p12_run_extraction.py'
$Pyz       = Join-Path $RepoRoot '.scratch\pyz'
$DirsDir   = Join-Path $RepoRoot '.scratch\p12_extract'
$VenvPy    = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Manifest  = Join-Path $RepoRoot 'artifacts\phase12_selection\selection_manifest_seed10.jsonl'

if (-not $LogRoot) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $LogRoot = Join-Path $RepoRoot "logs\phase12_extraction\run_$stamp"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$RunLog       = Join-Path $LogRoot 'run.log'
$PreflightJson= Join-Path $LogRoot 'preflight.json'
$IntegrityJson= Join-Path $LogRoot 'integrity.json'
$RunRecord    = Join-Path $LogRoot 'run_record.json'
$SummaryTxt   = Join-Path $LogRoot 'summary.txt'

$script:RunLog = $RunLog
$script:StartedUtc = (Get-Date).ToUniversalTime()

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
    try {
        $all = & $VenvPy $ScriptPath @Arguments 2>&1
        $rc = 0
        if (Test-Path -Path variable:LASTEXITCODE) { $rc = $LASTEXITCODE }
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($LogFile) { $all | Out-File -FilePath $LogFile -Encoding UTF8 }
    return @{ ExitCode = $rc; Output = ($all | Out-String) }
}

# ----------------------------------------------------------------------------
# Stage table  (byte counts are the archives' exact Content-Length values)
# ----------------------------------------------------------------------------
$StageDefs = [ordered]@{
    s2 = @{
        Label          = 'BigEarthNet-S2'
        Url            = 'https://zenodo.org/api/records/10891137/files/BigEarthNet-S2.tar.zst/content'
        DirsFile       = (Join-Path $DirsDir 's2_dirs.txt')
        ExpectedFiles  = 336000
        ExpectedBytes  = 63251710377
        EstDiskGB      = 4.61
    }
    s1 = @{
        Label          = 'BigEarthNet-S1'
        Url            = 'https://zenodo.org/api/records/10891137/files/BigEarthNet-S1.tar.zst/content'
        DirsFile       = (Join-Path $DirsDir 's1_dirs.txt')
        ExpectedFiles  = 56000
        ExpectedBytes  = 54439153171
        EstDiskGB      = 3.25
    }
    ref = @{
        Label          = 'Reference_Maps'
        Url            = 'https://zenodo.org/api/records/10891137/files/Reference_Maps.tar.zst/content'
        DirsFile       = (Join-Path $DirsDir 'ref_dirs.txt')
        ExpectedFiles  = 28000
        ExpectedBytes  = 282391301
        EstDiskGB      = 0.08
    }
}

$Selected = @()
foreach ($s in $Stages) {
    $key = $s.Trim().ToLower()
    if (-not $StageDefs.Contains($key)) {
        Write-Host "Unknown stage '$s'. Valid: s2,s1,ref" -ForegroundColor Red
        exit 1
    }
    if ($Selected -notcontains $key) { $Selected += $key }
}
# always run in the fixed safe order
$Ordered = @()
foreach ($k in @('s2','s1','ref')) { if ($Selected -contains $k) { $Ordered += $k } }

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
Write-Rule
Write-Host 'SatQuery AI - Phase 12 streaming extraction runner' -ForegroundColor Cyan
Write-Rule
Write-Log "repo root    : $RepoRoot"
Write-Log "output root  : $OutRoot"
Write-Log "log root     : $LogRoot"
Write-Log "stages       : $($Ordered -join ', ')"
Write-Log "check only   : $([bool]$CheckOnly)"
Write-Log "started (UTC): $($script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
Write-Host ''

# ============================================================================
# STEP 1 - PREFLIGHT  (fail loudly BEFORE anything is downloaded)
# ============================================================================
Write-Rule '-'
Write-Host 'STEP 1/5  PREFLIGHT' -ForegroundColor Cyan
Write-Rule '-'

$preflightFailures = @()

foreach ($f in @($VenvPy, $Extractor, $Manifest)) {
    if (-not (Test-Path -LiteralPath $f)) { $preflightFailures += "missing required file: $f" }
}
foreach ($k in @('s2','s1','ref')) {
    $d = $StageDefs[$k].DirsFile
    if (-not (Test-Path -LiteralPath $d)) { $preflightFailures += "missing dirs file: $d" }
}
if (-not (Test-Path -LiteralPath $Pyz)) { $preflightFailures += "missing zstandard target dir: $Pyz" }

if ($preflightFailures.Count -gt 0) {
    Write-Log 'PREFLIGHT FAILED - required files are missing. Nothing was downloaded.' 'ERROR'
    foreach ($p in $preflightFailures) { Write-Log "  - $p" 'ERROR' }
    exit 1
}

# --- dirs-file line counts
Write-Log 'dirs-file line counts:'
foreach ($k in @('s2','s1','ref')) {
    $d    = $StageDefs[$k].DirsFile
    $n    = @(Get-Content -LiteralPath $d | Where-Object { $_.Trim() -ne '' }).Count
    $want = 28000
    $okStr = 'OK'
    if ($n -ne $want) { $okStr = "MISMATCH (want $want)"; $preflightFailures += "$($StageDefs[$k].Label) dirs file has $n lines, want $want" }
    Write-Log ("  {0,-18} {1,8} lines  {2}" -f $StageDefs[$k].Label, $n, $okStr)
}

# --- free disk
$drive = (Get-Item -LiteralPath $RepoRoot).PSDrive.Name
$freeMB = [math]::Round((Get-PSDrive -Name $drive).Free / 1MB, 0)
Write-Log ("free disk on {0}: {1:N0} MB" -f $drive, $freeMB)
if ($freeMB -lt 20000) { $preflightFailures += "only $freeMB MB free; want >= 20,000 MB" }

# --- deep preflight in Python: manifest hash, composition, path-set equality, env
Write-Log 'running deep preflight (manifest hash, composition, path-set equality, env)...'
$env:PYTHONPATH = $Pyz
$pf = Invoke-PythonFile -ScriptPath (Join-Path $RepoRoot 'scripts\p12_preflight_verify.py') `
        -Arguments @('--repo', $RepoRoot, '--out-json', $PreflightJson) `
        -LogFile (Join-Path $LogRoot 'preflight_stdout.txt')
Write-Log $pf.Output.Trim()

$pfData = $null
if (Test-Path -LiteralPath $PreflightJson) {
    $pfData = Get-Content -LiteralPath $PreflightJson -Raw | ConvertFrom-Json
}

if ($pf.ExitCode -ne 0 -or -not $pfData -or -not $pfData.ok) {
    Write-Log 'PREFLIGHT FAILED - see preflight.json. Nothing was downloaded.' 'ERROR'
    exit 1
}

# ============================================================================
# STEP 2 - IDENTITY, COMPOSITION, COST
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 2/5  MANIFEST IDENTITY AND COMPOSITION' -ForegroundColor Cyan
Write-Rule '-'

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
Write-Log ("scene key         : {0}" -f $m.scene_key)
Write-Log ("config hash       : {0}" -f $m.config_hash)
Write-Host ''

Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host '  PATCH COMPOSITION OF THE 28,000-PATCH SLICE' -ForegroundColor Yellow
Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host ("    clean (no snow/cloud/shadow) : {0,7:N0}" -f $c.selected_from_clean) -ForegroundColor Yellow
Write-Host ("    snow / cloud / shadow        : {0,7:N0}" -f $c.selected_from_snow_cloud) -ForegroundColor Yellow
Write-Host ("    total                        : {0,7:N0}" -f $m.count) -ForegroundColor Yellow
Write-Host ("    (parquet rows: clean {0:N0} + snow/cloud {1:N0} = {2:N0}; overlap {3})" -f `
    $c.parquet_clean_rows, $c.parquet_snow_cloud_rows, $c.parquet_union, $c.parquet_overlap) -ForegroundColor Yellow
Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host ("  {0:N0} of the 28,000 patches ({1:N1}%) carry seasonal snow and/or cloud/shadow." -f `
    $c.selected_from_snow_cloud, (100.0 * $c.selected_from_snow_cloud / $m.count)) -ForegroundColor Yellow
Write-Host '  This is the existing, verified manifest. It has NOT been altered.' -ForegroundColor Yellow
Write-Host '  If this composition is not what you want, press Ctrl+C now.' -ForegroundColor Yellow
Write-Host '  ------------------------------------------------------------------' -ForegroundColor Yellow
Write-Host ''

Add-Content -Path $script:RunLog -Value ("composition: clean={0} snow_cloud={1} total={2}" -f $c.selected_from_clean, $c.selected_from_snow_cloud, $m.count) -Encoding UTF8

Write-Host 'STAGE PLAN' -ForegroundColor Cyan
Write-Log ("  {0,-18} {1,10} {2,18} {3,10}" -f 'stage', 'files', 'stream bytes', 'est. disk')
foreach ($k in $Ordered) {
    $d = $StageDefs[$k]
    Write-Log ("  {0,-18} {1,10:N0} {2,18:N0} {3,9:N2} GB" -f $d.Label, $d.ExpectedFiles, $d.ExpectedBytes, $d.EstDiskGB)
}
Write-Log ("  {0,-18} {1,10:N0} {2,18:N0} {3,9:N2} GB" -f 'TOTAL', $totalFiles, $totalBytes, $estDiskGB)
Write-Host ''

# --- duration projection, from the measured diagnostics
$bytesMB = $totalBytes / 1MB
$writeHours = ($totalFiles * 21.2 / 1000.0) / 3600.0   # measured 21.2 ms/file create cost
Write-Host 'EXPECTED DURATION  (measured rates: 1.16-2.38 MB/s; 21.2 ms/file create cost)' -ForegroundColor Cyan
Write-Log ("  per-file create overhead (measured) : {0:N2} h for {1:N0} files" -f $writeHours, $totalFiles)
$proj = @(
    @{ n = 'slowest measured  (1.16 MB/s)'; r = 1.16  },
    @{ n = 'mean curl         (1.21 MB/s)'; r = 1.214 },
    @{ n = 'mean pipeline     (1.80 MB/s)'; r = 1.799 },
    @{ n = 'fastest measured  (2.38 MB/s)'; r = 2.379 },
    @{ n = '504-stall tail    (0.55 MB/s)'; r = 0.55  }
)
foreach ($p in $proj) {
    $h = ($bytesMB / $p.r) / 3600.0
    Write-Log ("  {0}  ->  stream {1,6:N1} h  +  writes {2:N1} h  =  {3,6:N1} h" -f $p.n, $h, $writeHours, ($h + $writeHours))
}
Write-Log '  CENTRAL ESTIMATE: ~16-31 h.  TAIL RISK: up to ~62 h if 504 stalls dominate.'
Write-Host ''
Write-Log ("EXPECTED DISK: {0:N2} GB extracted, ~{1:N2} GB peak.  Free now: {2:N0} MB." -f $estDiskGB, ($estDiskGB * 1.15), $freeMB)
Write-Log 'The archives are STREAMED and are never written to disk.'
Write-Host ''

Write-Log ("environment: Python {0}  |  zstandard {1} (libzstd {2})" -f $pfData.env.python, $pfData.env.zstandard, $pfData.env.libzstd) 'OK'
Write-Log ("extractor  : {0}" -f $pfData.extractor.path)
Write-Log ("driver     : {0}   (the original 2-stage driver; this launcher supersedes it by adding Reference_Maps)" -f $Driver)

if ($CheckOnly) {
    Write-Host ''
    Write-Rule
    Write-Log 'CHECK-ONLY MODE: preflight passed. Nothing was downloaded, nothing was extracted.' 'OK'
    Write-Rule
    exit 0
}

# ============================================================================
# STEP 3 - CONFIRMATION
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 3/5  CONFIRMATION' -ForegroundColor Cyan
Write-Rule '-'
Write-Host ("You are about to stream {0:N0} bytes (~{1:N1} GB) and write {2:N0} files (~{3:N1} GB)." -f $totalBytes, ($totalBytes / 1GB), $totalFiles, $estDiskGB)
Write-Host ("This includes {0:N0} snow/cloud/shadow patches." -f $c.selected_from_snow_cloud) -ForegroundColor Yellow
Write-Host 'The run is NOT resumable: if it aborts, it restarts from byte 0.'
Write-Host 'It may take 16-31 hours (tail risk ~62 h). Run it somewhere it will not be interrupted.'
Write-Host ''

if (-not $SkipConfirm) {
    $answer = Read-Host "Type EXTRACT (all caps) to begin, anything else to abort"
    if ($answer -cne 'EXTRACT') {
        Write-Log 'Aborted by user at the confirmation prompt. Nothing was downloaded.' 'WARN'
        exit 0
    }
} else {
    Write-Log 'SkipConfirm set: proceeding without an interactive prompt.' 'WARN'
}

# ============================================================================
# STEP 4 - RUN THE STAGES
# ============================================================================
$stageResults = [ordered]@{}
$aggregate = 0     # 0 = success so far

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

    $cmdLine = ('"{0}" {1}' -f $VenvPy, (($stageArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '))
    Write-Log ("command   : {0}" -f $cmdLine)
    Add-Content -Path $stageLog -Value ("# stage={0} archive={1}" -f $k, $d.Label) -Encoding UTF8
    Add-Content -Path $stageLog -Value ("# command={0}" -f $cmdLine) -Encoding UTF8
    Add-Content -Path $stageLog -Value ("# started={0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
    Add-Content -Path $stageLog -Value '' -Encoding UTF8

    $t0 = Get-Date
    Write-Log 'streaming... (live progress below; the extractor prints every 20 s)' 'OK'

    # Stream live to console AND to the stage log. The extractor's own guards,
    # exit codes and validation semantics are untouched.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $rc = -1
    $stageAbort = $null
    try {
        # Tee the child's combined output to the stage log AND to the console.
        # Tee-Object is deliberately NOT used here: under Windows PowerShell 5.1
        # it writes through Out-File, whose default encoding is Unicode
        # (UTF-16LE). Combined with the UTF-8-with-BOM header written above via
        # Add-Content -Encoding UTF8, that produced a stage log holding TWO
        # encodings; Get-Content (no -Encoding) then honoured the UTF-8 BOM and
        # decoded the whole file as UTF-8, so the network-warning collector below
        # could never match a curl/network line (measured: 0 matches on a log
        # that actually contained 7). Add-Content -Encoding UTF8 keeps the stage
        # log in ONE encoding while still emitting every line to the console as
        # it arrives, so live progress is unchanged. The extractor's own output,
        # curl arguments, retry semantics and exit codes are untouched.
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
        # -Encoding UTF8 is explicit: the stage log is now written as UTF-8
        # throughout, so the collector must not rely on BOM sniffing.
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
    $rep = $null
    if (Test-Path -LiteralPath $stageReport) {
        try { $rep = Get-Content -LiteralPath $stageReport -Raw | ConvertFrom-Json } catch { $rep = $null }
    }

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
        Write-Log 'The underlying stream is a single non-seekable zstd frame: this run is NOT resumable.' 'ERROR'
        break
    }
    Write-Log ("stage {0} VALIDATED" -f $k) 'OK'
}

# ============================================================================
# STEP 5 - FINAL INTEGRITY VERIFICATION (independent on-disk scan)
# ============================================================================
Write-Host ''
Write-Rule '-'
Write-Host 'STEP 5/5  FINAL INTEGRITY VERIFICATION' -ForegroundColor Cyan
Write-Rule '-'

$ig = Invoke-PythonFile -ScriptPath (Join-Path $RepoRoot 'scripts\p12_integrity_verify.py') `
        -Arguments @('--repo', $RepoRoot, '--out-root', $OutRoot, '--log-root', $LogRoot, `
                     '--stages', ($Ordered -join ','), '--out-json', $IntegrityJson) `
        -LogFile (Join-Path $LogRoot 'integrity_stdout.txt')
Write-Log $ig.Output.Trim()

$igData = $null
if (Test-Path -LiteralPath $IntegrityJson) {
    $igData = Get-Content -LiteralPath $IntegrityJson -Raw | ConvertFrom-Json
}
foreach ($k in $Ordered) {
    if ($igData -and $igData.trees.PSObject.Properties.Name -contains $k) {
        $t = $igData.trees.$k
        Write-Log ("{0,-18} files {1,9:N0} / {2,9:N0}   bytes {3,14:N0}   patch dirs {4,7:N0}   bad {5}" -f `
            $t.tree, $t.files, $t.expected_files, $t.bytes, $t.patch_dirs, $t.bad_patch_dirs)
    }
}
if ($ig.ExitCode -ne 0) {
    if ($aggregate -eq 0) { $aggregate = 3 }
    Write-Log 'FINAL INTEGRITY VERIFICATION FAILED' 'ERROR'
} else {
    Write-Log 'FINAL INTEGRITY VERIFICATION PASSED' 'OK'
}

# ============================================================================
# SUMMARY
# ============================================================================
$endedUtc = (Get-Date).ToUniversalTime()

Write-Host ''
Write-Rule
Write-Host 'EXTRACTION SUMMARY' -ForegroundColor Cyan
Write-Rule

Write-Log 'PER-STAGE RESULT'
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
foreach ($k in @('s2','s1','ref')) {
    if ($Ordered -notcontains $k) {
        Write-Log ("  {0,-18} {1,-10}" -f $StageDefs[$k].Label, 'NOT REQUESTED')
    }
}

$allOk = $true
foreach ($k in $Ordered) {
    if (-not $stageResults.Contains($k)) { $allOk = $false; continue }
    if (-not ($stageResults[$k].validated -and $stageResults[$k].extractor_rc -eq 0)) { $allOk = $false }
}
$integrityOk = ($ig.ExitCode -eq 0)

$overall = 'FAILED'
if ($allOk -and $integrityOk) { $overall = 'PASSED' }
elseif (-not $allOk) { $overall = 'INCOMPLETE (partial output preserved)' }
else { $overall = 'COMPLETED BUT INTEGRITY CHECK FAILED' }

Write-Host ''
if ($overall -eq 'PASSED') { Write-Log ("WHOLE EXTRACTION: {0}" -f $overall) 'OK' }
else { Write-Log ("WHOLE EXTRACTION: {0}" -f $overall) 'ERROR' }

Write-Log ("started (UTC) : {0}" -f $script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))
Write-Log ("ended   (UTC) : {0}" -f $endedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))
Write-Log ("elapsed       : {0:N2} h" -f (($endedUtc - $script:StartedUtc).TotalHours))
Write-Log ("log directory : {0}" -f $LogRoot)

if ($aggregate -ne 0) {
    Write-Log 'PARTIAL OUTPUT HAS BEEN LEFT IN PLACE. DO NOT DELETE IT WITHOUT REVIEWING THE REPORT FIRST.' 'ERROR'
    Write-Log 'This extraction is NOT resumable (single non-seekable zstd frame).' 'ERROR'
}

# --- machine-readable run record
$record = [ordered]@{
    script            = $MyInvocation.MyCommand.Path
    started_utc       = $script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ')
    ended_utc         = $endedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ')
    elapsed_hours     = [math]::Round(($endedUtc - $script:StartedUtc).TotalHours, 3)
    repo_root         = $RepoRoot
    out_root          = $OutRoot
    log_root          = $LogRoot
    check_only        = [bool]$CheckOnly
    stages_requested  = $Ordered
    python_version    = $pfData.env.python
    python_executable = $pfData.env.executable
    zstandard_version = $pfData.env.zstandard
    libzstd_version   = $pfData.env.libzstd
    extractor         = $pfData.extractor.path
    manifest          = $pfData.manifest
    composition       = $pfData.composition
    dirs_check        = $pfData.dirs
    plan              = $pfData.plan
    free_disk_mb      = $freeMB
    expected_bytes    = $totalBytes
    expected_files    = $totalFiles
    stage_results     = $stageResults
    integrity_ok      = $integrityOk
    overall           = $overall
    launcher_exit_code= $aggregate
}
$record | ConvertTo-Json -Depth 8 | Out-File -FilePath $RunRecord -Encoding UTF8

# --- human summary
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("Phase 12 extraction summary")
[void]$sb.AppendLine("overall           : $overall")
[void]$sb.AppendLine("started (UTC)     : $($script:StartedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))")
[void]$sb.AppendLine("ended   (UTC)     : $($endedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'))")
[void]$sb.AppendLine("elapsed           : $([math]::Round(($endedUtc - $script:StartedUtc).TotalHours,2)) h")
[void]$sb.AppendLine("manifest hash     : $($m.declared_hash)")
[void]$sb.AppendLine("composition       : clean=$($c.selected_from_clean) snow_cloud=$($c.selected_from_snow_cloud) total=$($m.count)")
[void]$sb.AppendLine("log directory     : $LogRoot")
[void]$sb.AppendLine("")
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
[void]$sb.AppendLine("integrity check   : $(if($integrityOk){'PASSED'}else{'FAILED'})")
$sb.ToString() | Out-File -FilePath $SummaryTxt -Encoding UTF8

Write-Host ''
Write-Rule
Write-Log ("Files to send back: {0}" -f $RunRecord)
Write-Log ("                    {0}" -f $SummaryTxt)
Write-Log ("                    {0}" -f $PreflightJson)
Write-Log ("                    {0}" -f $IntegrityJson)
Write-Log ("                    {0}\s2_report.json | s1_report.json | ref_report.json" -f $LogRoot)
Write-Rule

exit $aggregate
