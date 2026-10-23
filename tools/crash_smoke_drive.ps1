# Drives resolve_crash_smoke (src/core/tests/crash_smoke.cpp) through its three
# modes and checks the Windows crash handler from outside, by exit code:
#   clean          -> exits 0, no fault.
#   fault-midrun   -> TerminateProcess with the fault's NTSTATUS (0xC0000005),
#                     fast, with no JIT-debugger dialog holding the process.
#   fault-teardown -> TerminateProcess with the shutdown code 0.
# A run that exceeds the timeout is a hang, which is the failure the handler
# exists to prevent.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\crash_smoke_drive.ps1 -Build src\core\build_cpu
# -Build is the CMake build directory; torch_cpu.dll and the other libtorch
# DLLs must sit in it or on PATH.
param(
    [Parameter(Mandatory = $true)][string]$Build,
    [int]$TimeoutSec = 20
)

$Build = (Resolve-Path $Build).Path
$env:PATH = "$Build;$env:PATH"
# Ninja places the test executables under the tests/ subdirectory.
$exe = Join-Path $Build "tests\resolve_crash_smoke.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path $Build "resolve_crash_smoke.exe" }
if (-not (Test-Path $exe)) { Write-Host "MISSING: resolve_crash_smoke.exe under $Build" -ForegroundColor Red; exit 2 }

$AV = -1073741819   # 0xC0000005 STATUS_ACCESS_VIOLATION as a signed int32

# Windows Error Reporting and the JIT debugger attach to a faulting process as
# separate processes that name its PID on their command line (-p <pid>). Only
# those are stopped, never other instances of the same programs.
function Stop-FaultReporters([int]$TargetPid) {
    Get-CimInstance Win32_Process -Filter "Name = 'WerFault.exe' OR Name = 'vsjitdebugger.exe'" |
        Where-Object { $_.CommandLine -match "-p\s+$TargetPid(\s|$)" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Invoke-Mode([string]$Mode) {
    $p = Start-Process -FilePath $exe -ArgumentList $Mode -NoNewWindow -PassThru
    $null = $p.Handle
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        Write-Host ("[{0}] HANG (> {1}s), stopping PID {2}" -f $Mode, $TimeoutSec, $p.Id) -ForegroundColor Red
        Stop-FaultReporters $p.Id
        try { $p.Kill() } catch {}
        return "HANG"
    }
    return $p.ExitCode
}

$results = [ordered]@{}
foreach ($m in "clean", "fault-midrun", "fault-teardown") { $results[$m] = Invoke-Mode $m }

$pass = $true
function Test-Exit([string]$Mode, $Actual, $Expected, [string]$Desc) {
    $ok = ($Actual -eq $Expected)
    if (-not $ok) { $script:pass = $false }
    $color = if ($ok) { "Green" } else { "Red" }
    $verdict = if ($ok) { "OK" } else { "FAIL" }
    Write-Host ("[{0}] exit={1} expected={2} ({3}) {4}" -f $Mode, $Actual, $Expected, $Desc, $verdict) -ForegroundColor $color
}

Test-Exit "clean"          $results["clean"]          0    "clean exit"
Test-Exit "fault-midrun"   $results["fault-midrun"]   $AV  "fault NTSTATUS, no hang"
Test-Exit "fault-teardown" $results["fault-teardown"] 0    "benign teardown exit"

Write-Host ""
if ($pass) { Write-Host "CRASH SMOKE: PASS" -ForegroundColor Green; exit 0 }
else       { Write-Host "CRASH SMOKE: FAIL" -ForegroundColor Red;  exit 1 }
