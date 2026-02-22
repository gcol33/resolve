# Launch RESOLVE build via Git Bash -> cmd.exe.
#
# Direct PowerShell -> cmd.exe fails: nvcc's cudafe++ subprocess resolution
# breaks in the PowerShell environment. Git Bash -> cmd.exe works reliably.
# A temporary .sh script avoids Start-Process argument quoting issues.

$logFile = "C:\tmp\resolve_build_log.txt"
$bash = "C:\Program Files\Git\bin\bash.exe"
$batFile = "$PSScriptRoot\do_build.bat"
$shScript = "C:\tmp\_resolve_build.sh"

if (-not (Test-Path "C:\tmp")) { New-Item -ItemType Directory -Path "C:\tmp" | Out-Null }
if (-not (Test-Path $bash)) {
    Write-Host "ERROR: Git Bash not found at $bash"
    exit 1
}

# Write a bash script that runs the build (avoids quoting issues)
$batMingw = $batFile -replace '\\','/' -replace '^C:','/c'
$logMingw = $logFile -replace '\\','/' -replace '^C:','/c'
"cmd.exe //c `"$batMingw`" > `"$logMingw`" 2>&1" | Set-Content -Path $shScript -Encoding ASCII

$p = Start-Process -FilePath $bash -ArgumentList $shScript `
    -Wait -PassThru -NoNewWindow

if (Test-Path $logFile) {
    Get-Content $logFile
} else {
    Write-Host "ERROR: Log file not created at $logFile"
}

exit $p.ExitCode
