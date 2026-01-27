# Launch build via cmd.exe process, redirect output to log file
$batFile = "$PSScriptRoot\do_build.bat"
$logFile = "C:\tmp\resolve_build_log.txt"

$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = "$env:SystemRoot\System32\cmd.exe"
$pinfo.Arguments = "/c `"`"$batFile`" > `"$logFile`" 2>&1`""
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true
$pinfo.WorkingDirectory = $PSScriptRoot

$p = [System.Diagnostics.Process]::Start($pinfo)
$p.WaitForExit(600000)

if (Test-Path $logFile) {
    Get-Content $logFile
} else {
    Write-Host "ERROR: Log file not created"
}
exit $p.ExitCode
