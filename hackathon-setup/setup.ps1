param(
    [string]$Python = "python",
    [switch]$Online
)
$ErrorActionPreference = "Stop"
$SetupRoot = $PSScriptRoot
$VenvRoot = Join-Path $SetupRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$LockFile = Join-Path $SetupRoot "requirements.windows-py313.lock.txt"
$WheelDir = Join-Path $SetupRoot "wheels"

& $Python -c "import sys,platform; assert sys.platform == 'win32' and sys.version_info[:2] == (3,13) and platform.machine().upper() == 'AMD64', 'This bundle needs standard Windows x64 CPython 3.13.'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Install standard Windows x64 Python 3.13, then rerun with -Python pointing to it." }
if (!(Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

if ($Online) {
    Write-Host "Installing the pinned setup dependencies from your configured package index."
    & $VenvPython -m pip install --disable-pip-version-check -r $LockFile
} else {
    if (!(Test-Path -LiteralPath $WheelDir)) { throw "Offline wheels are missing. Prepare them first, or use -Online while connected." }
    & $VenvPython -m pip install --disable-pip-version-check --no-index --find-links $WheelDir -r $LockFile
}
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the pip error above." }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency conflicts detected." }
& $VenvPython (Join-Path $SetupRoot "preflight.py")
if ($LASTEXITCODE -ne 0) { throw "Readiness checks failed. See the JSON results above." }
Write-Host "Setup passed on this computer. ALMA connectivity is a separate check at the venue."
