param(
    [Parameter(Mandatory=$true)][string]$BaseUrl,
    [Parameter(Mandatory=$true)][string]$Model
)
$ErrorActionPreference = "Stop"
$Endpoint = [Uri]$BaseUrl
if (!$Endpoint.IsAbsoluteUri -or $Endpoint.Scheme -notin @("http", "https")) {
    throw "Enter the ALMA HTTP(S) base URL supplied by the organizers."
}
if ($Endpoint.UserInfo -or $Endpoint.Query -or $Endpoint.Fragment) {
    throw "Use an endpoint URL without credentials, query parameters, or fragments."
}
$env:USE_GOOGLE_GENAI = "false"
$env:GEMINI_API_KEY = ""
$env:LLM_BASE_URL = $BaseUrl.TrimEnd('/')
$env:LLM_MODEL = $Model
$env:LLM_API_KEY = "alma-local"
$env:FASTEMBED_CACHE_PATH = Join-Path $PSScriptRoot "model-cache"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
Write-Host "ALMA settings applied to this PowerShell session. No connection has been made."
Write-Host "Use .venv\Scripts\python.exe for the application you build at the event."
