# Construit dist\Retime\Retime.exe (+ ffmpeg embarqué) et crée un raccourci sur le Bureau.
#   .\build.ps1          reconstruction incrémentale
#   .\build.ps1 -Clean   reconstruction complète
param([switch]$Clean)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

& $py (Join-Path $root "tools\make_icon.py")
$opts = @("--noconfirm", "--distpath", (Join-Path $root "dist"), "--workpath", (Join-Path $root "build"))
if ($Clean) { $opts += "--clean" }
& (Join-Path $root ".venv\Scripts\pyinstaller.exe") (Join-Path $root "Retime.spec") @opts
if ($LASTEXITCODE -ne 0) { throw "PyInstaller a échoué" }

# ffmpeg / ffprobe à côté de l'exe (l'application les cherche d'abord dans .\bin)
$bin = Join-Path $root "dist\Retime\bin"
New-Item -ItemType Directory -Force $bin | Out-Null
foreach ($tool in "ffmpeg", "ffprobe") {
    $src = (Get-Command $tool -ErrorAction SilentlyContinue).Source
    if (-not $src) { $src = "C:\Program Files\FFmpeg\bin\$tool.exe" }
    Copy-Item $src $bin -Force
}

# Raccourci Bureau
$exe = Join-Path $root "dist\Retime\Retime.exe"
$lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Retime.lnk"
$sh = New-Object -ComObject WScript.Shell
$s = $sh.CreateShortcut($lnk)
$s.TargetPath = $exe
$s.WorkingDirectory = Split-Path $exe
$s.IconLocation = "$exe,0"
$s.Description = "Recalage de sous-titres sur l'audio japonais"
$s.Save()
Write-Host "OK : $exe"
