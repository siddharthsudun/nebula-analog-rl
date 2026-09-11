$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoPath
$env:PYTHONPATH = 'src'
$runName = 'snr_full_20260910_01'
$runPath = Join-Path $repoPath ('results/' + $runName)
$controlPath = Join-Path $repoPath ('work/' + $runName + '_control')
$launchPath = Join-Path $repoPath ('work/' + $runName + '_launch')
if ((Test-Path -LiteralPath $runPath) -or (Test-Path -LiteralPath $controlPath) -or (Test-Path -LiteralPath $launchPath)) { throw 'Refusing to overwrite or duplicate an existing run.' }
New-Item -ItemType Directory -Path $launchPath | Out-Null
$pythonPath = Join-Path $repoPath '.venv/Scripts/python.exe'
$supervisorProcess = Start-Process -FilePath $pythonPath -ArgumentList @('scripts/supervise_noise_pilot.py','--full','--wall-seconds','64800','--run-dir',$runPath,'--control-dir',$controlPath) -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $launchPath 'stdout.log') -RedirectStandardError (Join-Path $launchPath 'stderr.log') -PassThru
@{supervisor_pid=$supervisorProcess.Id; run_dir=$runPath; control_dir=$controlPath; started_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $launchPath 'launch.json')
Get-Content -LiteralPath (Join-Path $launchPath 'launch.json')
