[CmdletBinding()]
param(
    [string]$Prefix = '',
    [string]$Output = '',
    [switch]$HighPrecision
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'windows-env.ps1') -Prefix $Prefix
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $repo ('target\native-windows\qualification-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
}
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$cargo = Join-Path $env:XC_NATIVE_PREFIX 'bin\cargo.exe'
$env:XC_CACHE_REMOTE = 'none'
$env:XC_PUBLISH_EXECUTE = 'false'
$env:XC_RUN_PROFILE = 'normal'
$env:XC_CACHE_MODE = 'reuse'
$env:XC_PUBLISH_TARGET = 'none'
$env:XC_PUBLISH_REPLACE = 'false'
# The general regression lane has a fixed CPU backend. Explicit device tests
# qualify CUDA separately, including its required execution and equality gates.
$env:XC_DENSE_BACKEND = 'cpu'
$env:XC_PRIME_RESPONSE_BACKEND = 'cpu'
foreach ($name in @('XC_PUBLISH_STAGING_ROOT','XC_VALIDATION_CACHE_ROOT','XC_VALIDATION_REPORT_ROOT','XC_VALIDATION_REFERENCE')) {
    [Environment]::SetEnvironmentVariable($name,$null,'Process')
}
$env:XC_CACHE_ROOT = Join-Path $Output 'cache'
$env:RUSTDOCFLAGS = '-D warnings'
$features = if ($HighPrecision) { @('--features','hp,arb') } else { @() }
$consumerFeatures = if ($HighPrecision) { @('--features','hp','--features','xc-numerics/arb') } else { @() }
$profile = if ($HighPrecision) { @('--release') } else { @() }
$checks = @(
    @{name='workspace-tests'; args=@('test','--workspace','--all-targets','--locked','--no-fail-fast') + $profile + $features},
    @{name='workspace-clippy'; args=@('clippy','--workspace','--all-targets','--locked') + $features + @('--','-D','warnings')},
    @{name='consumer-tests'; args=@('test','--manifest-path','tests/external-consumer/Cargo.toml','--locked') + $consumerFeatures},
    @{name='consumer-clippy'; args=@('clippy','--manifest-path','tests/external-consumer/Cargo.toml','--all-targets','--locked') + $consumerFeatures + @('--','-D','warnings')},
    @{name='rustdoc'; args=@('doc','--workspace','--no-deps','--locked') + $features},
    @{name='doctests'; args=@('test','--workspace','--doc','--locked') + $features}
)
$results = @()
foreach ($check in $checks) {
    Write-Output ('Starting native check: ' + $check.name)
    $stdout = Join-Path $Output ($check.name + '.stdout.log')
    $stderr = Join-Path $Output ($check.name + '.stderr.log')
    $timer = [Diagnostics.Stopwatch]::StartNew()
    # Arguments are fixed tokens without whitespace; no command shell is used.
    $process = Start-Process -FilePath $cargo -ArgumentList $check.args -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $timer.Stop()
    $results += [ordered]@{
        name=$check.name; arguments=$check.args; exit_code=$process.ExitCode
        seconds=[Math]::Round($timer.Elapsed.TotalSeconds,3)
        stdout_sha256=(Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash.ToLowerInvariant()
        stderr_sha256=(Get-FileHash -LiteralPath $stderr -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    [ordered]@{
        native_windows=$true; high_precision=[bool]$HighPrecision
        prefix=$env:XC_NATIVE_PREFIX; compiler=@(& $env:RUSTC -vV)
        checks=$results; completed=($results.Count -eq $checks.Count)
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $Output 'results.json') -Encoding UTF8
    Write-Output ('Finished ' + $check.name + ': exit ' + $process.ExitCode + ', ' + $results[-1].seconds + 's')
    if ($process.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 55
        Get-Content -LiteralPath $stdout -Tail 55
        throw ('Native qualification failed: ' + $check.name + '. Evidence: ' + $Output)
    }
}
Write-Output ('Native Windows qualification passed. Evidence: ' + $Output)
