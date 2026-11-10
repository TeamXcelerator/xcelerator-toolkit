[CmdletBinding()]
param([string]$Root = '', [string]$PackageCache = '', [switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-native-lock.ps1')
$nativeLock = Get-XcWindowsDependencyLock
if ([string]::IsNullOrWhiteSpace($Root)) { $Root = Join-Path $PSScriptRoot '..\target\native-windows' }
$Root = [IO.Path]::GetFullPath($Root)
if ($Root -match '\s') { throw 'Select a native toolchain directory without whitespace.' }
$msys = Join-Path $Root 'msys64'
$prefix = Join-Path $msys 'ucrt64'
if ($VerifyOnly -or (Test-Path -LiteralPath (Join-Path $prefix 'bin\rustc.exe'))) {
    Assert-XcWindowsNativeFiles -Prefix $prefix -Lock $nativeLock
    Write-Output "Qualified native dependencies verified: $prefix. No installation or update performed."
    return
}
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$archive = Join-Path $Root 'msys2-base-x86_64-20260927.tar.zst'
$expected = '8a2095647afbf799d113d0cae35f7544589d059819042635504e75bd2179bd02'
$url = 'https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20260927.tar.zst'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'MSYS2 bootstrap archive SHA-256 mismatch.'
}
$bash = Join-Path $msys 'usr\bin\bash.exe'
if (-not (Test-Path -LiteralPath $bash)) {
    $entries = @(& tar -tf $archive)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the MSYS2 bootstrap archive.' }
    foreach ($entry in $entries) {
        if ($entry -notmatch '^msys64/' -or $entry -match '(^[/\\]|(^|[/\\])\.\.([/\\]|$)|:)') {
            throw 'Unsafe bootstrap archive member path.'
        }
    }
    & tar -xf $archive -C $Root
    if ($LASTEXITCODE -ne 0) { throw 'MSYS2 bootstrap extraction failed.' }
}
# Install the exact qualified native packages, even if the rolling repository
# offers newer versions. Package and signature bytes are bound by the lock.
if ([string]::IsNullOrWhiteSpace($PackageCache)) { $PackageCache = Join-Path $msys 'var\cache\pacman\pkg' }
$PackageCache = [IO.Path]::GetFullPath($PackageCache)
if ($PackageCache -match '\s') { throw 'Select a package cache path without whitespace.' }
New-Item -ItemType Directory -Force -Path $PackageCache | Out-Null
foreach ($package in $nativeLock.packages) {
    foreach ($suffix in @('', '.sig')) {
        $file = $package.file + $suffix
        $path = Join-Path $PackageCache $file
        $hash = if ($suffix -eq '') { $package.sha256 } else { $package.signature_sha256 }
        if (-not (Test-Path -LiteralPath $path)) {
            $temporary = $path + '.download-' + [Guid]::NewGuid().ToString('N')
            try {
                Invoke-WebRequest -UseBasicParsing -Uri ($nativeLock.package_base_url + $file) -OutFile $temporary
                Assert-XcWindowsPackageFile -Path $temporary -Sha256 $hash
                Move-Item -LiteralPath $temporary -Destination $path
            } finally {
                if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
            }
        }
        Assert-XcWindowsPackageFile -Path $path -Sha256 $hash
    }
}
# Use the official primary server; all packages retain pacman's signature checks.
foreach ($name in @('msys','mingw','ucrt64','clang64','clangarm64','mingw64')) {
    $mirror = Join-Path $msys ('etc\pacman.d\mirrorlist.' + $name)
    if (Test-Path -LiteralPath $mirror) {
        $server = if ($name -eq 'msys') { 'Server = https://repo.msys2.org/msys/$arch' }
                  elseif ($name -eq 'mingw') { 'Server = https://repo.msys2.org/mingw/$repo/' }
                  else { 'Server = https://repo.msys2.org/mingw/' + $name }
        Set-Content -LiteralPath $mirror -Value $server -Encoding ASCII
    }
}
$oldSystem = $env:MSYSTEM
$oldChere = $env:CHERE_INVOKING
$oldPathType = $env:MSYS2_PATH_TYPE
try {
    $env:MSYSTEM = 'UCRT64'
    $env:CHERE_INVOKING = '1'
    $env:MSYS2_PATH_TYPE = 'strict'
    # A core-runtime update exits its shell. Start a new shell for the remainder.
    & $bash --login -c 'pacman -Syu --noconfirm'
    if ($LASTEXITCODE -ne 0) { throw 'MSYS2 core update failed.' }
    & $bash --login -c 'pacman -Syu --noconfirm && pacman -S --noconfirm --needed make m4 diffutils'
    if ($LASTEXITCODE -ne 0) { throw 'Native build utility installation failed.' }
    $packageArguments = @($nativeLock.packages | ForEach-Object { (Join-Path $PackageCache $_.file).Replace('\','/') })
    & $bash --login (Join-Path $PSScriptRoot 'windows-install-packages.sh') @packageArguments
    if ($LASTEXITCODE -ne 0) { throw 'Pinned native dependency installation failed; no newer package is substituted.' }
    $inventory = @(& $bash --login -c 'pacman -Q')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record native package inventory.' }
    $inventory | Set-Content -LiteralPath (Join-Path $Root 'packages.txt') -Encoding UTF8
} finally {
    $env:MSYSTEM = $oldSystem
    $env:CHERE_INVOKING = $oldChere
    $env:MSYS2_PATH_TYPE = $oldPathType
}
Assert-XcWindowsNativeFiles -Prefix $prefix -Lock $nativeLock
Write-Output "Native dependencies installed under $msys. Dot-source tools/windows-env.ps1 to build."
