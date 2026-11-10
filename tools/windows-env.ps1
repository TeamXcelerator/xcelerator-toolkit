# Dot-source this file to configure this PowerShell process for native UCRT.
# No registry, global PATH, WSL, or toolchain default is modified.
[CmdletBinding()]
param([string]$Prefix = '')
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Prefix)) { $Prefix = Join-Path $PSScriptRoot '..\target\native-windows\msys64\ucrt64' }
$Prefix = [IO.Path]::GetFullPath($Prefix)
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($Prefix -match '\s' -or $repoRoot -match '\s') {
    throw 'The qualified GMP/MPFR native build lane requires checkout and dependency paths without whitespace (for example C:\src\xcelerator-toolkit).'
}
. (Join-Path $PSScriptRoot 'windows-native-lock.ps1')
$nativeLock = Get-XcWindowsDependencyLock
Assert-XcWindowsNativeFiles -Prefix $Prefix -Lock $nativeLock
$bin = Join-Path $Prefix 'bin'
foreach ($relative in @('bin\rustc.exe','bin\cargo.exe','bin\gcc.exe','bin\pkg-config.exe','include\gmp.h','include\mpfr.h','include\flint\flint.h')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Prefix $relative))) {
        throw "Missing native dependency $relative. Run tools/windows-bootstrap.ps1 or supply a coherent UCRT64 prefix."
    }
}
$msysBin = Join-Path (Split-Path -Parent $Prefix) 'usr\bin'
$env:PATH = $bin + ';' + $msysBin + ';' + $env:PATH
$env:XC_NATIVE_PREFIX = $Prefix
$env:RUSTC = Join-Path $bin 'rustc.exe'
$env:RUSTDOC = Join-Path $bin 'rustdoc.exe'
$env:CC = Join-Path $bin 'gcc.exe'
$env:AR = Join-Path $bin 'ar.exe'
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = $env:CC
$env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-gnu'
foreach ($name in @('CC_x86_64-pc-windows-gnu','CC_x86_64_pc_windows_gnu','HOST_CC','TARGET_CC')) {
    [Environment]::SetEnvironmentVariable($name,$env:CC,'Process')
}
foreach ($name in @('AR_x86_64-pc-windows-gnu','AR_x86_64_pc_windows_gnu','HOST_AR','TARGET_AR')) {
    [Environment]::SetEnvironmentVariable($name,$env:AR,'Process')
}
[Environment]::SetEnvironmentVariable('PKG_CONFIG_SYSROOT_DIR',$null,'Process')
$env:PKG_CONFIG = Join-Path $bin 'pkg-config.exe'
$env:PKG_CONFIG_PATH = Join-Path $Prefix 'lib\pkgconfig'
$env:PKG_CONFIG_LIBDIR = $env:PKG_CONFIG_PATH
$env:LIBRARY_PATH = Join-Path $Prefix 'lib'
$env:CARGO_TARGET_DIR = Join-Path (Split-Path -Parent $Prefix) 'build'
$version = & $env:RUSTC --version
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^rustc 1\.99\.1 ') {
    throw "Expected Rust 1.99.1 in the selected UCRT prefix; found $version"
}
$hostInfo = @(& $env:RUSTC -vV)
if ($LASTEXITCODE -ne 0 -or $hostInfo -notcontains 'host: x86_64-pc-windows-gnu') {
    throw 'Native Windows qualification requires a matching GNU host compiler, not an MSVC-to-GNU cross build.'
}
Write-Output "Native Windows UCRT environment: $Prefix ($version)"
