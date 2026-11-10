# Shared, process-local verification for the qualified native Windows toolchain.
function Get-XcWindowsDependencyLock {
    param([string]$Path = (Join-Path $PSScriptRoot 'windows-dependencies.json'))
    $lock = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($lock.schema_version -ne 1 -or $lock.target -ne 'x86_64-pc-windows-gnu' -or
        $lock.rust_version -ne '1.99.1' -or $lock.packages.Count -eq 0 -or $lock.native_files.Count -eq 0 -or
        $lock.package_base_url -ne 'https://repo.msys2.org/mingw/ucrt64/') {
        throw 'Unsupported native Windows dependency lock.'
    }
    foreach ($package in $lock.packages) {
        if ($package.name -notmatch '^mingw-w64-ucrt-x86_64-[a-z0-9+_-]+$' -or
            $package.version -notmatch '^[A-Za-z0-9.~+_-]+$' -or
            $package.file -cne ($package.name + '-' + $package.version + '-any.pkg.tar.zst') -or
            $package.sha256 -notmatch '^[a-f0-9]{64}$' -or $package.signature_sha256 -notmatch '^[a-f0-9]{64}$') {
            throw 'Malformed native Windows package binding.'
        }
    }
    foreach ($file in $lock.native_files) {
        if ($file.path -notmatch '^(bin|include|lib/gcc|x86_64-w64-mingw32/bin)/[A-Za-z0-9_./+-]+$' -or
            $file.path -match '(^|/)\.\.(/|$)' -or $file.sha256 -notmatch '^[a-f0-9]{64}$') {
            throw 'Malformed native Windows file binding.'
        }
    }
    return $lock
}

function Assert-XcWindowsNativeFiles {
    param([Parameter(Mandatory=$true)][string]$Prefix, [Parameter(Mandatory=$true)]$Lock)
    foreach ($file in $Lock.native_files) {
        $path = Join-Path $Prefix $file.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.sha256) {
            throw "Native dependency differs from the qualified v0.17.1 build: $($file.path). Use a fresh pinned bootstrap or requalify the changed toolchain."
        }
    }
}

function Assert-XcWindowsPackageFile {
    param([Parameter(Mandatory=$true)][string]$Path, [Parameter(Mandatory=$true)][string]$Sha256)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Sha256) {
        throw "Native package SHA-256 mismatch: $([IO.Path]::GetFileName($Path))"
    }
}
