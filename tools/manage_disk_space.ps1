# Report checkout storage, or remove explicitly selected abandoned Git transports.
# Windows PowerShell 5.1; no network access, builds, or remote Git operations.
[CmdletBinding()]
param(
    [string]$RepositoryRoot = '',
    [string[]]$JournalRoot = @(),
    [switch]$CleanTransport,
    [ValidateRange(1, 3650)][int]$MinimumAgeDays = 7
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = Split-Path -Parent $PSScriptRoot }

function Assert-PlainPath([string]$Path) {
    $current = $Path
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse point: $current"
            }
        }
        $parent = Split-Path -Parent $current
        if ($parent -eq $current) { break }
        $current = $parent
    }
}

function Measure-Tree([string]$Path, [switch]$Strict) {
    $files = [Collections.Generic.List[object]]::new()
    $errors = [Collections.Generic.List[string]]::new()
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Path)
    [long]$bytes = 0
    $newest = [datetime]::MinValue
    while ($pending.Count) {
        $directory = $pending.Pop()
        try {
            $directoryItem = [IO.DirectoryInfo]::new($directory)
            if ($directoryItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse point: $directory"
            }
            if ($directoryItem.LastWriteTimeUtc -gt $newest) {
                $newest = $directoryItem.LastWriteTimeUtc
            }
            foreach ($item in $directoryItem.GetFileSystemInfos()) {
                if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    if ($Strict) { throw "Refusing reparse point: $($item.FullName)" }
                    $errors.Add("Skipped reparse point: $($item.FullName)")
                    continue
                }
                if ($item -is [IO.DirectoryInfo]) { $pending.Push($item.FullName); continue }
                $bytes += $item.Length
                if ($item.LastWriteTimeUtc -gt $newest) { $newest = $item.LastWriteTimeUtc }
                if ($Strict) {
                    $files.Add([ordered]@{
                        path = $item.FullName.Substring($Path.Length + 1)
                        bytes = $item.Length
                        modified_ticks = $item.LastWriteTimeUtc.Ticks
                    })
                }
            }
        } catch {
            if ($Strict) { throw }
            $errors.Add($_.Exception.Message)
        }
    }
    [pscustomobject]@{
        bytes = $bytes
        newest_utc = $newest
        files = @($files.ToArray() | Sort-Object { $_.path })
        errors = $errors.ToArray()
    }
}

$repo = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
Assert-PlainPath $repo
$gitRoot = & git -C $repo rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($gitRoot).TrimEnd('\', '/') -ne $repo) {
    throw 'RepositoryRoot must be the root of an existing Git checkout.'
}
$target = Join-Path $repo 'target'
$targetPrefix = $target + [IO.Path]::DirectorySeparatorChar
if ($CleanTransport -and $JournalRoot.Count -eq 0) {
    throw 'Cleanup requires explicit -JournalRoot paths. Run without -CleanTransport to inspect first.'
}

$report = [ordered]@{
    measured_utc = [datetime]::UtcNow.ToString('o')
    repository = $repo
    mode = $(if ($CleanTransport) { 'cleanup' } else { 'report' })
    minimum_age_days = $MinimumAgeDays
    directories = @()
    transports = @()
    removed_bytes = [long]0
}

if (-not $CleanTransport) {
    foreach ($entry in Get-ChildItem -LiteralPath $repo -Force) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($entry.PSIsContainer) {
            $size = Measure-Tree $entry.FullName
            $report.directories += [ordered]@{
                path = $entry.Name; bytes = $size.bytes; errors = $size.errors
            }
        } else {
            $report.directories += [ordered]@{ path = $entry.Name; bytes = $entry.Length; errors = @() }
        }
    }
    $report.total_bytes = [long]0
    foreach ($directory in $report.directories) { $report.total_bytes += $directory.bytes }
    $report.total_gib = [math]::Round($report.total_bytes / 1GB, 3)
    # Historical managed publication journals are direct children of target/.
    # Deeper or custom journals can be inspected by passing -JournalRoot.
    if ($JournalRoot.Count -eq 0 -and (Test-Path -LiteralPath $target)) {
        $JournalRoot = @(Get-ChildItem -LiteralPath $target -Directory -Force |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'git-transport') } |
            ForEach-Object { $_.FullName })
    }
}

$locks = [Collections.Generic.List[IO.FileStream]]::new()
$plans = [Collections.Generic.List[object]]::new()
$receipt = $null
try {
    foreach ($journalArgument in $JournalRoot | Select-Object -Unique) {
        try {
            $journal = if ([IO.Path]::IsPathRooted($journalArgument)) {
                [IO.Path]::GetFullPath($journalArgument)
            } else { [IO.Path]::GetFullPath((Join-Path $repo $journalArgument)) }
            $journal = $journal.TrimEnd('\', '/')
            if (-not $journal.StartsWith($targetPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Journal must be strictly inside this checkout's target directory: $journal"
            }
            Assert-PlainPath $journal
            $transport = Join-Path $journal 'git-transport'
            Assert-PlainPath $transport
            if (-not (Test-Path -LiteralPath $transport -PathType Container)) {
                throw "No Git transport directory exists: $transport"
            }
            if ($CleanTransport) {
                # Same lock path as ManagedTransportWorkspace. Exclusive sharing and
                # an overlapping byte-range lock reject a live native publisher.
                $lockPath = Join-Path $journal 'git-transport.lock'
                Assert-PlainPath $lockPath
                $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate,
                    [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
                $locks.Add($lock)
                $lock.Lock(0, 1)
            }
            $snapshot = Measure-Tree $transport -Strict
            $relative = $transport.Substring($repo.Length + 1).Replace('\', '/')
            $tracked = @(& git -C $repo ls-files -- $relative)
            if ($LASTEXITCODE -ne 0 -or $tracked.Count) {
                throw "Refusing transport containing tracked files: $transport"
            }
            $bareRoots = @{}
            foreach ($file in $snapshot.files) {
                $relativeFile = $file.path.Replace('\', '/')
                if ($relativeFile -notmatch '^(?<session>(?:.*/)?[0-9a-f]{64})/(?:remote\.git/.+|publication-(?:root-)?index-[0-9a-f]{64}(?:\.lock)?)$') {
                    throw "Unrecognized transport contents: $relative/$relativeFile"
                }
                # Git's scratch index and lock can survive an interrupted write
                # beside remote.git. They must belong to a verified bare session.
                $bareRelative = $Matches.session + '/remote.git'
                $bareRoots[$bareRelative] = $true
            }
            foreach ($bareRelative in $bareRoots.Keys) {
                $barePath = Join-Path $transport $bareRelative
                $bare = & git config --file (Join-Path $barePath 'config') --get core.bare
                if ($LASTEXITCODE -ne 0 -or $bare -ne 'true') { throw "Not a temporary bare repository: $barePath" }
                $localRefs = @(& git --git-dir=$barePath for-each-ref --format='%(refname)' refs/heads refs/tags)
                if ($LASTEXITCODE -ne 0 -or $localRefs.Count) {
                    throw "Refusing transport with local branches or tags: $barePath"
                }
            }
            $oldEnough = $snapshot.newest_utc -lt [datetime]::UtcNow.AddDays(-$MinimumAgeDays)
            if ($CleanTransport -and -not $oldEnough) { throw "Transport was modified recently: $transport" }
            $record = [ordered]@{
                path = $relative; bytes = $snapshot.bytes; files = $snapshot.files.Count
                newest_utc = $snapshot.newest_utc.ToString('o'); old_enough = $oldEnough
                status = 'inspected'
            }
            $report.transports += $record
            $plans.Add([pscustomobject]@{ path = $transport; snapshot = $snapshot; record = $record })
        } catch {
            if ($CleanTransport) { throw }
            $report.transports += [ordered]@{
                path = $journalArgument; status = 'review_required'; error = $_.Exception.Message
            }
        }
    }

    if ($CleanTransport) {
        $receiptDirectory = Join-Path $target 'disk-maintenance'
        Assert-PlainPath $receiptDirectory
        [IO.Directory]::CreateDirectory($receiptDirectory) | Out-Null
        $receipt = Join-Path $receiptDirectory ("transport-cleanup-{0}-{1}.json" -f
            [datetime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'), [guid]::NewGuid().ToString('N'))
        $report.receipt = $receipt
        $report.free_bytes_before = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($repo)).AvailableFreeSpace
        $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receipt -Encoding UTF8
        foreach ($plan in $plans) {
            # Revalidate absolute containment, links, and the complete file
            # inventory immediately before deleting exactly git-transport/.
            Assert-PlainPath $plan.path
            if (-not $plan.path.StartsWith($targetPrefix, [StringComparison]::OrdinalIgnoreCase) -or
                (Split-Path -Leaf $plan.path) -ne 'git-transport') { throw 'Cleanup path escaped target.' }
            $current = Measure-Tree $plan.path -Strict
            if (($current | ConvertTo-Json -Depth 6 -Compress) -ne
                ($plan.snapshot | ConvertTo-Json -Depth 6 -Compress)) {
                throw "Transport changed after inspection: $($plan.path)"
            }
            Remove-Item -LiteralPath $plan.path -Recurse -Force
            if (Test-Path -LiteralPath $plan.path) { throw "Transport removal was incomplete: $($plan.path)" }
            $report.removed_bytes += $plan.snapshot.bytes
            $plan.record.status = 'removed'
            $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receipt -Encoding UTF8
        }
        $report.free_bytes_after = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($repo)).AvailableFreeSpace
    }
} catch {
    $report.error = $_.Exception.Message
    throw
} finally {
    foreach ($lock in $locks) { $lock.Dispose() }
    if ($receipt) { $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receipt -Encoding UTF8 }
}
$report | ConvertTo-Json -Depth 8
