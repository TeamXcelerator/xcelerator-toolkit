#!/usr/bin/env bash
# Package paths are already SHA-256 checked by windows-bootstrap.ps1.
set -euo pipefail
(($# > 0)) || { echo 'No pinned native packages supplied.' >&2; exit 2; }
exec pacman -U --noconfirm --needed "$@"
