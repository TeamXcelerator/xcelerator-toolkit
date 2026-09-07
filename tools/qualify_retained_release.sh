#!/usr/bin/env bash
# Run from a toolkit checkout with the documented HP native prerequisites.
# Output is operational qualification data, not a scientific campaign result.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export XC_CACHE_REMOTE=none XC_PUBLISH_TARGET=none XC_PUBLISH_EXECUTE=false
cargo build -p xc-spectral --release --features hp,arb --example qualify_retained_evidence --locked
binary="${CARGO_TARGET_DIR:-target}/release/examples/qualify_retained_evidence"
for configuration in "256 256 1" "512 256 1" "512 256 4" "1024 256 1" "512 512 1"; do
  read -r dimension bits workers <<< "$configuration"
  "$binary" "$dimension" "$bits" "$workers" rank-one
done
"$binary" 128 256 1 toeplitz
"$binary" 512 256 1 dense-reflector
"$binary" 512 256 4 dense-reflector
"$binary" 1024 256 4 dense-reflector
"$binary" 256 256 4 dense-reflector
"$binary" 256 256 1 dense-reflector third-moment
"$binary" 256 256 4 dense-reflector third-moment
"$binary" 256 256 4 dense-reflector third-moment-no-cancellation
