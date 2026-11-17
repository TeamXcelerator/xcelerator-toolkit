// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Platform-native default location of the per-user toolkit cache.

use std::path::PathBuf;

/// The per-user cache root used when `XC_CACHE_ROOT` is not set.
///
/// `$XDG_CACHE_HOME/xcelerator` when that is set; otherwise
/// `%LOCALAPPDATA%\Xcelerator\cache` on Windows and `$HOME/.cache/xcelerator`
/// elsewhere. Only when none of these variables exist does it fall back to
/// `.xcelerator-cache` relative to the working directory.
pub fn default_cache_root() -> PathBuf {
    if let Some(root) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(root).join("xcelerator");
    }
    if cfg!(windows) {
        if let Some(root) = std::env::var_os("LOCALAPPDATA").filter(|value| !value.is_empty()) {
            return PathBuf::from(root).join("Xcelerator").join("cache");
        }
    }
    if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home).join(".cache").join("xcelerator");
    }
    PathBuf::from(".xcelerator-cache")
}

/// `XC_CACHE_ROOT` when set and non-empty, else [`default_cache_root`].
pub fn configured_cache_root() -> PathBuf {
    std::env::var_os("XC_CACHE_ROOT")
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var_os("XC_TYPED_CACHE_ROOT").filter(|v| !v.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(default_cache_root)
}

/// Resolve the root for an actual cache operation. An explicitly empty override
/// is a configuration error, so a malformed invocation cannot populate the default cache.
pub fn checked_cache_root() -> std::io::Result<PathBuf> {
    for name in ["XC_CACHE_ROOT", "XC_TYPED_CACHE_ROOT"] {
        if let Some(value) = std::env::var_os(name) {
            if value.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("{name} must not be empty"),
                ));
            }
            return Ok(PathBuf::from(value));
        }
    }
    Ok(default_cache_root())
}
