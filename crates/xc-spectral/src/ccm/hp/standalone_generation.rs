//! Atomic, content-addressed generations for standalone compressed caches.
//! The manifest is the commit point. Previously committed parts remain available
//! to concurrent readers; abandoned/stale generations may be removed offline.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
const MANIFEST_LIMIT: u64 = 1 << 20;
const PAYLOAD_LIMIT: u64 = 8u64 << 30;
const MAX_PARTS: usize = 8192;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    sha256: String,
    bytes: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    sha256: String,
    bytes: u64,
    parts: Vec<Part>,
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn sibling(path: &Path, suffix: &str) -> io::Result<PathBuf> {
    let mut name = path
        .file_name()
        .ok_or_else(|| invalid("cache path has no filename"))?
        .to_os_string();
    name.push(suffix);
    Ok(path.with_file_name(name))
}
fn read_file(path: &Path, limit: u64) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    if file.metadata()?.len() > limit {
        return Err(invalid("cache file exceeds byte budget"));
    }
    let mut bytes = Vec::new();
    file.take(
        limit
            .checked_add(1)
            .ok_or_else(|| invalid("cache limit overflow"))?,
    )
    .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(invalid("cache file grew beyond byte budget"));
    }
    Ok(bytes)
}
fn atomic_publish(path: &Path, bytes: &[u8], replace: bool) -> io::Result<()> {
    let mut attempts = 0;
    // create_new prevents accidental reuse, including a symlink at a stale name.
    let (temporary, mut file) = loop {
        attempts += 1;
        if attempts > 1024 {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "cache temporary-name budget exhausted",
            ));
        }
        let suffix = format!(
            ".tmp-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        );
        let temporary = sibling(path, &suffix)?;
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        if replace {
            fs::rename(&temporary, path)
        } else {
            // Published chunks are immutable. A concurrent writer may already
            // have inserted this digest; verify its bytes without replacing it.
            match fs::hard_link(&temporary, path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    if read_file(path, bytes.len() as u64).ok().as_deref() != Some(bytes) {
                        // The caller holds the generation's exclusive writer lock.
                        // Restore a corrupt chunk using the newly computed canonical bytes.
                        xc_cache::atomic_replace_cache_file(path, bytes)
                            .map_err(|e| io::Error::other(e.to_string()))?;
                    }
                }
                Err(error) => return Err(error),
            }
            fs::remove_file(&temporary)
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
/// None means no committed generation. Corrupt/unreadable generations are
/// errors; callers must not fall back to a different, older legacy payload.
pub(super) fn read(zip_path: &Path, limit: u64) -> io::Result<Option<Vec<u8>>> {
    if limit > PAYLOAD_LIMIT + (1 << 20) {
        return Err(invalid("unsupported cache read budget"));
    }
    let path = sibling(zip_path, ".manifest.json")?;
    let lock = match File::open(sibling(zip_path, ".lock")?) {
        Ok(lock) => lock,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return match fs::symlink_metadata(&path) {
                Ok(_) => Err(invalid("committed generation has no coordination lock")),
                Err(missing) if missing.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(other) => Err(other),
            };
        }
        Err(error) => return Err(error),
    };
    // The stable lock inode protects readers even on filesystems whose rename
    // implementation exposes a transient missing destination (observed on WSL).
    lock.lock_shared()?;
    let bytes = match read_file(&path, MANIFEST_LIMIT) {
        Ok(value) => value,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match fs::symlink_metadata(&path) {
                Err(missing) if missing.kind() == io::ErrorKind::NotFound => return Ok(None),
                _ => return Err(error),
            }
        }
        Err(error) => return Err(error),
    };
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| invalid(&e.to_string()))?;
    if manifest.schema != 1
        || manifest.bytes == 0
        || manifest.bytes > limit
        || !valid_digest(&manifest.sha256)
        || manifest.parts.is_empty()
        || manifest.parts.len() > MAX_PARTS
    {
        return Err(invalid("invalid cache generation manifest"));
    }
    let total = manifest.parts.iter().try_fold(0u64, |sum, p| {
        if p.bytes == 0 || !valid_digest(&p.sha256) {
            return Err(invalid("invalid cache part identity"));
        }
        sum.checked_add(p.bytes)
            .filter(|n| *n <= limit)
            .ok_or_else(|| invalid("cache part size overflow/budget"))
    })?;
    if total != manifest.bytes {
        return Err(invalid("inconsistent generation size"));
    }
    let dir = sibling(zip_path, ".chunks")?;
    let mut bytes = Vec::new();
    for part in &manifest.parts {
        let chunk = read_file(&dir.join(&part.sha256), part.bytes)?;
        if chunk.len() as u64 != part.bytes || digest(&chunk) != part.sha256 {
            return Err(invalid("cache chunk size/hash mismatch"));
        }
        bytes.extend_from_slice(&chunk);
    }
    if digest(&bytes) != manifest.sha256 {
        return Err(invalid("cache generation digest mismatch"));
    }
    Ok(Some(bytes))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Part(usize),
    BeforeCommit,
}
fn write_with_hook(
    zip_path: &Path,
    bytes: &[u8],
    part_limit: usize,
    hook: impl Fn(Stage) -> io::Result<()>,
) -> io::Result<()> {
    if bytes.is_empty()
        || bytes.len() as u64 > PAYLOAD_LIMIT
        || part_limit == 0
        || bytes.len().div_ceil(part_limit) > MAX_PARTS
    {
        return Err(invalid("cache generation exceeds write budget"));
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(sibling(zip_path, ".lock")?)?;
    lock.lock()?;
    let manifest = Manifest {
        schema: 1,
        sha256: digest(bytes),
        bytes: bytes.len() as u64,
        parts: bytes
            .chunks(part_limit)
            .map(|p| Part {
                sha256: digest(p),
                bytes: p.len() as u64,
            })
            .collect(),
    };
    let metadata = serde_json::to_vec(&manifest).map_err(|e| invalid(&e.to_string()))?;
    if metadata.len() as u64 > MANIFEST_LIMIT {
        return Err(invalid("cache manifest exceeds budget"));
    }
    let dir = sibling(zip_path, ".chunks")?;
    fs::create_dir_all(&dir)?;
    for (index, (part, chunk)) in manifest
        .parts
        .iter()
        .zip(bytes.chunks(part_limit))
        .enumerate()
    {
        atomic_publish(&dir.join(&part.sha256), chunk, false)?;
        hook(Stage::Part(index))?;
    }
    hook(Stage::BeforeCommit)?;
    atomic_publish(&sibling(zip_path, ".manifest.json")?, &metadata, true)
}
pub(super) fn write(zip_path: &Path, bytes: &[u8], part_limit: usize) -> io::Result<()> {
    write_with_hook(zip_path, bytes, part_limit, |_| Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn path(label: &str) -> PathBuf {
        crate::fresh_test_dir(label).join("source.json.zip")
    }
    #[test]
    fn generation_failed_replacement_keeps_previous_complete_payload() {
        let path = path("generation-failed");
        let old = b"previous valid compressed payload";
        write(&path, old, 7).unwrap();
        for fail in [Stage::Part(0), Stage::Part(2), Stage::BeforeCommit] {
            assert!(write_with_hook(
                &path,
                b"a different complete compressed generation",
                6,
                |stage| if stage == fail {
                    Err(io::Error::other("injected interrupted write"))
                } else {
                    Ok(())
                }
            )
            .is_err());
            assert_eq!(read(&path, 100).unwrap().unwrap(), old);
        }
        write(&path, b"replacement", 3).unwrap();
        assert_eq!(read(&path, 100).unwrap().unwrap(), b"replacement");
    }
    #[test]
    fn generation_rejects_corrupt_missing_oversize_and_unsafe_parts() {
        let path = path("generation-corrupt");
        write(&path, b"abcdefghijkl", 4).unwrap();
        let manifest_path = sibling(&path, ".manifest.json").unwrap();
        let original = fs::read(&manifest_path).unwrap();
        let m: Manifest = serde_json::from_slice(&original).unwrap();
        let part = sibling(&path, ".chunks").unwrap().join(&m.parts[0].sha256);
        fs::write(&part, b"XXXX").unwrap();
        assert!(read(&path, 100).is_err());
        fs::remove_file(&part).unwrap();
        assert!(read(&path, 100).is_err());
        write(&path, b"abcdefghijkl", 4).unwrap();
        assert!(read(&path, 11).is_err());
        for field in ["sha256", "bytes", "schema", "parts"] {
            let mut v: serde_json::Value = serde_json::from_slice(&original).unwrap();
            v[field] = match field {
                "sha256" => "../escape".into(),
                "bytes" => u64::MAX.into(),
                "schema" => 2.into(),
                _ => serde_json::json!([{ "sha256":"../escape", "bytes":12 }]),
            };
            fs::write(&manifest_path, v.to_string()).unwrap();
            assert!(read(&path, 100).is_err());
        }
        fs::write(&manifest_path, vec![b' '; MANIFEST_LIMIT as usize + 1]).unwrap();
        assert!(read(&path, 100).is_err());
        assert!(write(&path, b"", 1).is_err());
        assert!(write(&path, b"x", 0).is_err());
    }
    #[test]
    fn generation_concurrent_readers_observe_whole_committed_generations() {
        let path = path("generation-concurrent");
        let a = vec![b'a'; 1001];
        let b = vec![b'b'; 2011];
        write(&path, &a, 97).unwrap();
        std::thread::scope(|scope| {
            for bytes in [&a, &b] {
                let path = &path;
                scope.spawn(move || {
                    for _ in 0..12 {
                        write(path, bytes, 97).unwrap();
                    }
                });
            }
            for _ in 0..3 {
                let path = &path;
                let a = &a;
                let b = &b;
                scope.spawn(move || {
                    for _ in 0..80 {
                        let got = read(path, 4096).unwrap().unwrap();
                        assert!(got == *a || got == *b);
                    }
                });
            }
        });
    }
    #[test]
    fn generation_absent_and_uncommitted_parts_do_not_publish() {
        let path = path("generation-uncommitted");
        assert_eq!(read(&path, 100).unwrap(), None);
        assert!(
            write_with_hook(&path, b"not committed", 3, |_| Err(io::Error::other(
                "interrupted"
            )))
            .is_err()
        );
        assert_eq!(read(&path, 100).unwrap(), None);
        let manifest = sibling(&path, ".manifest.json").unwrap();
        fs::write(&manifest, b"broken").unwrap();
        assert!(read(&path, 100).is_err());
    }
}

#[cfg(test)]
mod generation_regression_tests {
    use super::*;
    #[test]
    fn corrupt_content_addressed_chunk_is_repaired() {
        let root = crate::fresh_test_dir("remaining-corrupt-generation");
        let path = root.join("tau.zip");
        let bytes = b"independent expected canonical generation";
        write(&path, bytes, 8).unwrap();
        let chunk = sibling(&path, ".chunks").unwrap().join(digest(&bytes[..8]));
        fs::write(&chunk, b"bad").unwrap();
        assert!(read(&path, 1000).is_err());
        write(&path, bytes, 8).unwrap();
        assert_eq!(read(&path, 1000).unwrap().unwrap(), bytes);
    }
}
