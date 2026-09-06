#!/usr/bin/env python3
"""One-shot review corrections, applied before qualification and then removed.
AI-generated assistance; owner-authorized scope in docs/CCM_HARDENING.md.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]

def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"ambiguous review edit: {old[:100]!r}")
    return text.replace(old, new, 1)

path = ROOT / "crates/xc-numerics/src/quadrature.rs"
data = path.read_bytes()
assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == "06c0065c0ccf43bae31c5fd00b630734c9f94476"
s = data.decode()
s = once(s, "    fn gl_cache_dir() -> Option<std::path::PathBuf> {", """    #[cfg(test)]
    thread_local! {
        static TEST_CACHE_ROOT: std::cell::RefCell<Option<std::path::PathBuf>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Test-only, thread-local cache placement. Never mutates the process
    /// environment or cwd, and never redirects another concurrently running
    /// test. Production continues to honor XC_CACHE_ROOT unchanged.
    #[cfg(test)]
    pub(super) fn replace_test_cache_root(root: Option<std::path::PathBuf>) -> Option<std::path::PathBuf> {
        TEST_CACHE_ROOT.with(|current| current.replace(root))
    }

    fn gl_cache_dir() -> Option<std::path::PathBuf> {
        #[cfg(test)]
        if let Some(root) = TEST_CACHE_ROOT.with(|current| current.borrow().clone()) {
            let dir = root.join(\"gl_cache\");
            std::fs::create_dir_all(&dir).ok()?;
            return Some(dir);
        }""")
start = s.index("    /// Guard that restores the original cwd", s.index("mod hp_cache_tests"))
end = s.index("    /// Make a fresh, unique throwaway directory", start)
s = s[:start] + """    /// A panic-safe per-thread override, independent of the user's cache
    /// environment. No global cwd mutation is necessary for cache tests.
    struct CacheRootGuard {
        original: Option<PathBuf>,
    }
    impl CacheRootGuard {
        fn enter(temp: &std::path::Path) -> Self {
            Self { original: hp::replace_test_cache_root(Some(temp.join(\"data\"))) }
        }
    }
    impl Drop for CacheRootGuard {
        fn drop(&mut self) {
            hp::replace_test_cache_root(self.original.take());
        }
    }

""" + s[end:]
s = s.replace("CwdGuard::enter", "CacheRootGuard::enter")
s = s.replace("    use std::sync::Mutex;\n", "")
start = s.index("    /// Serialize all cwd-mutating tests", s.index("mod hp_cache_tests"))
end = s.index("    #[test]", start)
s = s[:start] + s[end:]
start = s.index("    //! These tests exercise the cwd-relative", s.index("mod hp_cache_tests"))
end = s.index("    use super::*;", start)
s = s[:start] + """    //! Cache lookup tests use a thread-local test root. They neither depend
    //! on nor mutate XC_CACHE_ROOT or the process working directory, and can
    //! safely execute concurrently with numerical tests.

""" + s[end:]
s = once(s, "        serde_json::json!([nodes, weights]).to_string()", """        serde_json::json!({
            \"schema_version\": 1,
            \"toolkit_version\": hp::toolkit_version_for_test(),
            \"n_pts\": n,
            \"precision_bits\": 64,
            \"nodes\": nodes,
            \"weights\": weights,
        }).to_string()""")
path.write_text(s)

path = ROOT / "crates/xc-spectral/src/ccm/hp.rs"
s = path.read_text()
start = s.index("/// Opt-in exact-rational-input matrix assembly for mechanism experiments.")
end = s.index("#[cfg(test)]\nmod audit_research_tests", start)
helper = s[start:end]
s = s[:start] + s[end:]
anchor = "#[cfg(test)]\nmod tests {"
if anchor not in s:
    raise RuntimeError("could not locate existing HP test module")
s = s.replace(anchor, helper + anchor, 1)
start = s.index('            println!("CCM_BENCH ')
end = s.index('\n', start)
s = s[:start] + """            println!(\"CCM_BENCH {}\", serde_json::json!({
                \"cutoff\": 500, \"n_modes\": n, \"precision_bits\": p, \"samples\": 3,
                \"canonical_median_ns\": canonical[1], \"aggregate_median_ns\": aggregate[1],
                \"peak_rss_bytes\": peak_resident_memory_bytes(),
                \"scope\": \"prime_component_only_not_whole_solver_speedup\",
            }));""" + s[end:]
path.write_text(s)
print("isolated cache fixtures and finalized benchmark serialization")
