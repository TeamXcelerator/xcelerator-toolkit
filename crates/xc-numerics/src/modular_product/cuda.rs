// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Runtime-loaded CUDA/cuBLAS boundary. Each scoped worker owns its device,
//! handle, and reusable buffers. No CUDA installation is required at link time.

use super::{centered, CudaDeviceInfo, CudaStatus};
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

// Failed synchronization/ownership can make freeing live transfer storage
// unsafe. Quarantine then disables new CUDA engines for this process, so a
// later invocation cannot repeatedly allocate beyond a released admission.
static CUDA_QUARANTINED: AtomicBool = AtomicBool::new(false);
fn require_usable_backend() -> Result<()> {
    ensure!(
        !CUDA_QUARANTINED.load(Ordering::Acquire),
        "CUDA backend quarantined after an unresolved resource cleanup failure"
    );
    Ok(())
}

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(handle: *mut c_void) -> c_int;
}

struct Library(*mut c_void);
// Loaded libraries are immutable, held for the process lifetime by API, and
// their documented CUDA entry points support concurrent host threads.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Library {
    fn open(names: &[&str]) -> Result<Self> {
        let mut roots = Vec::new();
        if let Some(path) = std::env::var_os("XC_CUDA_LIB_DIR") {
            roots.extend(std::env::split_paths(&path));
        }
        if let Some(path) = std::env::var_os("CUDA_PATH") {
            let root = PathBuf::from(path);
            if cfg!(windows) {
                roots.push(root.join("bin/x64"));
            }
            roots.push(root.join(if cfg!(windows) { "bin" } else { "lib64" }));
        }
        // Standard loader search is last. On Windows LOAD_LIBRARY_SEARCH flags
        // exclude the working directory and search the DLL's own dependencies.
        let paths: Vec<_> = roots
            .iter()
            .flat_map(|r| names.iter().map(|n| r.join(n)))
            .chain(names.iter().map(PathBuf::from))
            .collect();
        for path in paths {
            #[cfg(unix)]
            let handle = {
                use std::os::unix::ffi::OsStrExt;
                let name = CString::new(path.as_os_str().as_bytes())?;
                unsafe { dlopen(name.as_ptr(), 2) }
            };
            #[cfg(windows)]
            let handle = {
                use std::os::windows::ffi::OsStrExt;
                let name: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
                let flags = 0x00001000 | if path.is_absolute() { 0x00000100 } else { 0 };
                unsafe { LoadLibraryExW(name.as_ptr(), ptr::null_mut(), flags) }
            };
            #[cfg(not(any(unix, windows)))]
            let handle: *mut c_void = ptr::null_mut();
            if !handle.is_null() {
                return Ok(Self(handle));
            }
        }
        bail!(
            "could not load any of {} (set XC_CUDA_LIB_DIR to the installed CUDA library directories)",
            names.join(", ")
        )
    }
    unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T> {
        let name = CString::new(name)?;
        #[cfg(unix)]
        let address = unsafe { dlsym(self.0, name.as_ptr()) };
        #[cfg(windows)]
        let address = unsafe { GetProcAddress(self.0, name.as_ptr()) };
        #[cfg(not(any(unix, windows)))]
        let address: *mut c_void = ptr::null_mut();
        ensure!(
            !address.is_null(),
            "missing CUDA symbol {}",
            name.to_string_lossy()
        );
        ensure!(
            std::mem::size_of::<T>() == std::mem::size_of_val(&address),
            "unsupported function pointer ABI"
        );
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            dlclose(self.0);
        }
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.0);
        }
    }
}

type Gemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type BatchedGemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    i64,
    *const c_void,
    c_int,
    c_int,
    i64,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    i64,
    c_int,
    c_int,
    c_int,
) -> c_int;

// Optional symbols are resolved only when the experimental route is requested;
// an older installation can continue to use the synchronous exact engine.
struct PipelineApi {
    host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> c_int,
    host_free: unsafe extern "C" fn(*mut c_void) -> c_int,
    stream_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> c_int,
    stream_sync: unsafe extern "C" fn(*mut c_void) -> c_int,
    stream_destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    copy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> c_int,
    set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    gemm: BatchedGemm,
}
impl PipelineApi {
    fn load(api: &Api) -> Result<Self> {
        unsafe {
            Ok(Self {
                host_alloc: api._runtime_library.symbol("cudaHostAlloc")?,
                host_free: api._runtime_library.symbol("cudaFreeHost")?,
                stream_create: api._runtime_library.symbol("cudaStreamCreateWithFlags")?,
                stream_sync: api._runtime_library.symbol("cudaStreamSynchronize")?,
                stream_destroy: api._runtime_library.symbol("cudaStreamDestroy")?,
                copy_async: api._runtime_library.symbol("cudaMemcpyAsync")?,
                set_stream: api._blas_library.symbol("cublasSetStream_v2")?,
                gemm: api._blas_library.symbol("cublasGemmStridedBatchedEx")?,
            })
        }
    }
}

const PIPELINE_LANES: usize = 2;
const DIGIT_PRODUCTS: usize = 3;
const BLAS_WORKSPACE_BYTES: usize = 4 << 20;
const DEFAULT_PIPELINE_MODULI_PER_DEVICE: usize = 128;

/// Explicit overrides and the bounded default policy select an execution mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ExecutionMode {
    #[default]
    Synchronous,
    Pipelined,
}
impl ExecutionMode {
    fn parse(text: Option<&str>) -> Result<Self> {
        match text {
            None | Some("0") => Ok(Self::Synchronous),
            Some("1") => Ok(Self::Pipelined),
            _ => bail!("XC_CUDA_PRODUCT_PIPELINE must be 0 or 1"),
        }
    }
    #[cfg(test)]
    pub(super) fn from_env() -> Result<Self> {
        match std::env::var("XC_CUDA_PRODUCT_PIPELINE") {
            Ok(text) => Self::parse(Some(&text)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(error).context("invalid XC_CUDA_PRODUCT_PIPELINE"),
        }
    }
    fn product_request(text: Option<&str>, moduli: usize) -> Result<(Self, bool)> {
        let explicit = Self::parse(text)?;
        if text.is_none() && moduli >= DEFAULT_PIPELINE_MODULI_PER_DEVICE {
            Ok((Self::Pipelined, true))
        } else {
            Ok((explicit, false))
        }
    }
    pub(super) fn for_product(moduli: usize) -> Result<(Self, bool)> {
        match std::env::var("XC_CUDA_PRODUCT_PIPELINE") {
            Ok(text) => Self::product_request(Some(&text), moduli),
            Err(std::env::VarError::NotPresent) => Self::product_request(None, moduli),
            Err(error) => Err(error).context("invalid XC_CUDA_PRODUCT_PIPELINE"),
        }
    }
    pub(super) fn batch_size(self) -> usize {
        match self {
            Self::Synchronous => 1,
            Self::Pipelined => PIPELINE_LANES,
        }
    }
}

pub(super) fn pipeline_symbols_available() -> Result<bool> {
    require_usable_backend()?;
    let api = api()?;
    Ok(PipelineApi::load(&api).is_ok())
}

#[derive(Debug)]
struct DeviceHeadroomLimit(usize);
impl std::fmt::Display for DeviceHeadroomLimit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "selected CUDA device {} lacks free memory after headroom",
            self.0
        )
    }
}
impl std::error::Error for DeviceHeadroomLimit {}
fn is_admission_limit(error: &anyhow::Error) -> bool {
    error.downcast_ref::<DeviceHeadroomLimit>().is_some()
        || error
            .downcast_ref::<super::ModularWorkspaceLimit>()
            .is_some()
}

fn same_devices(left: &[CudaDeviceInfo], right: &[CudaDeviceInfo]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.ordinal == b.ordinal)
}

fn default_pipeline_work(moduli: usize, devices: usize) -> bool {
    devices > 0 && moduli / devices >= DEFAULT_PIPELINE_MODULI_PER_DEVICE
}

pub(super) fn default_pipeline_devices(
    moduli: usize,
    synchronous: &[CudaDeviceInfo],
    available: impl FnOnce() -> Result<bool>,
    select: impl FnOnce() -> Result<(Vec<CudaDeviceInfo>, u64)>,
) -> Result<Option<(Vec<CudaDeviceInfo>, u64)>> {
    if !default_pipeline_work(moduli, synchronous.len()) || !available()? {
        return Ok(None);
    }
    match select() {
        Ok((devices, bytes)) if same_devices(synchronous, &devices) => Ok(Some((devices, bytes))),
        Ok(_) => Ok(None),
        Err(error) if is_admission_limit(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

struct Api {
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    free: unsafe extern "C" fn(*mut c_void) -> c_int,
    copy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    count: unsafe extern "C" fn(*mut c_int) -> c_int,
    get_device: unsafe extern "C" fn(*mut c_int) -> c_int,
    set_device: unsafe extern "C" fn(c_int) -> c_int,
    pci_bus_id: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    memory: unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
    driver: unsafe extern "C" fn(*mut c_int) -> c_int,
    runtime: unsafe extern "C" fn(*mut c_int) -> c_int,
    create: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    version: unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int,
    property: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    workspace: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int,
    gemm: Gemm,
    _runtime_library: Library,
    _blas_library: Library,
}
impl Api {
    fn load() -> Result<Self> {
        let runtime = Library::open(if cfg!(windows) {
            &["cudart64_13.dll", "cudart64_12.dll"]
        } else {
            &["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        })?;
        let blas = Library::open(if cfg!(windows) {
            &["cublas64_13.dll", "cublas64_12.dll"]
        } else {
            &["libcublas.so.13", "libcublas.so.12", "libcublas.so"]
        })?;
        // Each symbol is assigned the documented C ABI signature. The library
        // owners remain in Api until every borrowed function pointer is dead.
        unsafe {
            Ok(Self {
                malloc: runtime.symbol("cudaMalloc")?,
                free: runtime.symbol("cudaFree")?,
                copy: runtime.symbol("cudaMemcpy")?,
                count: runtime.symbol("cudaGetDeviceCount")?,
                get_device: runtime.symbol("cudaGetDevice")?,
                set_device: runtime.symbol("cudaSetDevice")?,
                pci_bus_id: runtime.symbol("cudaDeviceGetPCIBusId")?,
                attribute: runtime.symbol("cudaDeviceGetAttribute")?,
                memory: runtime.symbol("cudaMemGetInfo")?,
                driver: runtime.symbol("cudaDriverGetVersion")?,
                runtime: runtime.symbol("cudaRuntimeGetVersion")?,
                create: blas.symbol("cublasCreate_v2")?,
                destroy: blas.symbol("cublasDestroy_v2")?,
                version: blas.symbol("cublasGetVersion_v2")?,
                workspace: blas.symbol("cublasSetWorkspace_v2")?,
                property: blas.symbol("cublasGetProperty")?,
                gemm: blas.symbol("cublasGemmEx")?,
                _runtime_library: runtime,
                _blas_library: blas,
            })
        }
    }
}
fn api() -> Result<Arc<Api>> {
    static API: OnceLock<std::result::Result<Arc<Api>, String>> = OnceLock::new();
    match API.get_or_init(|| Api::load().map(Arc::new).map_err(|e| e.to_string())) {
        Ok(api) => Ok(Arc::clone(api)),
        Err(message) => bail!("{message}"),
    }
}
fn checked(status: c_int, name: &str) -> Result<()> {
    ensure!(
        status == 0,
        "CUDA operation {name} returned status {status}"
    );
    Ok(())
}
fn ordinals(text: Option<&str>, count: usize) -> Result<Vec<usize>> {
    let Some(text) = text else {
        return Ok((0..count).collect());
    };
    ensure!(
        !text.trim().is_empty(),
        "XC_CUDA_DEVICES must contain visible device ordinals"
    );
    let mut result = Vec::new();
    for item in text.split(',') {
        let ordinal: usize = item
            .trim()
            .parse()
            .context("invalid XC_CUDA_DEVICES ordinal")?;
        ensure!(
            ordinal < count,
            "XC_CUDA_DEVICES ordinal {ordinal} is not visible (visible count {count})"
        );
        ensure!(
            !result.contains(&ordinal),
            "duplicate XC_CUDA_DEVICES ordinal {ordinal}"
        );
        result.push(ordinal);
    }
    Ok(result)
}
// Keep device state separate from the loaded API so ownership and restoration
// failures can be exercised without a CUDA installation.
struct DeviceSelection {
    owner: c_int,
    previous: Option<c_int>,
}
impl DeviceSelection {
    fn select(
        owner: c_int,
        get: impl FnOnce(&mut c_int) -> c_int,
        mut set: impl FnMut(c_int) -> c_int,
    ) -> Result<Self> {
        let mut previous = 0;
        checked(get(&mut previous), "get current device")?;
        if let Err(error) = checked(set(owner), "select device") {
            return with_cleanup(
                Err(error),
                checked(
                    set(previous),
                    "restore previous device after selection failure",
                ),
            );
        }
        Ok(Self {
            owner,
            previous: Some(previous),
        })
    }
    fn reassert(&self, set: impl FnOnce(c_int) -> c_int) -> Result<()> {
        checked(set(self.owner), "reselect owning device")
    }
    fn restore(&mut self, set: impl FnOnce(c_int) -> c_int) -> Result<()> {
        if let Some(previous) = self.previous {
            checked(set(previous), "restore previous device")?;
            self.previous = None;
        }
        Ok(())
    }
}

// Always attempt cleanup. The original operation remains the root cause if
// cleanup also fails; a cleanup-only failure prevents successful acceptance.
fn with_cleanup<T>(operation: Result<T>, cleanup: Result<()>) -> Result<T> {
    match (operation, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) | (Err(error), Ok(())) => Err(error),
        (Err(error), Err(cleanup)) => {
            Err(error.context(format!("CUDA cleanup also failed: {cleanup:#}")))
        }
    }
}

struct DeviceGuard {
    api: Arc<Api>,
    selection: DeviceSelection,
    // Device selection is OS-thread-local, including the unwind fallback.
    _owning_thread: PhantomData<Rc<()>>,
}
impl DeviceGuard {
    fn select(api: Arc<Api>, ordinal: usize) -> Result<Self> {
        let owner = c_int::try_from(ordinal).context("CUDA device ordinal exceeds index domain")?;
        let selection = DeviceSelection::select(
            owner,
            |previous| unsafe { (api.get_device)(previous) },
            |device| unsafe { (api.set_device)(device) },
        )?;
        Ok(Self {
            api,
            selection,
            _owning_thread: PhantomData,
        })
    }
    fn reassert(&self) -> Result<()> {
        self.selection
            .reassert(|device| unsafe { (self.api.set_device)(device) })
    }
    fn restore(&mut self) -> Result<()> {
        let set = self.api.set_device;
        self.selection.restore(|device| unsafe { set(device) })
    }
}
impl Drop for DeviceGuard {
    fn drop(&mut self) {
        // Successful paths explicitly check restoration; Drop covers unwinds
        // and retries a previously reported restoration failure.
        let _ = self.restore();
    }
}

fn pci_bus_id_text(bytes: &[u8]) -> Result<String> {
    let text = CStr::from_bytes_until_nul(bytes)
        .context("CUDA PCI bus ID is not NUL-terminated")?
        .to_str()
        .context("CUDA PCI bus ID is not valid UTF-8")?;
    ensure!(!text.is_empty(), "CUDA PCI bus ID is empty");
    Ok(text.to_owned())
}

pub(super) fn status() -> Result<CudaStatus> {
    let selection = std::env::var("XC_CUDA_DEVICES").ok();
    // Invalid configuration is still an error on a host without CUDA libraries.
    // The actual visibility bound is applied after device enumeration.
    if selection.is_some() {
        ordinals(selection.as_deref(), usize::MAX)?;
    }
    if let Err(error) = require_usable_backend() {
        return Ok(CudaStatus::unavailable(error.to_string()));
    }
    let api = match api() {
        Ok(api) => api,
        Err(e) => return Ok(CudaStatus::unavailable(e.to_string())),
    };
    let mut result = CudaStatus::default();
    let (mut driver, mut runtime, mut count) = (0, 0, 0);
    unsafe {
        if let Err(e) = checked((api.driver)(&mut driver), "query driver version")
            .and_then(|()| checked((api.runtime)(&mut runtime), "query runtime version"))
            .and_then(|()| checked((api.count)(&mut count), "enumerate devices"))
        {
            return Ok(CudaStatus::unavailable(e.to_string()));
        }
    }
    result.driver_version = Some(driver);
    result.runtime_version = Some(runtime);
    let mut blas_parts = [0; 3];
    for (kind, part) in blas_parts.iter_mut().enumerate() {
        unsafe {
            checked(
                (api.property)(kind as c_int, part),
                "query cuBLAS version property",
            )?;
        }
    }
    let blas_version = blas_parts[0] * 10_000 + blas_parts[1] * 100 + blas_parts[2];
    result.cublas_version = Some(blas_version);
    if !(12_000..14_000).contains(&runtime) {
        result.unavailable_reason = Some(format!(
            "CUDA runtime {runtime} is outside supported major versions 12 and 13"
        ));
        return Ok(result);
    }
    ensure!(
        (120_000..140_000).contains(&blas_version),
        "cuBLAS version {blas_version} is outside supported major versions 12 and 13"
    );
    // A driver newer than the runtime is expected. Minor-version compatibility
    // is determined by the runtime itself, not by comparing version numbers.
    for ordinal in ordinals(selection.as_deref(), count.max(0) as usize)? {
        let mut guard = DeviceGuard::select(Arc::clone(&api), ordinal)?;
        let query = (|| -> Result<Option<CudaDeviceInfo>> {
            let (mut major, mut minor, mut free, mut total) = (0, 0, 0, 0);
            let mut pci_bus_id = [0u8; 32];
            unsafe {
                checked(
                    (api.attribute)(&mut major, 75, ordinal as c_int),
                    "query compute capability major",
                )?;
                checked(
                    (api.attribute)(&mut minor, 76, ordinal as c_int),
                    "query compute capability minor",
                )?;
                checked((api.memory)(&mut free, &mut total), "query device memory")?;
                checked(
                    (api.pci_bus_id)(
                        pci_bus_id.as_mut_ptr().cast(),
                        pci_bus_id.len() as c_int,
                        ordinal as c_int,
                    ),
                    "query device PCI bus ID",
                )?;
            }
            if major < 7 || (major == 7 && minor < 5) {
                ensure!(
                    selection.is_none(),
                    "selected CUDA device {ordinal} lacks compute capability 7.5"
                );
                return Ok(None);
            }
            let pci_bus_id = pci_bus_id_text(&pci_bus_id)?;
            Ok(Some(CudaDeviceInfo {
                ordinal,
                pci_bus_id,
                compute_major: major,
                compute_minor: minor,
                free_memory_bytes: free as u64,
                total_memory_bytes: total as u64,
            }))
        })();
        if let Some(device) = with_cleanup(query, guard.restore())? {
            result.devices.push(device);
        }
    }
    result.available = !result.devices.is_empty();
    if !result.available {
        result.unavailable_reason = Some("no compatible visible CUDA devices".to_owned());
    }
    Ok(result)
}

pub(super) fn workspace(m: usize, k: usize, n: usize) -> Result<(u64, u64)> {
    let (pm, pk, pn) = (padded(m)?, padded_reduction(k)?, padded(n)?);
    let (a, b, c) = (
        pm as u128 * pk as u128,
        pk as u128 * pn as u128,
        pm as u128 * pn as u128,
    );
    // Explicit cuBLAS workspace prevents an unbounded implicit algorithm pool.
    let device = a + b + 4 * c + (4 << 20);
    let host = 2 * (a + b)
        + 12 * c
        + 2 * (m as u128 * k as u128 + k as u128 * n as u128 + m as u128 * n as u128);
    Ok((
        u64::try_from(device).context("device workspace overflow")?,
        u64::try_from(device + host + (1 << 20)).context("aggregate CUDA workspace overflow")?,
    ))
}
pub(super) fn workspace_for_mode(
    m: usize,
    k: usize,
    n: usize,
    mode: ExecutionMode,
) -> Result<(u64, u64)> {
    if mode == ExecutionMode::Synchronous {
        return workspace(m, k, n);
    }
    let shape = PipelineShape::new(m, k, n)?;
    // Each lane owns three device operands/results, the matching pinned host
    // staging and a private bounded cuBLAS workspace. Pageable gathered residue
    // operands and returned answers are charged separately while both lanes
    // are in flight. The setup self-test fits the additional 1 MiB allowance.
    let staging = shape.staging_bytes();
    let device = PIPELINE_LANES as u128 * (staging + BLAS_WORKSPACE_BYTES as u128);
    let host = PIPELINE_LANES as u128
        * (staging + 2 * (m as u128 * k as u128 + k as u128 * n as u128 + m as u128 * n as u128));
    Ok((
        u64::try_from(device).context("pipeline device workspace overflow")?,
        u64::try_from(device + host + (1 << 20))
            .context("aggregate pipeline workspace overflow")?,
    ))
}
fn padded(value: usize) -> Result<usize> {
    let value = value.checked_add(3).context("CUDA dimension overflow")? & !3;
    ensure!(
        value <= i32::MAX as usize,
        "padded matrix exceeds CUDA index domain"
    );
    Ok(value)
}

fn padded_reduction(value: usize) -> Result<usize> {
    // Some CUDA integer kernels reject non-tiled reduction sizes even when
    // the documented four-byte leading-dimension requirements are satisfied.
    // Ampere also rejects some rectangular products with 16-element reduction
    // tiles (for example a padded 152x112x152 product). Use complete 32-element
    // reduction tiles in both engines and their admission models. Packing
    // fills the added entries with zeros, so the exact product is unchanged.
    let value = value.checked_add(31).context("CUDA dimension overflow")? & !31;
    ensure!(
        value <= i32::MAX as usize,
        "padded matrix exceeds CUDA index domain"
    );
    Ok(value)
}
pub(super) fn choose_devices(
    status: &CudaStatus,
    device_bytes: u64,
    aggregate_bytes: u64,
    available_budget: u64,
    workers: usize,
    moduli: usize,
    require_device: bool,
) -> Result<Vec<CudaDeviceInfo>> {
    let headroom = match std::env::var("XC_CUDA_HEADROOM_MIB") {
        Ok(text) => text
            .parse::<u64>()
            .context("invalid XC_CUDA_HEADROOM_MIB")?
            .checked_mul(1 << 20)
            .context("CUDA headroom overflow")?,
        Err(_) => 512 << 20,
    };
    choose_devices_with_policy(
        status,
        device_bytes,
        aggregate_bytes,
        available_budget,
        workers,
        moduli,
        headroom,
        std::env::var_os("XC_CUDA_DEVICES").is_some(),
        require_device,
    )
}
#[allow(clippy::too_many_arguments)]
fn choose_devices_with_policy(
    status: &CudaStatus,
    device_bytes: u64,
    aggregate_bytes: u64,
    available_budget: u64,
    workers: usize,
    moduli: usize,
    headroom: u64,
    explicit: bool,
    require_device: bool,
) -> Result<Vec<CudaDeviceInfo>> {
    let count = workers
        .min(moduli)
        .min((available_budget / aggregate_bytes.max(1)) as usize);
    let mut devices = Vec::new();
    let mut any_fits = false;
    for device in &status.devices {
        let reserve = headroom.max(device.total_memory_bytes / 10);
        let fits = device.free_memory_bytes.saturating_sub(reserve) >= device_bytes;
        if explicit && !fits {
            return Err(DeviceHeadroomLimit(device.ordinal).into());
        }
        any_fits |= fits;
        if fits && devices.len() < count {
            devices.push(device.clone());
        }
    }
    // Check only after validating all explicitly selected devices. A missing
    // runtime, invalid selection, or insufficient physical VRAM is not this
    // typed host/device allocation-budget refusal. Auto retains its CPU policy.
    if require_device && status.available && any_fits && workers > 0 && moduli > 0 {
        super::admit_workspace(
            u128::from(aggregate_bytes),
            u128::from(available_budget),
            "CUDA product buffers",
        )?;
    }
    Ok(devices)
}

#[derive(Default)]
struct HostBuffers {
    a0: Vec<i8>,
    a1: Vec<i8>,
    b0: Vec<i8>,
    b1: Vec<i8>,
    c0: Vec<i32>,
    c2: Vec<i32>,
    cs: Vec<i32>,
}

// Host work borrows only CPU buffers, so nested Rayon work can use the caller's
// bounded product pool without moving a CUDA handle or device guard. Keep small
// matrices serial and give each split at least this much buffer work.
const HOST_PARALLEL_MIN_ENTRIES: usize = 16 * 1024;
const HOST_SPLIT_MIN_ENTRIES: usize = 4 * 1024;

fn parallel_host_work(entries: usize) -> bool {
    entries >= HOST_PARALLEL_MIN_ENTRIES && rayon::current_num_threads() > 1
}

fn pack_left(
    input: &[u16],
    k: usize,
    pk: usize,
    prime: u32,
    low: &mut [i8],
    high: &mut [i8],
) -> Result<()> {
    let m = input.len() / k;
    let pack_row = |(i, (low, high)): (usize, (&mut [i8], &mut [i8]))| -> Result<()> {
        if i < m {
            for t in 0..k {
                let value = input[i * k + t];
                ensure!(u32::from(value) < prime, "noncanonical left residue");
                let r = centered(value, prime);
                let hi = (r + 64).div_euclid(128);
                low[t] = (r - 128 * hi) as i8;
                high[t] = hi as i8;
            }
            low[k..].fill(0);
            high[k..].fill(0);
        } else {
            low.fill(0);
            high.fill(0);
        }
        Ok(())
    };
    if parallel_host_work(low.len()) {
        low.par_chunks_mut(pk)
            .zip(high.par_chunks_mut(pk))
            .enumerate()
            .with_min_len(HOST_SPLIT_MIN_ENTRIES.div_ceil(pk))
            .try_for_each(pack_row)
    } else {
        low.chunks_mut(pk)
            .zip(high.chunks_mut(pk))
            .enumerate()
            .try_for_each(pack_row)
    }
}

fn pack_right(
    input: &[u16],
    n: usize,
    pk: usize,
    prime: u32,
    low: &mut [i8],
    high: &mut [i8],
) -> Result<()> {
    let k = input.len() / n;
    let pack_column = |(j, (low, high)): (usize, (&mut [i8], &mut [i8]))| -> Result<()> {
        if j < n {
            for t in 0..k {
                let value = input[t * n + j];
                ensure!(u32::from(value) < prime, "noncanonical right residue");
                let r = centered(value, prime);
                let hi = (r + 64).div_euclid(128);
                low[t] = (r - 128 * hi) as i8;
                high[t] = hi as i8;
            }
            low[k..].fill(0);
            high[k..].fill(0);
        } else {
            low.fill(0);
            high.fill(0);
        }
        Ok(())
    };
    if parallel_host_work(low.len()) {
        low.par_chunks_mut(pk)
            .zip(high.par_chunks_mut(pk))
            .enumerate()
            .with_min_len(HOST_SPLIT_MIN_ENTRIES.div_ceil(pk))
            .try_for_each(pack_column)
    } else {
        low.chunks_mut(pk)
            .zip(high.chunks_mut(pk))
            .enumerate()
            .try_for_each(pack_column)
    }
}

fn recombine_products(
    c0: &[i32],
    c2: &[i32],
    cs: &[i32],
    pm: usize,
    n: usize,
    prime: u32,
    result: &mut [u16],
) {
    let recombine_row = |(i, row): (usize, &mut [u16])| {
        for (j, value) in row.iter_mut().enumerate() {
            let index = j * pm + i;
            let (low, high, sum) = (
                i64::from(c0[index]),
                i64::from(c2[index]),
                i64::from(cs[index]),
            );
            *value =
                (low + 128 * (sum - low - high) + 16384 * high).rem_euclid(i64::from(prime)) as u16;
        }
    };
    if parallel_host_work(result.len()) {
        result
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(HOST_SPLIT_MIN_ENTRIES.div_ceil(n))
            .for_each(recombine_row);
    } else {
        result.chunks_mut(n).enumerate().for_each(recombine_row);
    }
}

struct SerialEngine {
    api: Arc<Api>,
    device: DeviceGuard,
    handle: *mut c_void,
    buffers: [*mut c_void; 4],
    finished: bool,
    m: usize,
    k: usize,
    n: usize,
    pm: usize,
    pk: usize,
    pn: usize,
    host: HostBuffers,
}
impl SerialEngine {
    fn new(ordinal: usize, m: usize, k: usize, n: usize) -> Result<Self> {
        require_usable_backend()?;
        let (pm, pk, pn) = (padded(m)?, padded_reduction(k)?, padded(n)?);
        workspace(m, k, n)?;
        let api = api()?;
        let device = DeviceGuard::select(Arc::clone(&api), ordinal)?;
        let mut engine = Self {
            api,
            device,
            handle: ptr::null_mut(),
            buffers: [ptr::null_mut(); 4],
            finished: false,
            m,
            k,
            n,
            pm,
            pk,
            pn,
            host: HostBuffers {
                a0: vec![0; pm * pk],
                a1: vec![0; pm * pk],
                b0: vec![0; pk * pn],
                b1: vec![0; pk * pn],
                c0: vec![0; pm * pn],
                c2: vec![0; pm * pn],
                cs: vec![0; pm * pn],
            },
        };
        let initialized = (|| -> Result<()> {
            engine.device.reassert()?;
            unsafe {
                checked(
                    (engine.api.create)(&mut engine.handle),
                    "create BLAS handle",
                )?;
                let mut version = 0;
                checked(
                    (engine.api.version)(engine.handle, &mut version),
                    "query cuBLAS version",
                )?;
                ensure!(
                    (120_000..140_000).contains(&version),
                    "cuBLAS {version} is outside supported major versions 12 and 13"
                );
                for (pointer, size) in
                    engine
                        .buffers
                        .iter_mut()
                        .zip([pm * pk, pk * pn, 4 * pm * pn, 4 << 20])
                {
                    checked((engine.api.malloc)(pointer, size), "allocate device buffer")?;
                }
                checked(
                    (engine.api.workspace)(engine.handle, engine.buffers[3], 4 << 20),
                    "set bounded BLAS workspace",
                )?;
            }
            Ok(())
        })();
        match initialized {
            Ok(()) => Ok(engine),
            Err(error) => with_cleanup(Err(error), engine.finish()),
        }
    }
    fn self_test(ordinal: usize) -> Result<i32> {
        let mut engine = Self::new(ordinal, 2, 3, 2)?;
        let tested = (|| -> Result<i32> {
            let result = engine.product(
                &[0, 1, 16190, 8000, 16189, 3],
                &[16190, 8000, 1, 2, 16189, 3],
                16191,
            )?;
            let expected = super::cpu_residue_product(
                &[0, 1, 16190, 8000, 16189, 3],
                &[16190, 8000, 1, 2, 16189, 3],
                2,
                3,
                2,
                16191,
            );
            ensure!(
                result == expected,
                "CUDA device {ordinal} failed exact integer self-test"
            );
            let mut version = 0;
            // The CPU oracle above can yield to another device's Rayon task.
            engine.device.reassert()?;
            unsafe {
                checked(
                    (engine.api.version)(engine.handle, &mut version),
                    "query cuBLAS version",
                )?;
            }
            Ok(version)
        })();
        with_cleanup(tested, engine.finish())
    }
    fn gemm(&self, a: &[i8], b: &[i8], output: &mut [i32]) -> Result<()> {
        ensure!(
            a.len() == self.pm * self.pk && b.len() == self.pk * self.pn,
            "CUDA operand shape mismatch"
        );
        ensure!(
            output.len() == self.pm * self.pn,
            "CUDA product shape mismatch"
        );
        // Nested host work can execute another Engine on this same OS thread.
        // Check ownership immediately before every uninterrupted GPU phase.
        if let Err(error) = self.device.reassert() {
            CUDA_QUARANTINED.store(true, Ordering::Release);
            return Err(error);
        }
        let (alpha, beta) = (1i32, 0i32);
        unsafe {
            checked(
                (self.api.copy)(self.buffers[0], a.as_ptr().cast(), a.len(), 1),
                "copy left operand",
            )?;
            checked(
                (self.api.copy)(self.buffers[1], b.as_ptr().cast(), b.len(), 1),
                "copy right operand",
            )?;
            checked(
                (self.api.gemm)(
                    self.handle,
                    1,
                    0,
                    self.pm as i32,
                    self.pn as i32,
                    self.pk as i32,
                    (&alpha as *const i32).cast(),
                    self.buffers[0],
                    3,
                    self.pk as i32,
                    self.buffers[1],
                    3,
                    self.pk as i32,
                    (&beta as *const i32).cast(),
                    // CUBLAS_COMPUTE_32I retains >=32-bit integer intermediate
                    // precision and permits Tensor Cores. The partial-sum bound
                    // proves exactness for every reduction order.
                    self.buffers[2],
                    10,
                    self.pm as i32,
                    72,
                    -1,
                ),
                "exact INT8 GEMM",
            )?;
            checked(
                (self.api.copy)(
                    output.as_mut_ptr().cast(),
                    self.buffers[2],
                    output.len() * 4,
                    2,
                ),
                "copy product",
            )?;
        }
        Ok(())
    }
    fn product(&mut self, a: &[u16], b: &[u16], prime: u32) -> Result<Vec<u16>> {
        let mut host = std::mem::take(&mut self.host);
        let result = self.product_in_buffers(a, b, prime, &mut host);
        self.host = host;
        // Recombination can also yield after the last GEMM. Do not accept an
        // answer if restoring ownership before subsequent cleanup fails.
        with_cleanup(result, self.device.reassert())
    }
    fn product_in_buffers(
        &self,
        a: &[u16],
        b: &[u16],
        prime: u32,
        host: &mut HostBuffers,
    ) -> Result<Vec<u16>> {
        ensure!(
            a.len() == self.m * self.k && b.len() == self.k * self.n,
            "CUDA residue shape mismatch"
        );
        ensure!(
            (3..=16191).contains(&prime),
            "CUDA residue modulus outside exact encoding domain"
        );
        ensure!(
            (self.pk as u128) * 127 * 127 <= i32::MAX as u128,
            "CUDA integer accumulation domain exceeded"
        );
        let HostBuffers {
            a0,
            a1,
            b0,
            b1,
            c0,
            c2,
            cs,
        } = host;
        // Capture only dimensions and CPU slices in work-stealing closures.
        // Every padded digit is overwritten on reuse; no whole-buffer clear is
        // needed before packing the active entries.
        let (k, n, pk) = (self.k, self.n, self.pk);
        if parallel_host_work(a0.len() + b0.len()) {
            let (left, right) = rayon::join(
                || pack_left(a, k, pk, prime, a0, a1),
                || pack_right(b, n, pk, prime, b0, b1),
            );
            // Preserve left-operand error precedence if both are invalid.
            left?;
            right?;
        } else {
            pack_left(a, k, pk, prime, a0, a1)?;
            pack_right(b, n, pk, prime, b0, b1)?;
        }
        self.gemm(a0, b0, c0)?;
        self.gemm(a1, b1, c2)?;
        for (lo, hi) in a0.iter_mut().zip(a1.iter()) {
            *lo = lo.checked_add(*hi).context("INT8 digit-sum overflow")?;
        }
        for (lo, hi) in b0.iter_mut().zip(b1.iter()) {
            *lo = lo.checked_add(*hi).context("INT8 digit-sum overflow")?;
        }
        self.gemm(a0, b0, cs)?;
        let mut result = vec![0u16; self.m * self.n];
        recombine_products(c0, c2, cs, self.pm, self.n, prime, &mut result);
        Ok(result)
    }

    fn release_resources(&mut self) -> Result<()> {
        if self.handle.is_null() && self.buffers.iter().all(|pointer| pointer.is_null()) {
            return Ok(());
        }
        // If selection fails, leave all device resources untouched. Leaking
        // during an already reported CUDA failure is safer than freeing under
        // a different current device. Drop never bypasses this check.
        if let Err(error) = self.device.reassert() {
            CUDA_QUARANTINED.store(true, Ordering::Release);
            return Err(error);
        }
        let mut released = Ok(());
        let handle = std::mem::replace(&mut self.handle, ptr::null_mut());
        if !handle.is_null() {
            released = checked(unsafe { (self.api.destroy)(handle) }, "destroy BLAS handle");
        }
        for pointer in &mut self.buffers {
            let pointer = std::mem::replace(pointer, ptr::null_mut());
            if !pointer.is_null() {
                released = with_cleanup(
                    released,
                    checked(unsafe { (self.api.free)(pointer) }, "free device buffer"),
                );
            }
        }
        if released.is_err() {
            CUDA_QUARANTINED.store(true, Ordering::Release);
        }
        released
    }

    /// Complete a scheduler task, including checked cleanup and restoration.
    /// Call this even if a residue product failed; Drop is only a fallback.
    fn finish(mut self) -> Result<()> {
        // Prevent a second release after restoration, even if cleanup failed.
        self.finished = true;
        let released = self.release_resources();
        with_cleanup(released, self.device.restore())
    }
}
impl Drop for SerialEngine {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.release_resources();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PipelineShape {
    m: usize,
    k: usize,
    n: usize,
    pm: usize,
    pk: usize,
    pn: usize,
    a: usize,
    b: usize,
    c: usize,
}
impl PipelineShape {
    fn new(m: usize, k: usize, n: usize) -> Result<Self> {
        ensure!(m > 0 && k > 0 && n > 0, "CUDA dimensions must be positive");
        let (pm, pk, pn) = (padded(m)?, padded_reduction(k)?, padded(n)?);
        ensure!(
            pk as u128 * 127 * 127 <= i32::MAX as u128,
            "CUDA integer accumulation domain exceeded"
        );
        ensure!(
            pk as u128 * (4096 + 128 * (16129 + 4096 + 3969) + 16384 * 3969) <= i64::MAX as u128,
            "CUDA integer recombination domain exceeded"
        );
        let a = pm.checked_mul(pk).context("CUDA operand shape overflow")?;
        let b = pk.checked_mul(pn).context("CUDA operand shape overflow")?;
        let c = pm.checked_mul(pn).context("CUDA product shape overflow")?;
        // All pinned slices and their byte lengths must fit Rust's slice
        // address domain. The same bound proves the i64 element strides fit.
        for bytes in [3u128 * a as u128, 3u128 * b as u128, 12u128 * c as u128] {
            ensure!(
                bytes <= isize::MAX as u128,
                "CUDA pipeline storage overflow"
            );
        }
        Ok(Self {
            m,
            k,
            n,
            pm,
            pk,
            pn,
            a,
            b,
            c,
        })
    }
    fn bytes(self) -> [usize; 3] {
        [
            DIGIT_PRODUCTS * self.a,
            DIGIT_PRODUCTS * self.b,
            4 * DIGIT_PRODUCTS * self.c,
        ]
    }
    fn staging_bytes(self) -> u128 {
        self.bytes().into_iter().map(|bytes| bytes as u128).sum()
    }
}

fn pack_digit_batch(
    a: &[u16],
    b: &[u16],
    prime: u32,
    shape: PipelineShape,
    aa: &mut [i8],
    bb: &mut [i8],
) -> Result<()> {
    ensure!(
        a.len() == shape.m * shape.k && b.len() == shape.k * shape.n,
        "CUDA residue shape mismatch"
    );
    ensure!(
        (3..=16191).contains(&prime),
        "CUDA residue modulus outside exact encoding domain"
    );
    ensure!(
        aa.len() == DIGIT_PRODUCTS * shape.a && bb.len() == DIGIT_PRODUCTS * shape.b,
        "CUDA digit batch shape mismatch"
    );
    let (a0, rest) = aa.split_at_mut(shape.a);
    let (a1, asum) = rest.split_at_mut(shape.a);
    let (b0, rest) = bb.split_at_mut(shape.b);
    let (b1, bsum) = rest.split_at_mut(shape.b);
    if parallel_host_work(shape.a + shape.b) {
        let (left, right) = rayon::join(
            || pack_left(a, shape.k, shape.pk, prime, a0, a1),
            || pack_right(b, shape.n, shape.pk, prime, b0, b1),
        );
        left?;
        right?;
    } else {
        pack_left(a, shape.k, shape.pk, prime, a0, a1)?;
        pack_right(b, shape.n, shape.pk, prime, b0, b1)?;
    }
    for ((sum, low), high) in asum.iter_mut().zip(a0.iter()).zip(a1.iter()) {
        *sum = low.checked_add(*high).context("INT8 digit-sum overflow")?;
    }
    for ((sum, low), high) in bsum.iter_mut().zip(b0.iter()).zip(b1.iter()) {
        *sum = low.checked_add(*high).context("INT8 digit-sum overflow")?;
    }
    Ok(())
}

#[derive(Default)]
struct TransferState {
    pending: bool,
}
impl TransferState {
    fn begin(&mut self) -> Result<()> {
        ensure!(!self.pending, "CUDA pipeline lane has unresolved transfers");
        // Mark before the first enqueue, including an enqueue that reports an
        // earlier asynchronous failure. Only successful sync permits reuse.
        self.pending = true;
        Ok(())
    }
    fn synchronize(&mut self, sync: impl FnOnce() -> Result<()>) -> Result<()> {
        if self.pending {
            sync()?;
            self.pending = false;
        }
        Ok(())
    }
}

struct PipelineLane {
    handle: *mut c_void,
    stream: *mut c_void,
    device: [*mut c_void; 4],
    pinned: [*mut c_void; 3],
    transfers: TransferState,
}
impl Default for PipelineLane {
    fn default() -> Self {
        Self {
            handle: ptr::null_mut(),
            stream: ptr::null_mut(),
            device: [ptr::null_mut(); 4],
            pinned: [ptr::null_mut(); 3],
            transfers: TransferState::default(),
        }
    }
}
impl PipelineLane {
    // These allocations are raw rather than RAII buffers: an unresolved CUDA
    // stream must retain both host and device storage instead of freeing it.
    fn initialize(
        &mut self,
        api: &Api,
        pipeline: &PipelineApi,
        shape: PipelineShape,
    ) -> Result<()> {
        unsafe {
            checked(
                (pipeline.stream_create)(&mut self.stream, 1),
                "create nonblocking stream",
            )?;
            checked(
                (api.create)(&mut self.handle),
                "create pipeline BLAS handle",
            )?;
            checked(
                (pipeline.set_stream)(self.handle, self.stream),
                "set pipeline BLAS stream",
            )?;
            for (pointer, bytes) in self
                .device
                .iter_mut()
                .zip(shape.bytes().into_iter().chain([BLAS_WORKSPACE_BYTES]))
            {
                checked(
                    (api.malloc)(pointer, bytes),
                    "allocate pipeline device buffer",
                )?;
            }
            // SetStream resets cuBLAS's workspace, so bind it afterward, once.
            checked(
                (api.workspace)(self.handle, self.device[3], BLAS_WORKSPACE_BYTES),
                "set bounded pipeline BLAS workspace",
            )?;
            for (pointer, bytes) in self.pinned.iter_mut().zip(shape.bytes()) {
                checked(
                    (pipeline.host_alloc)(pointer, bytes, 0),
                    "allocate pinned staging",
                )?;
                ptr::write_bytes((*pointer).cast::<u8>(), 0, bytes);
            }
        }
        Ok(())
    }
    fn pack(&mut self, a: &[u16], b: &[u16], prime: u32, shape: PipelineShape) -> Result<()> {
        ensure!(
            !self.transfers.pending,
            "CUDA pipeline lane has unresolved transfers"
        );
        // Pinned allocation lengths/alignment are proved at initialization.
        // The lane is exclusive, with no pending DMA during these host writes.
        unsafe {
            pack_digit_batch(
                a,
                b,
                prime,
                shape,
                std::slice::from_raw_parts_mut(self.pinned[0].cast(), DIGIT_PRODUCTS * shape.a),
                std::slice::from_raw_parts_mut(self.pinned[1].cast(), DIGIT_PRODUCTS * shape.b),
            )
        }
    }
    fn submit(&mut self, pipeline: &PipelineApi, shape: PipelineShape) -> Result<()> {
        self.transfers.begin()?;
        let (alpha, beta) = (1i32, 0i32);
        let bytes = shape.bytes();
        unsafe {
            for (operand, size) in bytes[..2].iter().enumerate() {
                checked(
                    (pipeline.copy_async)(
                        self.device[operand],
                        self.pinned[operand],
                        *size,
                        1,
                        self.stream,
                    ),
                    "enqueue pinned operand transfer",
                )?;
            }
            checked(
                (pipeline.gemm)(
                    self.handle,
                    1,
                    0,
                    shape.pm as i32,
                    shape.pn as i32,
                    shape.pk as i32,
                    (&alpha as *const i32).cast(),
                    self.device[0],
                    3,
                    shape.pk as i32,
                    shape.a as i64,
                    self.device[1],
                    3,
                    shape.pk as i32,
                    shape.b as i64,
                    (&beta as *const i32).cast(),
                    self.device[2],
                    10,
                    shape.pm as i32,
                    shape.c as i64,
                    DIGIT_PRODUCTS as c_int,
                    72,
                    -1,
                ),
                "exact strided batched INT8 GEMM",
            )?;
        }
        Ok(())
    }
    fn submit_result(&mut self, pipeline: &PipelineApi, shape: PipelineShape) -> Result<()> {
        let bytes = shape.bytes();
        unsafe {
            checked(
                (pipeline.copy_async)(self.pinned[2], self.device[2], bytes[2], 2, self.stream),
                "enqueue pinned product transfer",
            )?;
        }
        Ok(())
    }
    fn synchronize(&mut self, pipeline: &PipelineApi) -> Result<()> {
        let stream = self.stream;
        let result = self.transfers.synchronize(|| {
            checked(
                unsafe { (pipeline.stream_sync)(stream) },
                "synchronize pipeline stream",
            )
        });
        if result.is_err() {
            CUDA_QUARANTINED.store(true, Ordering::Release);
        }
        result
    }
    fn result(&self, shape: PipelineShape, prime: u32) -> Vec<u16> {
        debug_assert!(!self.transfers.pending);
        // Called only after successful submission AND stream synchronization.
        let products = unsafe {
            std::slice::from_raw_parts(self.pinned[2].cast::<i32>(), DIGIT_PRODUCTS * shape.c)
        };
        let mut result = vec![0; shape.m * shape.n];
        recombine_products(
            &products[..shape.c],
            &products[shape.c..2 * shape.c],
            &products[2 * shape.c..],
            shape.pm,
            shape.n,
            prime,
            &mut result,
        );
        result
    }
    fn release(&mut self, api: &Api, pipeline: &PipelineApi) -> Result<()> {
        // cudaStreamDestroy itself need not wait for queued work. Never use it
        // as the lifetime barrier for pinned buffers or a private workspace.
        self.synchronize(pipeline)?;
        let mut released = Ok(());
        let handle = std::mem::replace(&mut self.handle, ptr::null_mut());
        if !handle.is_null() {
            released = checked(
                unsafe { (api.destroy)(handle) },
                "destroy pipeline BLAS handle",
            );
        }
        let stream = std::mem::replace(&mut self.stream, ptr::null_mut());
        if !stream.is_null() {
            released = with_cleanup(
                released,
                checked(
                    unsafe { (pipeline.stream_destroy)(stream) },
                    "destroy pipeline stream",
                ),
            );
        }
        for pointer in &mut self.device {
            let pointer = std::mem::replace(pointer, ptr::null_mut());
            if !pointer.is_null() {
                released = with_cleanup(
                    released,
                    checked(
                        unsafe { (api.free)(pointer) },
                        "free pipeline device buffer",
                    ),
                );
            }
        }
        for pointer in &mut self.pinned {
            let pointer = std::mem::replace(pointer, ptr::null_mut());
            if !pointer.is_null() {
                released = with_cleanup(
                    released,
                    checked(
                        unsafe { (pipeline.host_free)(pointer) },
                        "free pinned staging",
                    ),
                );
            }
        }
        if released.is_err() {
            CUDA_QUARANTINED.store(true, Ordering::Release);
        }
        released
    }
}

struct PipelineEngine {
    api: Arc<Api>,
    pipeline: PipelineApi,
    device: DeviceGuard,
    lanes: [PipelineLane; PIPELINE_LANES],
    shape: PipelineShape,
    finished: bool,
}
impl PipelineEngine {
    fn new(ordinal: usize, m: usize, k: usize, n: usize) -> Result<Self> {
        require_usable_backend()?;
        workspace_for_mode(m, k, n, ExecutionMode::Pipelined)?;
        let shape = PipelineShape::new(m, k, n)?;
        let api = api()?;
        let pipeline = PipelineApi::load(&api).context("requested CUDA pipeline is unavailable")?;
        let device = DeviceGuard::select(Arc::clone(&api), ordinal)?;
        let mut engine = Self {
            api,
            pipeline,
            device,
            lanes: std::array::from_fn(|_| PipelineLane::default()),
            shape,
            finished: false,
        };
        let initialized = (|| {
            engine.device.reassert()?;
            for lane in &mut engine.lanes {
                lane.initialize(&engine.api, &engine.pipeline, shape)?;
            }
            Ok(())
        })();
        match initialized {
            Ok(()) => Ok(engine),
            Err(error) => with_cleanup(Err(error), engine.finish()),
        }
    }
    fn products(&mut self, inputs: &[(&[u16], &[u16], u32)]) -> Vec<Result<Vec<u16>>> {
        if inputs.len() > PIPELINE_LANES {
            return inputs
                .iter()
                .map(|_| Err(anyhow::anyhow!("CUDA residue batch exceeds lane limit")))
                .collect();
        }
        // Finish the bounded host preparation before enqueueing either lane.
        // Otherwise CPU packing of the next residue can outlast the previous
        // lane's complete device work and prevent cross-stream overlap. Keep
        // each packing result separate so a rejected operand does not stop its
        // independent neighbor from completing.
        let packed: Vec<_> = inputs
            .iter()
            .zip(&mut self.lanes)
            .map(|(&(a, b, prime), lane)| lane.pack(a, b, prime, self.shape))
            .collect();
        // Queue both H2D/GEMM phases before either D2H. On devices/drivers
        // with a shared copy queue, a D2H waiting for its GEMM must not block
        // the next independent lane's H2D in that queue.
        let submitted: Vec<_> = packed
            .into_iter()
            .zip(&mut self.lanes)
            .map(|(packed, lane)| {
                packed?;
                self.device.reassert()?;
                lane.submit(&self.pipeline, self.shape)
            })
            .collect();
        let submitted: Vec<_> = submitted
            .into_iter()
            .zip(&mut self.lanes)
            .map(|(submitted, lane)| {
                submitted?;
                self.device.reassert()?;
                lane.submit_result(&self.pipeline, self.shape)
            })
            .collect();
        submitted
            .into_iter()
            .zip(inputs)
            .zip(&mut self.lanes)
            .map(|((submitted, &(_, _, prime)), lane)| {
                let synchronized = self
                    .device
                    .reassert()
                    .and_then(|()| lane.synchronize(&self.pipeline));
                if synchronized.is_err() {
                    CUDA_QUARANTINED.store(true, Ordering::Release);
                }
                with_cleanup(submitted, synchronized)?;
                Ok(lane.result(self.shape, prime))
            })
            .collect()
    }
    fn release_resources(&mut self) -> Result<()> {
        if let Err(error) = self.device.reassert() {
            CUDA_QUARANTINED.store(true, Ordering::Release);
            return Err(error);
        }
        let mut released = Ok(());
        for lane in &mut self.lanes {
            released = with_cleanup(released, lane.release(&self.api, &self.pipeline));
        }
        released
    }
    fn finish(mut self) -> Result<()> {
        self.finished = true;
        let released = self.release_resources();
        with_cleanup(released, self.device.restore())
    }
}
impl Drop for PipelineEngine {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.release_resources();
        }
    }
}

enum EngineKind {
    Synchronous(SerialEngine),
    Pipelined(PipelineEngine),
}
pub(super) struct Engine(EngineKind);
impl Engine {
    pub(super) fn new(
        ordinal: usize,
        m: usize,
        k: usize,
        n: usize,
        mode: ExecutionMode,
    ) -> Result<Self> {
        match mode {
            ExecutionMode::Synchronous => SerialEngine::new(ordinal, m, k, n)
                .map(|engine| Self(EngineKind::Synchronous(engine))),
            ExecutionMode::Pipelined => PipelineEngine::new(ordinal, m, k, n)
                .map(|engine| Self(EngineKind::Pipelined(engine))),
        }
    }
    pub(super) fn products(&mut self, inputs: &[(&[u16], &[u16], u32)]) -> Vec<Result<Vec<u16>>> {
        match &mut self.0 {
            EngineKind::Synchronous(engine) => inputs
                .iter()
                .map(|&(a, b, prime)| engine.product(a, b, prime))
                .collect(),
            EngineKind::Pipelined(engine) => engine.products(inputs),
        }
    }
    pub(super) fn product(&mut self, a: &[u16], b: &[u16], prime: u32) -> Result<Vec<u16>> {
        match &mut self.0 {
            EngineKind::Synchronous(engine) => engine.product(a, b, prime),
            EngineKind::Pipelined(engine) => engine
                .products(&[(a, b, prime)])
                .pop()
                .context("CUDA pipeline omitted scalar result")?,
        }
    }
    pub(super) fn self_test(ordinal: usize, mode: ExecutionMode) -> Result<()> {
        if mode == ExecutionMode::Pipelined {
            // Resolve the complete optional ABI before any device computation.
            PipelineApi::load(api()?.as_ref()).context("requested CUDA pipeline is unavailable")?;
        }
        // Preserve the existing scalar GEMM oracle, then independently test
        // the requested batching/stream path before accepting caller work.
        SerialEngine::self_test(ordinal)?;
        if mode == ExecutionMode::Pipelined {
            let mut engine = PipelineEngine::new(ordinal, 2, 3, 2)?;
            let a = [0, 1, 16190, 8000, 16189, 3];
            let b = [16190, 8000, 1, 2, 16189, 3];
            let zeros = [0; 6];
            let tested = (|| {
                let mut products = engine
                    .products(&[(&a, &b, 16191), (&zeros, &zeros, 3)])
                    .into_iter();
                ensure!(
                    products
                        .next()
                        .context("pipeline self-test omitted first residue")??
                        == super::cpu_residue_product(&a, &b, 2, 3, 2, 16191),
                    "CUDA device {ordinal} failed exact pipeline self-test"
                );
                ensure!(
                    products
                        .next()
                        .context("pipeline self-test omitted second residue")??
                        == vec![0; 4],
                    "CUDA device {ordinal} failed zero pipeline self-test"
                );
                Ok(())
            })();
            with_cleanup(tested, engine.finish())?;
        }
        Ok(())
    }
    pub(super) fn finish(self) -> Result<()> {
        match self.0 {
            EngineKind::Synchronous(engine) => engine.finish(),
            EngineKind::Pipelined(engine) => engine.finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    fn pipeline_configuration_is_explicit_and_strict() {
        assert_eq!(
            ExecutionMode::parse(None).unwrap(),
            ExecutionMode::Synchronous
        );
        assert_eq!(ExecutionMode::parse(Some("0")).unwrap().batch_size(), 1);
        assert_eq!(ExecutionMode::parse(Some("1")).unwrap().batch_size(), 2);
        for text in ["", "true", "2", " 1", "1 "] {
            assert!(ExecutionMode::parse(Some(text)).is_err());
        }
    }

    #[test]
    fn default_pipeline_requires_128_moduli_per_admitted_device_and_honors_overrides() {
        for devices in [1, 2, 4] {
            assert!(!default_pipeline_work(128 * devices - 1, devices));
            for per_device in [63, 64, 65, 127, 128, 129] {
                assert_eq!(
                    default_pipeline_work(per_device * devices, devices),
                    per_device >= 128
                );
            }
        }
        assert!(!default_pipeline_work(usize::MAX, 0));
        for moduli in [0, 1, 63, 64, 65, 127, 128, 129, usize::MAX] {
            assert_eq!(
                ExecutionMode::product_request(Some("0"), moduli).unwrap(),
                (ExecutionMode::Synchronous, false)
            );
            assert_eq!(
                ExecutionMode::product_request(Some("1"), moduli).unwrap(),
                (ExecutionMode::Pipelined, false)
            );
            assert_eq!(
                ExecutionMode::product_request(None, moduli).unwrap(),
                if moduli >= 128 {
                    (ExecutionMode::Pipelined, true)
                } else {
                    (ExecutionMode::Synchronous, false)
                }
            );
        }
    }

    #[test]
    fn default_pipeline_preflight_preserves_device_order_and_falls_back_only_before_work() {
        let device = |ordinal| CudaDeviceInfo {
            ordinal,
            pci_bus_id: format!("device-{ordinal}"),
            compute_major: 12,
            compute_minor: 0,
            free_memory_bytes: 1 << 30,
            total_memory_bytes: 2 << 30,
        };
        let baseline = vec![device(0), device(1)];
        let no_selection = || -> Result<(Vec<CudaDeviceInfo>, u64)> {
            panic!("must not allocate/select pipeline")
        };
        assert!(default_pipeline_devices(
            255,
            &baseline,
            || panic!("below per-device threshold"),
            no_selection
        )
        .unwrap()
        .is_none());
        assert!(
            default_pipeline_devices(256, &baseline, || Ok(false), no_selection)
                .unwrap()
                .is_none()
        );
        let accepted =
            default_pipeline_devices(256, &baseline, || Ok(true), || Ok((baseline.clone(), 123)))
                .unwrap()
                .unwrap();
        assert_eq!(accepted.1, 123);
        for candidate in [vec![], vec![device(0)], vec![device(1), device(0)]] {
            assert!(
                default_pipeline_devices(256, &baseline, || Ok(true), || Ok((candidate, 123)))
                    .unwrap()
                    .is_none()
            );
        }
        assert!(default_pipeline_devices(
            256,
            &baseline,
            || Ok(true),
            || Err(DeviceHeadroomLimit(1).into())
        )
        .unwrap()
        .is_none());
        assert!(default_pipeline_devices(
            256,
            &baseline,
            || Ok(true),
            || Err(super::super::ModularWorkspaceLimit {
                stage: "CUDA product buffers",
                required: 200,
                available: 100,
            }
            .into())
        )
        .unwrap()
        .is_none());
        let unavailable = default_pipeline_devices(
            256,
            &baseline,
            || bail!("backend quarantined"),
            no_selection,
        )
        .unwrap_err();
        assert_eq!(unavailable.to_string(), "backend quarantined");
        let invalid = default_pipeline_devices(
            256,
            &baseline,
            || Ok(true),
            || bail!("invalid configuration"),
        )
        .unwrap_err();
        assert_eq!(invalid.to_string(), "invalid configuration");
    }

    #[test]
    fn pending_transfer_storage_requires_successful_sync_before_reuse() {
        let mut transfers = TransferState::default();
        transfers
            .synchronize(|| panic!("idle lane must not synchronize"))
            .unwrap();
        transfers.begin().unwrap();
        assert!(transfers.begin().is_err());
        let failed = transfers.synchronize(|| bail!("injected synchronization failure"));
        assert!(failed.is_err());
        assert!(transfers.pending);
        assert!(transfers.begin().is_err());
        transfers.synchronize(|| Ok(())).unwrap();
        assert!(!transfers.pending);
        transfers.begin().unwrap();
        transfers.synchronize(|| Ok(())).unwrap();
    }

    #[test]
    fn pipeline_shape_and_workspace_charge_both_pinned_lanes_before_allocation() {
        let shape = PipelineShape::new(7, 9, 5).unwrap();
        assert_eq!((shape.pm, shape.pk, shape.pn), (8, 32, 8));
        assert_eq!(shape.bytes(), [3 * 8 * 32, 3 * 32 * 8, 12 * 8 * 8]);
        let staging = 3 * (8 * 32 + 32 * 8) + 12 * 8 * 8;
        let gathered_and_answers = 2 * (7 * 9 + 9 * 5 + 7 * 5);
        let (device, aggregate) = workspace_for_mode(7, 9, 5, ExecutionMode::Pipelined).unwrap();
        assert_eq!(device, 2 * (staging + (4 << 20)));
        assert_eq!(
            aggregate,
            device + 2 * (staging + gathered_and_answers) + (1 << 20)
        );
        assert_eq!(
            workspace_for_mode(7, 9, 5, ExecutionMode::Synchronous).unwrap(),
            workspace(7, 9, 5).unwrap()
        );
        let largest_k = (i32::MAX as usize / (127 * 127)) & !31;
        assert!(PipelineShape::new(1, largest_k, 1).is_ok());
        assert!(PipelineShape::new(1, largest_k + 1, 1).is_err());
        for dimensions in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (usize::MAX, 1, 1),
            (1, usize::MAX, 1),
            (1, 1, usize::MAX),
        ] {
            assert!(PipelineShape::new(dimensions.0, dimensions.1, dimensions.2).is_err());
        }
        let status = CudaStatus {
            available: true,
            devices: vec![CudaDeviceInfo {
                ordinal: 4,
                pci_bus_id: "0000:01:00.0".to_owned(),
                compute_major: 12,
                compute_minor: 0,
                free_memory_bytes: 16 << 30,
                total_memory_bytes: 16 << 30,
            }],
            ..Default::default()
        };
        let refusal = choose_devices_with_policy(
            &status,
            device,
            aggregate,
            aggregate - 1,
            2,
            5,
            0,
            false,
            true,
        )
        .unwrap_err();
        assert!(refusal.is::<super::super::ModularWorkspaceLimit>());
        let admitted =
            choose_devices_with_policy(&status, device, aggregate, aggregate, 2, 5, 0, false, true)
                .unwrap();
        assert_eq!(admitted.len(), 1);
    }

    #[test]
    fn reduction_padding_charges_complete_integer_tiles_without_changing_outer_shape() {
        for (m, k, n, pk) in [
            (151, 50, 151, 64),
            (226, 75, 226, 96),
            (301, 100, 301, 128),
            (150, 100, 150, 128),
            (151, 101, 149, 128),
        ] {
            let shape = PipelineShape::new(m, k, n).unwrap();
            assert_eq!(
                (shape.pm, shape.pk, shape.pn),
                (padded(m).unwrap(), pk, padded(n).unwrap())
            );
            let (device, _) = workspace(m, k, n).unwrap();
            assert_eq!(
                device,
                (shape.a + shape.b + 4 * shape.c + BLAS_WORKSPACE_BYTES) as u64
            );
        }
        for k in 1..=257 {
            let pk = padded_reduction(k).unwrap();
            assert!(pk >= k && pk - k < 32 && pk.is_multiple_of(32));
        }
        assert!(padded_reduction(usize::MAX).is_err());
        assert!(padded_reduction(i32::MAX as usize).is_err());
    }

    #[test]
    #[ignore = "requires explicitly selected idle CUDA devices"]
    fn device_integer_products_cover_non_tile_aligned_reductions_in_both_modes() {
        let status = status().unwrap();
        assert!(
            status.available,
            "{}",
            status.unavailable_reason.unwrap_or_default()
        );
        for device in &status.devices {
            for (m, k, n) in [
                (67, 15, 69),
                (67, 16, 69),
                (67, 17, 69),
                (65, 31, 67),
                (65, 32, 67),
                (65, 33, 67),
                (151, 50, 151),
                (226, 75, 226),
                (301, 100, 301),
                (150, 100, 150),
                (151, 101, 149),
                (150, 128, 150),
                (188, 63, 188),
            ] {
                let prime = 16187;
                let a: Vec<_> = (0..m * k)
                    .map(|i| ((i * 7919 + 1) % prime) as u16)
                    .collect();
                let b: Vec<_> = (0..k * n)
                    .map(|i| ((i * 3571 + 3) % prime) as u16)
                    .collect();
                let expected = super::super::cpu_residue_product(&a, &b, m, k, n, prime as u32);
                for mode in [ExecutionMode::Synchronous, ExecutionMode::Pipelined] {
                    let mut engine = Engine::new(device.ordinal, m, k, n, mode).unwrap();
                    let inputs = [
                        (&a[..], &b[..], prime as u32),
                        (&a[..], &b[..], prime as u32),
                    ];
                    for count in [2, 1, 2] {
                        for result in engine.products(&inputs[..count]) {
                            assert_eq!(
                                result.unwrap(),
                                expected,
                                "shape {m}x{k}x{n}, mode {mode:?}"
                            );
                        }
                    }
                    engine.finish().unwrap();
                }
            }
        }
    }

    #[test]
    fn batched_digit_layout_matches_independent_integer_products_and_reuses_padding() {
        for (m, k, n) in [(1, 1, 1), (3, 5, 2), (9, 7, 3), (129, 3, 5)] {
            let shape = PipelineShape::new(m, k, n).unwrap();
            let mut aa = vec![99; 3 * shape.a];
            let mut bb = vec![-99; 3 * shape.b];
            for prime in [16191, 3, 127, 16187] {
                let values = [0, 1, prime / 2, prime / 2 + 1, prime - 1];
                let a: Vec<_> = (0..m * k)
                    .map(|i| values[i % values.len()] as u16)
                    .collect();
                let b: Vec<_> = (0..k * n)
                    .map(|i| values[(3 * i + 1) % values.len()] as u16)
                    .collect();
                pack_digit_batch(&a, &b, prime, shape, &mut aa, &mut bb).unwrap();
                for (operands, stride) in [(&aa, shape.a), (&bb, shape.b)] {
                    for i in 0..stride {
                        assert_eq!(
                            operands[2 * stride + i],
                            operands[i].checked_add(operands[stride + i]).unwrap()
                        );
                    }
                }
                let products: Vec<_> = (0..3)
                    .map(|digit| {
                        host_digit_product(
                            &aa[digit * shape.a..(digit + 1) * shape.a],
                            &bb[digit * shape.b..(digit + 1) * shape.b],
                            shape.pm,
                            shape.pk,
                            shape.pn,
                        )
                    })
                    .collect();
                let mut actual = vec![0; m * n];
                recombine_products(
                    &products[0],
                    &products[1],
                    &products[2],
                    shape.pm,
                    n,
                    prime,
                    &mut actual,
                );
                assert_eq!(
                    actual,
                    super::super::cpu_residue_product(&a, &b, m, k, n, prime)
                );
                for i in 0..shape.pm {
                    for t in 0..shape.pk {
                        if i >= m || t >= k {
                            assert!((0..3).all(|digit| aa[digit * shape.a + i * shape.pk + t] == 0));
                        }
                    }
                }
                for j in 0..shape.pn {
                    for t in 0..shape.pk {
                        if j >= n || t >= k {
                            assert!((0..3).all(|digit| bb[digit * shape.b + j * shape.pk + t] == 0));
                        }
                    }
                }
            }
            let a = vec![0; m * k];
            let b = vec![0; k * n];
            assert!(pack_digit_batch(&a[..a.len() - 1], &b, 3, shape, &mut aa, &mut bb).is_err());
            assert!(pack_digit_batch(&a, &b, 2, shape, &mut aa, &mut bb).is_err());
            assert!(pack_digit_batch(&a, &b, 3, shape, &mut aa[1..], &mut bb).is_err());
        }
    }

    #[test]
    #[ignore = "requires explicitly selected idle CUDA devices; experimental pipeline qualification"]
    fn pipeline_device_matches_cpu_across_tail_batches_reuse_and_rejected_operands() {
        let status = status().unwrap();
        assert!(
            status.available,
            "{}",
            status.unavailable_reason.unwrap_or_default()
        );
        for device in status.devices {
            Engine::self_test(device.ordinal, ExecutionMode::Pipelined).unwrap();
            for (m, k, n) in [(1, 30, 1), (7, 9, 5), (129, 129, 131)] {
                let mut engine =
                    Engine::new(device.ordinal, m, k, n, ExecutionMode::Pipelined).unwrap();
                for primes in [[16191, 3], [127, 16187], [3, 16191]] {
                    let operands: Vec<_> = primes
                        .into_iter()
                        .map(|prime| {
                            let a: Vec<_> = (0..m * k)
                                .map(|i| ((7919 * i + 13) % prime as usize) as u16)
                                .collect();
                            let b: Vec<_> = (0..k * n)
                                .map(|i| ((8191 * i + 37) % prime as usize) as u16)
                                .collect();
                            (a, b, prime)
                        })
                        .collect();
                    for count in [2, 1, 2] {
                        let inputs: Vec<_> = operands[..count]
                            .iter()
                            .map(|(a, b, prime)| (a.as_slice(), b.as_slice(), *prime))
                            .collect();
                        let actual = engine.products(&inputs);
                        for ((a, b, prime), actual) in inputs.iter().zip(actual) {
                            assert_eq!(
                                actual.unwrap(),
                                super::super::cpu_residue_product(a, b, m, k, n, *prime)
                            );
                        }
                    }
                    let mut bad = operands[0].0.clone();
                    bad[0] = 16191;
                    let results = engine.products(&[
                        (&bad, &operands[0].1, 16191),
                        (&operands[1].0, &operands[1].1, operands[1].2),
                    ]);
                    let mut results = results.into_iter();
                    assert!(results.next().unwrap().is_err());
                    assert!(results.next().unwrap().is_ok());
                }
                engine.finish().unwrap();
            }
        }
    }

    #[test]
    fn injected_device_selection_restores_nested_owners_and_checks_failures() {
        let current = Cell::new(7);
        let fail_next = Cell::new(None);
        let calls = RefCell::new(Vec::new());
        let get = |device: &mut c_int| {
            *device = current.get();
            0
        };
        let set = |device| {
            calls.borrow_mut().push(device);
            // Model even an unsuccessful selection changing thread state.
            current.set(device);
            if fail_next.get() == Some(device) {
                fail_next.set(None);
                99
            } else {
                0
            }
        };
        let mut outer = DeviceSelection::select(0, get, set).unwrap();
        let mut inner = DeviceSelection::select(1, get, set).unwrap();
        assert_eq!(current.get(), 1);
        inner.restore(set).unwrap();
        assert_eq!(current.get(), 0);
        current.set(1);
        fail_next.set(Some(0));
        assert!(outer.reassert(set).is_err());
        outer.reassert(set).unwrap();
        assert_eq!(current.get(), 0);
        fail_next.set(Some(7));
        assert!(outer.restore(set).is_err());
        assert_eq!(outer.previous, Some(7));
        outer.restore(set).unwrap();
        assert_eq!(current.get(), 7);
        assert_eq!(outer.previous, None);
        let restored_calls = calls.borrow().len();
        outer.restore(set).unwrap();
        assert_eq!(calls.borrow().len(), restored_calls);

        fail_next.set(Some(1));
        let failed = DeviceSelection::select(1, get, set).err().unwrap();
        assert!(failed.to_string().contains("select device"));
        assert_eq!(current.get(), 7);
        fail_next.set(Some(0));
        assert!(DeviceSelection::select(0, get, set).is_err());
        assert_eq!(current.get(), 7);
    }

    #[test]
    fn checked_cleanup_rejects_success_and_preserves_original_failure() {
        let failed_cleanup = || -> Result<()> { bail!("restore failed") };
        assert!(with_cleanup(Ok(123), failed_cleanup()).is_err());
        let error = with_cleanup::<()>(Err(anyhow::anyhow!("product failed")), failed_cleanup())
            .unwrap_err();
        assert_eq!(error.root_cause().to_string(), "product failed");
        assert!(format!("{error:#}").contains("restore failed"));
    }

    #[test]
    fn pci_bus_id_requires_bounded_nonempty_utf8_c_string() {
        assert_eq!(
            pci_bus_id_text(b"0000:01:00.0\0unused").unwrap(),
            "0000:01:00.0"
        );
        assert!(pci_bus_id_text(b"0000:01:00.0").is_err());
        assert!(pci_bus_id_text(b"\0").is_err());
        assert!(pci_bus_id_text(&[0xff, 0]).is_err());
    }

    #[test]
    fn parallel_digit_packing_preserves_centered_values_and_reused_padding() {
        let (m, k, n) = (129, 129, 131);
        let (pm, pk, pn) = (
            padded(m).unwrap(),
            padded_reduction(k).unwrap(),
            padded(n).unwrap(),
        );
        let mut a0 = vec![i8::MAX; pm * pk];
        let mut a1 = vec![i8::MIN; pm * pk];
        let mut b0 = vec![i8::MAX; pk * pn];
        let mut b1 = vec![i8::MIN; pk * pn];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        // The first case covers every canonical residue in the largest
        // encoding domain. Later cases reuse the same buffers and padding.
        for prime in [16191, 3, 127] {
            let a: Vec<_> = (0..m * k).map(|i| (i as u32 % prime) as u16).collect();
            let b: Vec<_> = (0..k * n)
                .map(|i| ((i as u32 * 7919 + 1) % prime) as u16)
                .collect();
            pool.install(|| {
                let (left, right) = rayon::join(
                    || pack_left(&a, k, pk, prime, &mut a0, &mut a1),
                    || pack_right(&b, n, pk, prime, &mut b0, &mut b1),
                );
                left.unwrap();
                right.unwrap();
            });
            for i in 0..pm {
                for t in 0..pk {
                    let index = i * pk + t;
                    if i < m && t < k {
                        assert!((-64..=63).contains(&a0[index]));
                        assert!((-63..=63).contains(&a1[index]));
                        assert!(a0[index].checked_add(a1[index]).is_some());
                        assert_eq!(
                            i64::from(a0[index]) + 128 * i64::from(a1[index]),
                            centered(a[i * k + t], prime)
                        );
                    } else {
                        assert_eq!((a0[index], a1[index]), (0, 0));
                    }
                }
            }
            for j in 0..pn {
                for t in 0..pk {
                    let index = j * pk + t;
                    if j < n && t < k {
                        assert!((-64..=63).contains(&b0[index]));
                        assert!((-63..=63).contains(&b1[index]));
                        assert!(b0[index].checked_add(b1[index]).is_some());
                        assert_eq!(
                            i64::from(b0[index]) + 128 * i64::from(b1[index]),
                            centered(b[t * n + j], prime)
                        );
                    } else {
                        assert_eq!((b0[index], b1[index]), (0, 0));
                    }
                }
            }
        }
    }

    #[test]
    fn digit_packing_rejects_noncanonical_operands() {
        let mut low = vec![0; 4 * 4];
        let mut high = low.clone();
        assert_eq!(
            pack_left(&[0, 3], 2, 4, 3, &mut low, &mut high)
                .unwrap_err()
                .to_string(),
            "noncanonical left residue"
        );
        assert_eq!(
            pack_right(&[0, 3], 2, 4, 3, &mut low, &mut high)
                .unwrap_err()
                .to_string(),
            "noncanonical right residue"
        );
    }

    fn host_digit_product(a: &[i8], b: &[i8], pm: usize, pk: usize, pn: usize) -> Vec<i32> {
        let mut result = vec![0; pm * pn];
        for j in 0..pn {
            for i in 0..pm {
                result[j * pm + i] = (0..pk)
                    .map(|t| i32::from(a[i * pk + t]) * i32::from(b[j * pk + t]))
                    .sum();
            }
        }
        result
    }

    #[test]
    fn packed_three_product_recombination_matches_exact_residue_dots() {
        for (m, k, n) in [
            (3, 5, 2),
            (7, 9, 5),
            (1, 4, 9),
            (150, 100, 150),
            (151, 101, 149),
        ] {
            let (pm, pk, pn) = (
                padded(m).unwrap(),
                padded_reduction(k).unwrap(),
                padded(n).unwrap(),
            );
            for prime in [3, 127, 16191] {
                let values = [0, 1, prime / 2, prime / 2 + 1, prime - 1];
                let a: Vec<_> = (0..m * k)
                    .map(|i| values[i % values.len()] as u16)
                    .collect();
                let b: Vec<_> = (0..k * n)
                    .map(|i| values[(i * 3 + 1) % values.len()] as u16)
                    .collect();
                let mut a0 = vec![99; pm * pk];
                let mut a1 = a0.clone();
                let mut b0 = vec![99; pk * pn];
                let mut b1 = b0.clone();
                pack_left(&a, k, pk, prime, &mut a0, &mut a1).unwrap();
                pack_right(&b, n, pk, prime, &mut b0, &mut b1).unwrap();
                let c0 = host_digit_product(&a0, &b0, pm, pk, pn);
                let c2 = host_digit_product(&a1, &b1, pm, pk, pn);
                for (low, high) in a0.iter_mut().zip(a1) {
                    *low = low.checked_add(high).unwrap();
                }
                for (low, high) in b0.iter_mut().zip(b1) {
                    *low = low.checked_add(high).unwrap();
                }
                let cs = host_digit_product(&a0, &b0, pm, pk, pn);
                let mut actual = vec![0; m * n];
                recombine_products(&c0, &c2, &cs, pm, n, prime, &mut actual);
                assert_eq!(
                    actual,
                    super::super::cpu_residue_product(&a, &b, m, k, n, prime)
                );
            }
        }
    }

    #[test]
    fn parallel_recombination_keeps_row_order_and_signed_integer_exactness() {
        let (m, n) = (129, 131);
        let (pm, pn) = (padded(m).unwrap(), padded(n).unwrap());
        let values = [i32::MIN, i32::MAX, -1, 0, 1];
        let c0: Vec<_> = (0..pm * pn).map(|i| values[i % values.len()]).collect();
        let c2: Vec<_> = (0..pm * pn)
            .map(|i| values[(i + 1) % values.len()])
            .collect();
        let cs: Vec<_> = (0..pm * pn)
            .map(|i| values[(i + 3) % values.len()])
            .collect();
        let mut actual = vec![0; m * n];
        let prime = 16191;
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| recombine_products(&c0, &c2, &cs, pm, n, prime, &mut actual));
        for i in 0..m {
            for j in 0..n {
                let index = j * pm + i;
                let low = i128::from(c0[index]);
                let high = i128::from(c2[index]);
                let sum = i128::from(cs[index]);
                let expected = (low + 128 * (sum - low - high) + 16384 * high)
                    .rem_euclid(i128::from(prime)) as u16;
                assert_eq!(actual[i * n + j], expected);
            }
        }
    }

    #[test]
    fn device_selection_uses_visible_ordinals_and_refuses_duplicates() {
        assert_eq!(ordinals(None, 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(ordinals(Some("2, 0"), 3).unwrap(), vec![2, 0]);
        for text in ["", "0,0", "3", "-1", "all", "1,"] {
            assert!(ordinals(Some(text), 3).is_err());
        }
    }
    #[test]
    fn workspace_accounts_for_device_and_host_buffers_and_overflow() {
        let (device, aggregate) = workspace(7, 9, 5).unwrap();
        assert_eq!(device, 8 * 32 + 32 * 8 + 4 * 8 * 8 + (4 << 20));
        assert!(aggregate > device + (1 << 20));
        assert!(workspace(usize::MAX, 4, 4).is_err());
    }
    #[test]
    fn device_planning_enforces_aggregate_budget_and_free_headroom() {
        let gib = 1 << 30;
        let status = CudaStatus {
            available: true,
            devices: vec![
                CudaDeviceInfo {
                    ordinal: 0,
                    pci_bus_id: "0000:01:00.0".to_owned(),
                    compute_major: 12,
                    compute_minor: 0,
                    free_memory_bytes: 24 * gib,
                    total_memory_bytes: 32 * gib,
                },
                CudaDeviceInfo {
                    ordinal: 1,
                    pci_bus_id: "0000:02:00.0".to_owned(),
                    compute_major: 10,
                    compute_minor: 0,
                    free_memory_bytes: 100 * gib,
                    total_memory_bytes: 128 * gib,
                },
            ],
            ..Default::default()
        };
        let plan = |budget, device, workers, explicit| {
            choose_devices_with_policy(
                &status,
                device,
                5 * gib,
                budget,
                workers,
                100,
                gib / 2,
                explicit,
                false,
            )
        };
        assert_eq!(plan(10 * gib, 4 * gib, 8, false).unwrap().len(), 2);
        assert_eq!(plan(9 * gib, 4 * gib, 8, false).unwrap().len(), 1);
        assert!(plan(4 * gib, 4 * gib, 8, false).unwrap().is_empty());
        assert_eq!(plan(10 * gib, 4 * gib, 1, false).unwrap().len(), 1);
        assert_eq!(plan(10 * gib, 22 * gib, 8, false).unwrap()[0].ordinal, 1);
        assert!(plan(10 * gib, 22 * gib, 8, true).is_err());
        let forced = choose_devices_with_policy(
            &status,
            4 * gib,
            5 * gib,
            4 * gib,
            8,
            100,
            gib / 2,
            false,
            true,
        )
        .unwrap_err();
        let refusal = forced
            .downcast_ref::<super::super::ModularWorkspaceLimit>()
            .unwrap();
        assert_eq!(
            (refusal.required, refusal.available),
            (5 * gib as u128, 4 * gib as u128)
        );
        // Even with an insufficient aggregate budget, invalid explicit free
        // memory takes precedence and cannot be converted into a split retry.
        let headroom = choose_devices_with_policy(
            &status,
            22 * gib,
            5 * gib,
            4 * gib,
            8,
            100,
            gib / 2,
            true,
            true,
        )
        .unwrap_err();
        assert!(!headroom.is::<super::super::ModularWorkspaceLimit>());
        assert!(headroom
            .to_string()
            .contains("lacks free memory after headroom"));
        let no_free_device = choose_devices_with_policy(
            &status,
            120 * gib,
            5 * gib,
            4 * gib,
            8,
            100,
            gib / 2,
            false,
            true,
        )
        .unwrap();
        assert!(no_free_device.is_empty());
        assert!(choose_devices_with_policy(
            &CudaStatus::default(),
            4 * gib,
            5 * gib,
            4 * gib,
            8,
            100,
            gib / 2,
            false,
            true,
        )
        .unwrap()
        .is_empty());
    }
}
