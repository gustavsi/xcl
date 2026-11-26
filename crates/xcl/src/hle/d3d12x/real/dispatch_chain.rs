//! PSOs for a split compute entry. The first piece is the real PSO returned
//! to the caller. Other pieces and the lazily built whole form belong to it.
use super::*;
use crate::gpu::{dxil::kernel::DispatchChain, recomp::pso};
use std::sync::{Arc, LazyLock, OnceLock};

static PROGRAMS: AtomicU64 = AtomicU64::new(0);
static PIECES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// slabs dispatched (`pso::dispatch_slabs`)
static SLABS: AtomicU64 = AtomicU64::new(0);
/// `.plib` hits and misses for split compute pipelines (pieces and whole forms)
static PLIB_HITS: AtomicU64 = AtomicU64::new(0);
static PLIB_MISSES: AtomicU64 = AtomicU64::new(0);
/// scratch alive now, the most there was, and the most a dispatch would have needed at once (bytes)
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static WANTED: AtomicU64 = AtomicU64::new(0);
/// ExecuteIndirects run as chains, those of them that ran the whole form instead (more slabs than
/// rounds), and the slabs lost when there was no whole form yet
static INDIRECT: AtomicU64 = AtomicU64::new(0);
static INDIRECT_WHOLE: AtomicU64 = AtomicU64::new(0);
static INDIRECT_LOST: AtomicU64 = AtomicU64::new(0);
/// ExecuteIndirects that ran the whole form (not chainable)
static WHOLE_INDIRECT: AtomicU64 = AtomicU64::new(0);
/// fast check so non-split indirect dispatches avoid locking `CHAINS`
static HAS_CHAINS: AtomicBool = AtomicBool::new(false);
/// raw PSO address of the slicer already labeled for DRED
static SLICER_LABELED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn log_counts() {
    let mib = |a: &AtomicU64| a.load(Ordering::Relaxed) >> 20;
    let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
    crate::xlog_info!("compute split: {} programs split, {} pieces built, {} chained dispatches issued in {} slabs; plib {} hits / {} misses; scratch {} MiB (most {}, largest dispatch {} in one slab); ExecuteIndirect: {} chained ({} of them whole, {} slabs lost), {} whole",
        n(&PROGRAMS), n(&PIECES), n(&DISPATCHES), n(&SLABS), n(&PLIB_HITS), n(&PLIB_MISSES), mib(&LIVE), mib(&PEAK), mib(&WANTED),
        n(&INDIRECT), n(&INDIRECT_WHOLE), n(&INDIRECT_LOST), n(&WHOLE_INDIRECT));
}

/// The scratch a dispatch uses at once, `XCL_SPLIT_SCRATCH_MB` (256): more groups than fit run in
/// slabs. A retail title's split programs need 190 KB for each group (median), 3 GB for 960x540
/// pixels in groups of 8x4.
fn scratch_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(256u64).min(4096) << 20)
}

/// Thread name for the dedicated background whole-form compiler. The `xcl-pso` prefix ensures
/// `driver_compile_identified` routes its driver compiles through `compile_slots` (`BIG_COMPILES`
/// / `MEDIUM_COMPILES` / `GIANT_COMPILES`) and `big_compile_commit_free` commit-headroom gating.
const WHOLE_WORKER_THREAD_NAME: &str = "xcl-pso-whole";

/// Deterministic cross-run `.plib` key for a split compute pipeline (piece or whole form).
/// Prefixed with `s` (`sc-...`) so 8-parameter split-root-signature PSOs never collide with
/// 6-parameter standard compute PSOs (`c-...`) in the shared pipeline library.
fn split_plib_key(dxil: &[u8], samplers: &[pso::StaticSampler], flags: i32) -> String {
    format!("s{}", pso_lib::compute_key(dxil, samplers, flags))
}

struct PipelineBuilder {
    device: ID3D12Device,
    create: PsoFn,
    signature: ID3D12RootSignature,
    samplers: Vec<pso::StaticSampler>,
    flags: D3D12_PIPELINE_STATE_FLAGS,
    node: u32,
    record: u64,
}

impl PipelineBuilder {
    unsafe fn pipeline(&self, dxil: &[u8], iid: &GUID) -> Result<ID3D12PipelineState, HRESULT> {
        let lib_key = split_plib_key(dxil, &self.samplers, self.flags.0);
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { NodeMask: self.node, Flags: self.flags, ..Default::default() };
        let lib_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(self.signature.as_raw()),
            CS: bytecode(dxil),
            NodeMask: self.node,
            Flags: self.flags,
            ..Default::default()
        };
        let mut raw = std::ptr::null_mut();
        if pso_lib::try_load_compute(self.device.as_raw(), &lib_key, &lib_desc, iid, &mut raw) && !raw.is_null() {
            let hits = PLIB_HITS.fetch_add(1, Ordering::Relaxed) + 1;
            if hits <= 4 {
                crate::xlog_info!("compute split: .plib hit #{hits} for cs {:016x} ({lib_key})", self.record);
            }
            return Ok(ID3D12PipelineState::from_raw(raw));
        }
        let misses = PLIB_MISSES.fetch_add(1, Ordering::Relaxed) + 1;
        if misses <= 4 {
            crate::xlog_info!("compute split: .plib miss #{misses} for cs {:016x} ({lib_key})", self.record);
        }
        let (hr, _, _) = compute_container(self.device.as_raw(), self.create, &desc, self.signature.as_raw(),
            dxil, self.record, iid, &mut raw, true);
        if hr >= 0 && !raw.is_null() {
            pso_lib::store_pipeline(self.device.as_raw(), &lib_key, raw);
            Ok(ID3D12PipelineState::from_raw(raw))
        } else {
            Err(hr)
        }
    }
}

struct Chain {
    builder: Arc<PipelineBuilder>,
    tail: Vec<ID3D12PipelineState>,
    words: u32,
    whole_code: Arc<Vec<u8>>,
    whole: OnceLock<Option<ID3D12PipelineState>>,
    whole_started: AtomicBool,
    /// after a whole build that ran out of memory: when it may be tried, once more
    retry_at: Mutex<Option<std::time::Instant>>,
    /// its command signature for ExecuteIndirect (`pso::chain_command_signature`)
    signature: OnceLock<ID3D12CommandSignature>,
    /// the most slabs one of its ExecuteIndirects had
    slabs: std::sync::atomic::AtomicU32,
    /// a lost slab was reported once for this program
    warned: AtomicBool,
}

/// How long a split program's whole form waits after a build that ran out of memory, before its last try.
const WHOLE_RETRY: std::time::Duration = std::time::Duration::from_secs(10);

pub(in crate::hle::d3d12x) enum WholeStatus {
    /// Not a split compute pipeline.
    NotSplit,
    /// Whole pipeline is built and ready to bind without blocking.
    Ready(ID3D12PipelineState),
    /// Whole pipeline is compiling on a background thread; skip this indirect dispatch to avoid
    /// stalling the title thread past its watchdog.
    Pending,
    /// Background compilation of the whole pipeline failed.
    Failed,
}

impl Chain {
    unsafe fn whole(&self) -> Result<ID3D12PipelineState, String> {
        if let Some(p) = self.whole.get() {
            return p.clone().ok_or_else(|| "whole compute pipeline".into());
        }
        let mut retry_at = self.retry_at.lock().unwrap();
        if let Some(p) = self.whole.get() {
            return p.clone().ok_or_else(|| "whole compute pipeline".into());
        }
        if retry_at.is_some_and(|t| std::time::Instant::now() < t) {
            return Err("whole compute pipeline: out of memory".into());
        }
        match self.builder.pipeline(&self.whole_code, &pc_iid(Kind::PipelineState)) {
            Ok(p) => {
                super::super::dred::label_text(p.as_raw() as usize, || format!("cs {:016x} whole", self.builder.record));
                let _ = self.whole.set(Some(p.clone()));
                Ok(p)
            }
            Err(hr) if hr == crate::util::E_OUTOFMEMORY && retry_at.is_none() => {
                *retry_at = Some(std::time::Instant::now() + WHOLE_RETRY);
                Err("whole compute pipeline: out of memory".into())
            }
            Err(_) => {
                let _ = self.whole.set(None);
                Err("whole compute pipeline".into())
            }
        }
    }

    /// After a whole build that ran out of memory, before its retry. Never waits: `whole` holds
    /// `retry_at` for the whole driver compile (tens of seconds for the biggest programs), and a
    /// title thread asking here (an allocator reset dropping a recording, a dispatch) stood that
    /// long – a retail title's own GPU watchdog then ended it (`mov [0], 0x4d8bd421`). A build
    /// going on is no cooldown: `start_whole_async` sees it by `whole_started` or starts the
    /// async one, which waits for it on a worker.
    fn whole_in_cooldown(&self) -> bool {
        let in_cooldown = |at: &Option<std::time::Instant>| at.is_some_and(|t| std::time::Instant::now() < t);
        match self.retry_at.try_lock() {
            Ok(g) => in_cooldown(&g),
            Err(std::sync::TryLockError::WouldBlock) => false,
            Err(std::sync::TryLockError::Poisoned(p)) => in_cooldown(&p.into_inner()),
        }
    }

    fn start_whole_async(self: &Arc<Self>) {
        if self.whole.get().is_some() || self.whole_in_cooldown() {
            return;
        }
        if self.whole_started.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return;
        }
        if !enqueue_whole_async(Arc::clone(self)) {
            self.whole_started.store(false, Ordering::Release);
        }
    }
}

/// Queue a background whole-form build onto the single `xcl-pso-whole` worker so multiple split
/// programs never compile huge whole forms concurrently and every build goes through
/// `driver_compile_identified`'s `BIG_COMPILES` + commit-headroom gate.
fn enqueue_whole_async(chain: Arc<Chain>) -> bool {
    static QUEUE: LazyLock<(Mutex<std::collections::VecDeque<Arc<Chain>>>, std::sync::Condvar)> =
        LazyLock::new(|| (Mutex::new(std::collections::VecDeque::new()), std::sync::Condvar::new()));
    static WORKER_STARTED: AtomicBool = AtomicBool::new(false);
    if WORKER_STARTED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        if std::thread::Builder::new()
            .name(WHOLE_WORKER_THREAD_NAME.into())
            .spawn(|| {
                let (lock, cv) = &*QUEUE;
                loop {
                    let chain = {
                        let mut q = lock.lock().unwrap();
                        while q.is_empty() {
                            q = cv.wait(q).unwrap();
                        }
                        q.pop_front().unwrap()
                    };
                    let record = chain.builder.record;
                    if let Err(e) = unsafe { chain.whole() } {
                        if chain.whole.get().is_none() {
                            // Transient E_OUTOFMEMORY: allow one retry after WHOLE_RETRY.
                            chain.whole_started.store(false, Ordering::Release);
                        }
                        crate::xlog_warn!("D3D12.X: background whole compute PSO for cs {record:016x} failed: {e}");
                    }
                }
            })
            .is_err()
        {
            WORKER_STARTED.store(false, Ordering::Release);
            return false;
        }
    }
    let (lock, cv) = &*QUEUE;
    lock.lock().unwrap().push_back(chain);
    cv.notify_one();
    true
}

static CHAINS: LazyLock<Mutex<HashMap<usize, Arc<Chain>>>> = LazyLock::new(Default::default);

/// Canonical COM object identity (`IUnknown*`) for a D3D12 device pointer so tear-off interfaces
/// share cache slots while synthetic/unreadable test pointers preserve their raw address safely.
unsafe fn canonical_device_key(this: *mut c_void) -> usize {
    if this.is_null() || !crate::gpu::shader::readable(this as usize, std::mem::size_of::<usize>()) {
        return this as usize;
    }
    let v = *(this as *const usize);
    if !crate::gpu::shader::readable(v, std::mem::size_of::<usize>()) || *(v as *const usize) == 0 {
        return this as usize;
    }
    if is_ext(v) && orig_fn(this, 0) == 0 {
        return this as usize;
    }
    ID3D12Device::from_raw_borrowed(&this)
        .and_then(|d| d.cast::<windows::core::IUnknown>().ok())
        .map_or(this as usize, |u| u.as_raw() as usize)
}

/// Dispatch-chain root signatures shared per `(canonical_device_key, static_samplers)`.
/// Intentionally retained for the process lifetime (bounded to one `ID3D12RootSignature` per
/// distinct `(device, static_sampler_set)`) so split programs sharing pieces hit `COMPUTE_BLOBS`
/// instead of allocating a fresh root signature object per program.
static SIGNATURES: LazyLock<Mutex<HashMap<(usize, Vec<u32>), ID3D12RootSignature>>> = LazyLock::new(Default::default);

fn sampler_key(samplers: &[pso::StaticSampler]) -> Vec<u32> {
    samplers.iter().flat_map(|(w, c)| w.iter().copied().chain([*c as u32])).collect()
}

unsafe fn root_signature_device(rs: &ID3D12RootSignature) -> Option<ID3D12Device> {
    let mut owner: Option<ID3D12Device> = None;
    rs.GetDevice(&mut owner).ok()?;
    owner
}

unsafe fn rs_belongs_to_device(rs: &ID3D12RootSignature, dev_key: usize) -> bool {
    root_signature_device(rs).is_some_and(|owner| canonical_device_key(owner.as_raw()) == dev_key)
}

/// Shared dispatch-chain root signature for `dev` and `samplers`, isolating distinct D3D12 devices
/// so `COMPUTE_BLOBS` never reuses a root signature across devices or sampler layouts.
unsafe fn signature_for(dev: &ID3D12Device, samplers: &[pso::StaticSampler]) -> Option<ID3D12RootSignature> {
    let dev_key = canonical_device_key(dev.as_raw());
    let key = (dev_key, sampler_key(samplers));
    if let Some(rs) = SIGNATURES.lock().unwrap().get(&key).cloned() {
        if rs_belongs_to_device(&rs, dev_key) {
            return Some(rs);
        }
    }
    let created = match pso::dispatch_chain_root_signature(dev, samplers) {
        Ok(rs) => rs,
        Err(e) => {
            note_translation_error(format!("dispatch-chain root signature: {e}"));
            return None;
        }
    };
    let mut g = SIGNATURES.lock().unwrap();
    if g.get(&key).is_some_and(|rs| !rs_belongs_to_device(rs, dev_key)) {
        g.insert(key, created.clone());
        return Some(created);
    }
    Some(g.entry(key).or_insert(created).clone())
}

pub(super) unsafe fn build(this: *mut c_void, create: PsoFn, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    iid: *const GUID, result: *mut *mut c_void, record: u64, whole: Arc<Vec<u8>>, samplers: &[pso::StaticSampler],
    plan: Arc<DispatchChain>) -> Option<usize> {
    if result.is_null() || iid.is_null() || plan.pieces.len() < 2 { return None; }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let signature = signature_for(dev, samplers)?;
    let rs = signature.as_raw() as usize;
    let builder = Arc::new(PipelineBuilder { device: dev.clone(), create, signature,
        samplers: samplers.to_vec(), flags: desc.Flags, node: desc.NodeMask, record });
    let jobs = (0..plan.pieces.len()).map(|i| {
        let plan = plan.clone();
        let builder = builder.clone();
        let iid = if i == 0 { *iid } else { pc_iid(Kind::PipelineState) };
        Box::new(move || builder.pipeline(&plan.pieces[i], &iid).ok().map_or(0, |p| p.into_raw() as usize)) as super::super::pso_async::Job
    }).collect();
    // Take ownership of every success, including when a different piece failed.
    let built: Vec<_> = super::super::pso_async::build_pieces(jobs).into_iter()
        .map(|p| (p != 0).then(|| ID3D12PipelineState::from_raw(p as *mut c_void))).collect();
    let all = built.into_iter().collect::<Option<Vec<_>>>()?;
    for (i, piece) in all.iter().enumerate() {
        super::super::dred::label_text(piece.as_raw() as usize, || format!("cs {record:016x} piece {i} of {}", all.len()));
    }
    let mut pieces = all.into_iter();
    let first = pieces.next()?;
    let chain = Arc::new(Chain {
        builder,
        tail: pieces.collect(),
        words: plan.dwords_per_group,
        whole_code: whole,
        whole: OnceLock::new(),
        whole_started: AtomicBool::new(false),
        retry_at: Mutex::new(None),
        signature: OnceLock::new(),
        slabs: Default::default(),
        warned: AtomicBool::new(false),
    });
    PROGRAMS.fetch_add(1, Ordering::Relaxed);
    PIECES.fetch_add(plan.pieces.len() as u64, Ordering::Relaxed);
    *result = first.into_raw();
    // a chain this one replaces (the same address made again) is dropped after the lock: its
    // pipelines' release goes back into this module, which takes `CHAINS` (a self-deadlock that
    // held a retail title's thread for good)
    let replaced = CHAINS.lock().unwrap().insert(*result as usize, chain);
    drop(replaced);
    HAS_CHAINS.store(true, Ordering::Release);
    Some(rs)
}

pub(super) fn forget(pso: usize) {
    let gone = CHAINS.lock().unwrap().remove(&pso);
    drop(gone); // releasing a tail PSO also goes through forget_pso
}

/// Fast check for whether `pso` is a split compute pipeline.
pub(in crate::hle::d3d12x) fn is_split(pso: usize) -> bool {
    HAS_CHAINS.load(Ordering::Acquire) && CHAINS.lock().unwrap().contains_key(&pso)
}

/// Resolve the whole compute pipeline for an `ExecuteIndirect` without stalling the recording
/// thread: when async PSO builds are enabled and the whole form is not compiled yet, kick off
/// its build once on a background thread and return `Pending`.
pub(in crate::hle::d3d12x) unsafe fn whole_nonblocking(pso: usize) -> WholeStatus {
    if !HAS_CHAINS.load(Ordering::Acquire) {
        return WholeStatus::NotSplit;
    }
    let Some(chain) = CHAINS.lock().unwrap().get(&pso).cloned() else {
        return WholeStatus::NotSplit;
    };
    match chain.whole.get() {
        Some(Some(p)) => {
            WHOLE_INDIRECT.fetch_add(1, Ordering::Relaxed);
            return WholeStatus::Ready(p.clone());
        }
        Some(None) => return WholeStatus::Failed,
        None => {}
    }
    if !super::super::pso_async::enabled() {
        return match chain.whole() {
            Ok(p) => {
                WHOLE_INDIRECT.fetch_add(1, Ordering::Relaxed);
                WholeStatus::Ready(p)
            }
            Err(_) => WholeStatus::Failed,
        };
    }
    if chain.whole_in_cooldown() {
        return WholeStatus::Failed;
    }
    chain.start_whole_async();
    WholeStatus::Pending
}

struct Scratch {
    buffer: ID3D12Resource,
    bytes: u64,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct Recording {
    scratch: Arc<Scratch>,
    first: ID3D12PipelineState,
    chain: Arc<Chain>,
    indirect: Option<Indirect>,
}

/// The commands of a chained ExecuteIndirect (`pso::record_slicer`), read when the GPU is done
/// with them (the allocator is reset): with more slabs than rounds, the next ExecuteIndirect of
/// the program records more, and the whole form is built in the background for it.
struct Indirect {
    commands: ID3D12Resource,
    at: u64,
    rounds: u32,
    /// the whole form ran instead under predication; `checked`: the predicate is XCL's own
    whole: bool,
    checked: bool,
}

impl Recording {
    fn read_back(&self) {
        let Some(i) = self.indirect.as_ref().filter(|i| i.checked) else { return };
        let mut p = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: i.at as usize, End: i.at as usize + 16 };
        if unsafe { i.commands.Map(0, Some(&range), Some(&mut p)) }.is_err() {
            return;
        }
        let [partial, _, slabs, _] = unsafe { std::ptr::read_volatile((p as *const u8).add(i.at as usize) as *const [u32; 4]) };
        unsafe { i.commands.Unmap(0, Some(&D3D12_RANGE::default())) };
        if partial == 0 {
            return;
        }
        self.chain.slabs.fetch_max(slabs, Ordering::Relaxed);
        if i.whole {
            INDIRECT_WHOLE.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let lost = slabs.saturating_sub(i.rounds);
        INDIRECT_LOST.fetch_add(u64::from(lost), Ordering::Relaxed);
        if !self.chain.warned.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!(
                "D3D12.X: ExecuteIndirect of a split program ({:#x}) had {slabs} slabs for {} rounds: {lost} lost; its whole form is built for the next ones",
                self.first.as_raw() as usize,
                i.rounds
            );
        }
        self.chain.start_whole_async();
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.read_back();
    }
}

static RECORDINGS: LazyLock<Mutex<HashMap<usize, Vec<Arc<Recording>>>>> = LazyLock::new(Default::default);
/// list → the allocator it records on and the scratch its chains share: they are dispatched one
/// after another, with a barrier between them (the recordings keep it alive)
static SHARED: LazyLock<Mutex<HashMap<usize, (usize, std::sync::Weak<Scratch>)>>> = LazyLock::new(Default::default);

/// `list` begins another recording, which the GPU may run beside the last one.
pub(in crate::hle::d3d12x) fn list_reset(list: usize) {
    SHARED.lock().unwrap().remove(&list);
}

#[cfg(test)]
pub(in crate::hle::d3d12x) fn seed_shared_for_test(list: usize, allocator: usize) {
    SHARED.lock().unwrap().insert(list, (allocator, std::sync::Weak::new()));
}

#[cfg(test)]
pub(in crate::hle::d3d12x) fn has_shared_for_test(list: usize) -> bool {
    SHARED.lock().unwrap().contains_key(&list)
}

/// `to` records right after `from` on the same allocator: its chains go on in `from`'s scratch.
pub(in crate::hle::d3d12x) fn list_follows(from: usize, to: usize) {
    let mut shared = SHARED.lock().unwrap();
    if let Some(e) = shared.get(&from).cloned() {
        shared.insert(to, e);
    }
}

/// The scratch of all recordings alive together, `XCL_SPLIT_SCRATCH_TOTAL_MB` (384). A recording
/// keeps its scratch until its allocator is reset, and a title that records ahead of the GPU has
/// many: with a bound on one dispatch alone a retail title's world took 1.2 GB and went on growing.
fn scratch_total() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_TOTAL_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(384u64) << 20)
}

/// A recording gets at least this much (or one group's worth) however much the others hold.
const LEAST: u64 = 2 << 20;

/// How much scratch a dispatch of `needed` bytes (`group` for one group) runs with at once: the
/// list's own buffer (`had` bytes) when that is enough or no more may be made, else a new one of
/// the bytes given (second: make one).
fn scratch_bytes(needed: u64, group: u64, budget: u64, had: Option<u64>, live: u64, total: u64) -> (u64, bool) {
    let want = needed.min(budget).max(group);
    let room = total.saturating_sub(live);
    match had {
        Some(h) if h >= want || (room < want && h >= group) => (h, false),
        _ => (want.min(room.max(LEAST)).max(group), true),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScratchOomFallback {
    /// Reuse the recording's existing scratch buffer (running smaller slabs).
    ReuseExisting,
    /// Allocate a minimal 1-workgroup scratch buffer instead of falling back to `Chain::whole()`.
    TryMinimal(u64),
    /// Neither existing reuse nor a smaller allocation is viable.
    None,
}

/// Choose a smaller-slab fallback when growing `scratch_buffer` fails under VRAM pressure.
fn scratch_oom_fallback(had_bytes: Option<u64>, attempted: u64, group: u64) -> ScratchOomFallback {
    if had_bytes.is_some_and(|h| h >= group) {
        ScratchOomFallback::ReuseExisting
    } else if attempted > group {
        ScratchOomFallback::TryMinimal(group)
    } else {
        ScratchOomFallback::None
    }
}

/// Guard for bytes atomically reserved from `LIVE` before `scratch_buffer` allocates GPU memory:
/// releases on drop if allocation fails, shrinks on OOM fallback, or transfers ownership to `Scratch`.
struct ScratchReservation<'a> {
    live: &'a AtomicU64,
    bytes: u64,
}

impl<'a> ScratchReservation<'a> {
    fn shrink(&mut self, new_bytes: u64) {
        if new_bytes < self.bytes {
            let release = self.bytes - new_bytes;
            self.live.fetch_sub(release, Ordering::AcqRel);
            self.bytes = new_bytes;
        }
    }

    fn commit(mut self) -> u64 {
        let bytes = self.bytes;
        self.bytes = 0;
        bytes
    }
}

impl Drop for ScratchReservation<'_> {
    fn drop(&mut self) {
        if self.bytes != 0 {
            self.live.fetch_sub(self.bytes, Ordering::AcqRel);
            self.bytes = 0;
        }
    }
}

/// Atomically decide whether to reuse `had` or reserve new scratch bytes from `live` BEFORE
/// creating the GPU buffer so concurrent recording threads cannot overshoot `total`.
fn reserve_scratch_bytes<'a>(
    live: &'a AtomicU64,
    peak: &AtomicU64,
    needed: u64,
    group: u64,
    budget: u64,
    had: Option<u64>,
    total: u64,
) -> Option<ScratchReservation<'a>> {
    let mut cur = live.load(Ordering::Acquire);
    loop {
        let (bytes, make) = scratch_bytes(needed, group, budget, had, cur, total);
        if !make {
            return None;
        }
        let next = cur.saturating_add(bytes);
        match live.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                peak.fetch_max(next, Ordering::Relaxed);
                return Some(ScratchReservation { live, bytes });
            }
            Err(observed) => cur = observed,
        }
    }
}

/// A scratch for a chain of `needed` bytes recorded on `list` with `allocator`, and whether a
/// chain before it on the list used it.
unsafe fn scratch_for(dev: &ID3D12Device, list: usize, allocator: usize, needed: u64, group: u64) -> Option<(Arc<Scratch>, bool)> {
    let had = SHARED.lock().unwrap().get(&list).filter(|s| s.0 == allocator).and_then(|s| s.1.upgrade());
    let had_bytes = had.as_ref().map(|s| s.bytes);
    let Some(mut reservation) = reserve_scratch_bytes(&LIVE, &PEAK, needed, group, scratch_budget(), had_bytes, scratch_total()) else {
        return had.map(|s| (s, true));
    };
    let bytes = reservation.bytes;
    let buffer = match scratch_buffer(dev, bytes) {
        Some(b) => b,
        None => match scratch_oom_fallback(had_bytes, bytes, group) {
            ScratchOomFallback::ReuseExisting => return had.map(|s| (s, true)),
            ScratchOomFallback::TryMinimal(min_bytes) => {
                reservation.shrink(min_bytes);
                scratch_buffer(dev, min_bytes)?
            }
            ScratchOomFallback::None => return None,
        },
    };
    let actual = reservation.commit();
    let s = Arc::new(Scratch { buffer, bytes: actual });
    SHARED.lock().unwrap().insert(list, (allocator, Arc::downgrade(&s)));
    Some((s, false))
}

pub(in crate::hle::d3d12x) fn allocator_reset(allocator: usize) {
    let recordings = RECORDINGS.lock().unwrap().remove(&allocator);
    drop(recordings);
}

unsafe fn scratch_buffer(dev: &ID3D12Device, bytes: u64) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut resource).ok()?;
    resource
}

/// Called with the original six ABI views bound. The extended signature is
/// common to every piece and the whole fallback. Choose before recording any
/// part of the dispatch, including when scratch allocation fails.
pub(in crate::hle::d3d12x) unsafe fn dispatch(list: &ID3D12GraphicsCommandList, pso: usize, allocator: usize, grid: [u32; 3]) -> Result<(), String> {
    let [x, y, z] = grid;
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { list.Dispatch(x, y, z); return Ok(()) };
    if grid.contains(&0) { return Ok(()); }
    // A title Reset can switch to a different PC allocator while this
    // recording is still submitted on the old one. Retain by backing.
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    let group = u64::from(chain.words) * 4;
    let needed = grid.into_iter().fold(group, |s, n| s.saturating_mul(n as u64));
    WANTED.fetch_max(needed, Ordering::Relaxed);
    let scratch = (allocator != 0 && list.GetType() != D3D12_COMMAND_LIST_TYPE_BUNDLE)
        .then(|| scratch_for(&chain.builder.device, list.as_raw() as usize, allocator, needed, group)).flatten();
    let most = scratch.as_ref().map_or(1, |(s, _)| (s.bytes / group).max(1));
    let slabs = pso::dispatch_slabs(grid, most);
    let Some((scratch, used)) = scratch else {
        match chain.whole.get() {
            Some(Some(whole)) => {
                list.SetPipelineState(whole);
                list.Dispatch(x, y, z);
                return Ok(());
            }
            Some(None) => return Err("whole compute pipeline".into()),
            None => {}
        }
        if !super::super::pso_async::enabled() {
            list.SetPipelineState(&chain.whole()?);
            list.Dispatch(x, y, z);
            return Ok(());
        }
        if chain.whole_in_cooldown() {
            return Err("whole compute pipeline: out of memory".into());
        }
        chain.start_whole_async();
        return Err("whole compute pipeline still building in background".into());
    };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    let held = Arc::new(Recording { scratch, first, chain, indirect: None });
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(held.clone());
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    SLABS.fetch_add(slabs.len() as u64, Ordering::Relaxed);
    pso::record_dispatch_chain(list, grid, most, &held.scratch.buffer, used, std::iter::once(&held.first).chain(&held.chain.tail));
    Ok(())
}

/// The rounds a chained ExecuteIndirect records at most, `XCL_SPLIT_INDIRECT_ROUNDS` (16): each
/// is one ExecuteIndirect and one barrier per piece, also when the title's arguments leave it empty.
fn most_rounds() -> u32 {
    static B: OnceLock<u32> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("XCL_SPLIT_INDIRECT_ROUNDS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(16u32)
            .min(1024)
    })
}

/// The rounds for slabs of `most` groups: as many as the arguments there now (`seen`, the GPU's
/// last ones) or an earlier ExecuteIndirect of the program (`before`) had, a quarter more, at
/// least two and at most `cap`.
fn indirect_rounds(seen: &[[u32; 3]], most: u64, before: u32, cap: u32) -> u32 {
    let now: u64 = seen.iter().map(|&g| pso::slab_count(g, most)).sum();
    let want = now.max(u64::from(before));
    (want + want.div_ceil(4)).clamp(2, u64::from(cap.max(2))) as u32
}

/// An ExecuteIndirect of split program `pso` whose command signature is one Dispatch of `stride`
/// bytes: `commands` of them at GPU address `args` (their count at `count`, when given), run as
/// the chain. `seen`: the arguments the CPU sees there now; `predicated`: the title's predication
/// is set. `bind` binds the ABI after the slicer, and `whole` is the title's ExecuteIndirect, run
/// with the whole form (built already) when the arguments have more slabs than the rounds.
/// Ok(false): nothing recorded (no chain, a bundle, no scratch) – the caller runs the whole form.
#[allow(clippy::too_many_arguments)]
pub(in crate::hle::d3d12x) unsafe fn execute_indirect(
    list: &ID3D12GraphicsCommandList,
    pso: usize,
    allocator: usize,
    args: u64,
    stride: u32,
    commands: u32,
    count: Option<u64>,
    seen: &[[u32; 3]],
    predicated: bool,
    bind: impl FnOnce() -> Result<(), String>,
    whole: impl FnOnce(),
) -> Result<bool, String> {
    if !HAS_CHAINS.load(Ordering::Acquire) {
        return Ok(false);
    }
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { return Ok(false) };
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else {
        allocator
    };
    if allocator == 0 || list.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return Ok(false);
    }
    let dev = &chain.builder.device;
    let group = u64::from(chain.words) * 4;
    let Some((scratch, _)) = scratch_for(dev, list.as_raw() as usize, allocator, scratch_budget(), group) else {
        return Ok(false);
    };
    let most = (scratch.bytes / group).max(1);
    let sig = match chain.signature.get() {
        Some(s) => s.clone(),
        None => {
            let s = pso::chain_command_signature(dev, &chain.builder.signature)?;
            chain.signature.get_or_init(|| s).clone()
        }
    };
    let rounds = indirect_rounds(seen, most, chain.slabs.load(Ordering::Relaxed), most_rounds());
    let pieces = chain.tail.len() as u32 + 1;
    let buffer = pso::chain_commands_buffer(dev, rounds, pieces).ok_or("no buffer for the commands of a chained ExecuteIndirect")?;
    let slicer_pso = pso::record_slicer(list, args, stride, commands, count, rounds, most, pieces, &buffer)?;
    if slicer_pso != 0 && SLICER_LABELED.swap(slicer_pso, Ordering::Relaxed) != slicer_pso {
        super::super::dred::label_text(slicer_pso, || "xcl dispatch_chain_slicer".into());
    }
    if let Err(e) = bind() {
        // `record_slicer` already recorded a dispatch writing into `buffer` and `scratch`:
        // keep both alive until `allocator` resets so the GPU never page-faults on a freed buffer.
        if let Some(first) = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)) {
            let indirect = Indirect {
                commands: buffer,
                at: pso::chain_predicate_offset(rounds, pieces),
                rounds,
                whole: false,
                checked: false,
            };
            RECORDINGS
                .lock()
                .unwrap()
                .entry(allocator)
                .or_default()
                .push(Arc::new(Recording { scratch, first: first.clone(), chain, indirect: Some(indirect) }));
        }
        return Err(e);
    }
    // under the title's own predication XCL's cannot be set: the slabs past the rounds are lost
    let full = if predicated { None } else { chain.whole.get().and_then(|o| o.clone()) };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    pso::record_dispatch_chain_indirect(list, &sig, &buffer, rounds, full.is_some(), &scratch.buffer, std::iter::once(&first).chain(&chain.tail));
    let at = pso::chain_predicate_offset(rounds, pieces);
    if let Some(full) = &full {
        list.SetPredication(&buffer, at, D3D12_PREDICATION_OP_EQUAL_ZERO);
        list.SetPipelineState(full);
        whole();
        list.SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }
    let indirect = Indirect {
        commands: buffer,
        at,
        rounds,
        whole: full.is_some(),
        checked: !predicated,
    };
    RECORDINGS
        .lock()
        .unwrap()
        .entry(allocator)
        .or_default()
        .push(Arc::new(Recording { scratch, first, chain, indirect: Some(indirect) }));
    INDIRECT.fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{scratch_bytes, LEAST};
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{
        ID3D12Device, ID3D12PipelineState, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_PIPELINE_STATE_FLAG_NONE,
    };

    const MB: u64 = 1 << 20;

    fn test_device() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D12::D3D12CreateDevice;
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            let warp = unsafe { CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) };
            let _ = warp.and_then(|w| unsafe { D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev) });
        }
        dev
    }

    fn warp_device() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Dxgi::*;
        let warp = unsafe { CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) }.ok()?;
        crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0).ok()
    }

    #[test]
    fn a_list_recorded_after_another_goes_on_in_its_scratch() {
        let Some(dev) = test_device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let (a, b, c, alloc) = (0x7E57_5C01usize, 0x7E57_5C02usize, 0x7E57_5C03usize, 0x7E57_5CA0usize);
        let (first, used) = unsafe { super::scratch_for(&dev, a, alloc, MB, 4096) }.unwrap();
        assert!(!used);
        super::list_follows(a, b);
        let (second, used) = unsafe { super::scratch_for(&dev, b, alloc, MB, 4096) }.unwrap();
        assert!(used && std::sync::Arc::ptr_eq(&first, &second), "the list after it made a scratch of its own");
        let (third, _) = unsafe { super::scratch_for(&dev, c, alloc, MB, 4096) }.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &third));
        for list in [a, b, c] {
            super::list_reset(list);
        }
    }

    #[test]
    fn a_chains_scratch_is_bounded_by_what_all_recordings_hold() {
        let group = 187 << 10;
        // the first of a recording: as much as it needs, up to the budget of one dispatch
        assert_eq!(scratch_bytes(15 * MB, group, 64 * MB, None, 0, 384 * MB), (15 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 0, 384 * MB), (64 * MB, true));
        // the others hold nearly all: what is left, and at least LEAST
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 364 * MB, 384 * MB), (20 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 1200 * MB, 384 * MB), (LEAST, true));
        // the list's own buffer: kept when big enough, or when no more may be made
        assert_eq!(scratch_bytes(8 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (16 * MB, false));
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 380 * MB, 384 * MB), (16 * MB, false));
        // … but made again, larger, while there is room
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (60 * MB, true));
        // one group always fits, whatever the bounds say
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, None, 400 * MB, 384 * MB), (9 * MB, true));
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, Some(MB), 400 * MB, 384 * MB), (9 * MB, true));
    }

    #[test]
    fn whole_nonblocking_returns_not_split_for_unknown_pso() {
        assert!(matches!(unsafe { super::whole_nonblocking(0) }, super::WholeStatus::NotSplit));
        assert!(matches!(unsafe { super::whole_nonblocking(0x7800_0001) }, super::WholeStatus::NotSplit));
    }

    #[test]
    fn scratch_oom_fallback_reuses_existing_or_shrinks_to_one_group() {
        use super::{scratch_oom_fallback, ScratchOomFallback};
        let group = 256 << 10;
        // Existing list scratch fits >= 1 group: reuse it with smaller slabs instead of Chain::whole.
        assert_eq!(scratch_oom_fallback(Some(2 * MB), 64 * MB, group), ScratchOomFallback::ReuseExisting);
        assert_eq!(scratch_oom_fallback(Some(group), 64 * MB, group), ScratchOomFallback::ReuseExisting);
        // No usable existing scratch, attempted > 1 group: shrink to minimal 1-group scratch.
        assert_eq!(scratch_oom_fallback(None, 64 * MB, group), ScratchOomFallback::TryMinimal(group));
        assert_eq!(scratch_oom_fallback(Some(group - 1), 4 * MB, group), ScratchOomFallback::TryMinimal(group));
        // Already attempted minimal 1-group allocation with no usable existing scratch: give up.
        assert_eq!(scratch_oom_fallback(None, group, group), ScratchOomFallback::None);
    }

    #[test]
    fn concurrent_scratch_admission_serializes_budget_reservations() {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc, Barrier,
        };
        let group = 256 << 10;
        let budget = 64 * MB;
        let total = 384 * MB;

        // Case 1: 12 concurrent threads whose lists already have a 4 MiB scratch buffer race to
        // grow to 64 MiB. Atomic reservation caps `peak <= total` (at most 6 grow; 6 reuse).
        let live = Arc::new(AtomicU64::new(0));
        let peak = Arc::new(AtomicU64::new(0));
        let start = Arc::new(Barrier::new(12));
        let hold = Arc::new(Barrier::new(12));
        let mut threads = Vec::new();
        for _ in 0..12 {
            let (live, peak, start, hold) = (Arc::clone(&live), Arc::clone(&peak), Arc::clone(&start), Arc::clone(&hold));
            threads.push(std::thread::spawn(move || {
                start.wait();
                let res = super::reserve_scratch_bytes(&live, &peak, budget, group, budget, Some(4 * MB), total);
                let granted = res.as_ref().map_or(0, |r| r.bytes);
                hold.wait();
                drop(res);
                granted
            }));
        }
        let grants: Vec<u64> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(grants.iter().filter(|&&b| b == budget).count(), 6);
        assert_eq!(grants.iter().filter(|&&b| b == 0).count(), 6);
        assert_eq!(peak.load(Ordering::Relaxed), total);
        assert_eq!(live.load(Ordering::Relaxed), 0);

        // Case 2: shrink + rollback on OOM fallback releases reserved bytes accurately.
        let mut r = super::reserve_scratch_bytes(&live, &peak, budget, group, budget, None, total).unwrap();
        assert_eq!(r.bytes, budget);
        assert_eq!(live.load(Ordering::Relaxed), budget);
        r.shrink(group);
        assert_eq!(r.bytes, group);
        assert_eq!(live.load(Ordering::Relaxed), group);
        drop(r);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn indirect_rounds_clamps_and_adds_headroom() {
        assert_eq!(super::indirect_rounds(&[], 100, 0, 16), 2);
        assert_eq!(super::indirect_rounds(&[[100, 4, 1]], 100, 0, 16), 5);
        assert_eq!(super::indirect_rounds(&[[100, 1, 1]], 100, 8, 16), 10);
        assert_eq!(super::indirect_rounds(&[[100, 40, 1]], 100, 0, 16), 16);
    }

    #[test]
    fn whole_worker_thread_name_matches_driver_compile_gate() {
        assert!(
            super::WHOLE_WORKER_THREAD_NAME.starts_with("xcl-pso"),
            "whole-form background worker must carry the xcl-pso prefix so driver_compile_identified gates it"
        );
    }

    #[test]
    fn split_plib_key_is_distinct_from_standard_compute_key() {
        let dxil = b"DXBC-fake-split-whole-dxil";
        let samplers = [([1, 2, 3, 4], false)];
        let std_key = super::super::pso_lib::compute_key(dxil, &samplers, 0);
        let split_key = super::split_plib_key(dxil, &samplers, 0);
        assert!(split_key.starts_with("sc-"));
        assert_ne!(split_key, std_key);
        assert_eq!(split_key, super::split_plib_key(dxil, &samplers, 0));
        assert_ne!(split_key, super::split_plib_key(dxil, &[], 0));
        assert_ne!(split_key, super::split_plib_key(dxil, &samplers, 1));
    }

    #[test]
    fn signature_for_reuses_root_signature_per_device_and_separates_devices_and_samplers() {
        let Some(dev1) = test_device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let Some(dev2) = warp_device().filter(|d| d.as_raw() != dev1.as_raw()) else {
            crate::gpu::recomp::runner::skip_test("second distinct D3D12 device unavailable");
            return;
        };
        let sampler: super::pso::StaticSampler = ([0x1111_0001, 0x2222_0002, 0x3333_0003, 0x4444_0004], false);
        unsafe {
            let rs1_a = super::signature_for(&dev1, &[]).expect("dispatch chain RS 1");
            let rs1_b = super::signature_for(&dev1, &[]).expect("dispatch chain RS 1 cached");
            assert_eq!(rs1_a.as_raw(), rs1_b.as_raw(), "repeated builds on the same device must share one RS");

            let rs1_samp_a = super::signature_for(&dev1, &[sampler]).expect("dispatch chain RS with sampler");
            let rs1_samp_b = super::signature_for(&dev1, &[sampler]).expect("dispatch chain RS with sampler cached");
            assert_eq!(rs1_samp_a.as_raw(), rs1_samp_b.as_raw());
            assert_ne!(rs1_samp_a.as_raw(), rs1_a.as_raw(), "static samplers must partition the RS cache");

            let rs2 = super::signature_for(&dev2, &[]).expect("dispatch chain RS for second device");
            assert_ne!(rs2.as_raw(), rs1_a.as_raw(), "distinct D3D12 devices must never share COM root signatures");

            let cs = super::super::dummy().cs.clone();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs2.clone())),
                CS: super::super::bytecode(&cs),
                ..Default::default()
            };
            let _pso: ID3D12PipelineState = dev2.CreateComputePipelineState(&desc).expect("dev2 accepts its own dispatch-chain RS");
        }
    }

    #[test]
    fn signature_retention_and_device_child_lifetime() {
        let Some(dev_hw) = test_device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let Some(dev_warp) = warp_device() else {
            crate::gpu::recomp::runner::skip_test("WARP device unavailable");
            return;
        };
        unsafe {
            let warp_key = super::canonical_device_key(dev_warp.as_raw());
            let refs_of = |d: &ID3D12Device| -> u32 {
                type F = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
                let vt = *(d.as_raw() as *const *const usize);
                let add: F = std::mem::transmute(*vt.add(1));
                let rel: F = std::mem::transmute(*vt.add(2));
                add(d.as_raw());
                rel(d.as_raw())
            };
            let before_refs = refs_of(&dev_warp);
            let rs_warp = super::signature_for(&dev_warp, &[]).expect("WARP dispatch chain RS");
            let after_refs = refs_of(&dev_warp);
            assert!(super::rs_belongs_to_device(&rs_warp, warp_key));
            let owner = super::root_signature_device(&rs_warp).expect("GetDevice while device live");
            assert_eq!(super::canonical_device_key(owner.as_raw()), warp_key);
            drop(owner);
            drop(rs_warp);
            drop(dev_warp);

            // While `SIGNATURES` retains the child root signature, inspect `GetDevice` and cache entry count.
            let retained = super::SIGNATURES.lock().unwrap().get(&(warp_key, Vec::new())).cloned().expect("retained child RS");
            let resurrected_owner = super::root_signature_device(&retained).expect("retained ID3D12RootSignature keeps parent device queryable");
            assert_eq!(super::canonical_device_key(resurrected_owner.as_raw()), warp_key);

            let sig_count_before = super::SIGNATURES.lock().unwrap().len();
            let _ = super::signature_for(&dev_hw, &[]).expect("HW dispatch chain RS");
            let _ = super::signature_for(&dev_hw, &[]).expect("HW dispatch chain RS cached");
            let sig_count_after = super::SIGNATURES.lock().unwrap().len();
            assert!(sig_count_after <= sig_count_before + 1, "repeated calls must not grow SIGNATURES");
            println!("SIGNATURES_LIFETIME before_refs={before_refs} after_refs={after_refs} cached_entries={sig_count_after}");
        }
    }

    #[test]
    fn split_programs_sharing_large_piece_hit_compute_blobs_and_reuse_root_signature() {
        use std::ffi::c_void;
        use std::sync::{atomic::Ordering, Arc, Mutex};

        if crate::gpu::recomp::dxc::location().is_err() {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        }
        let Some(dev) = test_device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };

        // Compile unique >= 16 KB DXIL pieces (`>= COMPUTE_BLOB_MIN_DXIL`) with a per-run salt so
        // earlier runs' disk caches cannot satisfy Program A cold builds.
        let nonce = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u32) ^ std::process::id();
        let compile_piece = |rounds: u32, extra_xor: u32| {
            let hlsl = format!(
                "RWByteAddressBuffer dst : register(u0, space0);\n\
                 [numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {{\n\
                     uint x = id.x ^ 0x{nonce:08x}u ^ 0x{extra_xor:08x}u;\n\
                     [unroll] for (uint i = 0; i < {rounds}; ++i) {{\n\
                         x = (x * 1664525u + (i ^ 0x9e3779b9u)) ^ (x >> 13);\n\
                     }}\n\
                     dst.Store(0, x);\n\
                 }}\n"
            );
            crate::gpu::recomp::dxc::compile(&hlsl, "main", "cs_6_0", &[]).expect("compile cs_6_0 piece")
        };
        let mut rounds = 600u32;
        let mut shared = compile_piece(rounds, 0);
        while shared.len() < super::super::COMPUTE_BLOB_MIN_DXIL {
            rounds *= 2;
            shared = compile_piece(rounds, 0);
        }
        let tail_a = compile_piece(12, 1);
        let tail_b = compile_piece(12, 2);
        let whole = compile_piece(16, 3);

        static SPY_LOCK: Mutex<()> = Mutex::new(());
        static SPY_CALLS: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::new());
        let _serial = SPY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SPY_CALLS.lock().unwrap().clear();

        struct FixtureCleanup {
            psos: Vec<ID3D12PipelineState>,
            blob_keys: Vec<(u64, usize, i32)>,
            disk_dxils: Vec<Vec<u8>>,
        }
        impl Drop for FixtureCleanup {
            fn drop(&mut self) {
                for pso in self.psos.drain(..) {
                    super::forget(pso.as_raw() as usize);
                }
                for key in self.blob_keys.drain(..) {
                    super::super::forget_compute_blob(key);
                }
                for bytes in self.disk_dxils.drain(..) {
                    if let Some(p) = super::super::driver_cache::path_for(&bytes, 0, true) {
                        let _ = std::fs::remove_file(p);
                    }
                }
                SPY_CALLS.lock().unwrap().clear();
            }
        }
        let mut cleanup = FixtureCleanup {
            psos: Vec::new(),
            blob_keys: Vec::new(),
            disk_dxils: vec![shared.clone(), tail_a.clone(), tail_b.clone()],
        };

        unsafe extern "system" fn spy_create(this: *mut c_void, desc: *const c_void, _iid: *const super::GUID, out: *mut *mut c_void) -> super::HRESULT {
            let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
            SPY_CALLS.lock().unwrap().push((d.CS.BytecodeLength, !d.CachedPSO.pCachedBlob.is_null()));
            let dev = ID3D12Device::from_raw_borrowed(&this).unwrap();
            match dev.CreateComputePipelineState::<ID3D12PipelineState>(d) {
                Ok(pso) => {
                    *out = pso.into_raw();
                    0
                }
                Err(e) => e.code().0,
            }
        }

        let base_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        let iid = &ID3D12PipelineState::IID as *const windows::core::GUID as *const super::GUID;
        let plan_a = Arc::new(super::DispatchChain { pieces: vec![shared.clone(), tail_a.clone()], dwords_per_group: 4 });
        let plan_b = Arc::new(super::DispatchChain { pieces: vec![shared.clone(), tail_b.clone()], dwords_per_group: 4 });

        let whole = Arc::new(whole);
        unsafe {
            let mut raw_a: *mut c_void = std::ptr::null_mut();
            let rs_a = super::build(dev.as_raw(), spy_create, &base_desc, iid, &mut raw_a, 1, whole.clone(), &[], plan_a.clone()).expect("build Program A");
            cleanup.psos.push(ID3D12PipelineState::from_raw(raw_a));
            cleanup.blob_keys.push(super::super::compute_blob_key(&shared, rs_a, 0));
            cleanup.blob_keys.push(super::super::compute_blob_key(&tail_a, rs_a, 0));
            let a_calls = std::mem::take(&mut *SPY_CALLS.lock().unwrap());
            assert_eq!(a_calls.len(), 2, "Program A should compile 2 pieces, got {a_calls:?}");
            assert!(a_calls.iter().any(|&(len, cached)| len == shared.len() && !cached), "Program A shared piece should start cold: {a_calls:?}");

            let sigs_after_a = super::SIGNATURES.lock().unwrap().len();
            let key_shared = super::super::compute_blob_key(&shared, rs_a, 0);
            let count_shared_keys = || {
                super::super::COMPUTE_BLOBS.lock().unwrap().as_ref().map_or(0, |(m, _)| m.keys().filter(|k| k.0 == key_shared.0).count())
            };
            let stored_blob = count_shared_keys();
            assert!(stored_blob <= 1, "shared piece driver blob stored at most once in COMPUTE_BLOBS (0 on AMD stub-blob drivers)");

            let plib_hits_before_b = super::PLIB_HITS.load(Ordering::Relaxed);
            let mut raw_b: *mut c_void = std::ptr::null_mut();
            let rs_b = super::build(dev.as_raw(), spy_create, &base_desc, iid, &mut raw_b, 2, whole.clone(), &[], plan_b).expect("build Program B");
            cleanup.psos.push(ID3D12PipelineState::from_raw(raw_b));
            cleanup.blob_keys.push(super::super::compute_blob_key(&tail_b, rs_b, 0));
            let b_calls = std::mem::take(&mut *SPY_CALLS.lock().unwrap());
            assert_eq!(rs_b, rs_a, "Program B must reuse Program A's cached dispatch-chain root signature");
            assert_eq!(super::SIGNATURES.lock().unwrap().len(), sigs_after_a, "Program B must not allocate a new SIGNATURES entry");
            if stored_blob == 1 {
                assert!(
                    super::PLIB_HITS.load(Ordering::Relaxed) > plib_hits_before_b
                        || b_calls.iter().any(|&(len, cached)| len == shared.len() && cached),
                    "Program B must reuse shared piece from .plib or COMPUTE_BLOBS via CachedPSO: {b_calls:?}"
                );
            }
            assert_eq!(count_shared_keys(), stored_blob, "Program B must not insert a duplicate COMPUTE_BLOBS entry under another root signature");

            // Dropping and rebuilding Program A must also hit .plib or COMPUTE_BLOBS with no extra root signature.
            let pso_a = cleanup.psos.remove(0);
            super::forget(pso_a.as_raw() as usize);
            drop(pso_a);
            let plib_hits_before_a2 = super::PLIB_HITS.load(Ordering::Relaxed);
            let mut raw_a2: *mut c_void = std::ptr::null_mut();
            let rs_a2 = super::build(dev.as_raw(), spy_create, &base_desc, iid, &mut raw_a2, 3, whole.clone(), &[], plan_a).expect("rebuild Program A");
            cleanup.psos.push(ID3D12PipelineState::from_raw(raw_a2));
            let a2_calls = std::mem::take(&mut *SPY_CALLS.lock().unwrap());
            assert_eq!(rs_a2, rs_a);
            assert_eq!(super::SIGNATURES.lock().unwrap().len(), sigs_after_a);
            if stored_blob == 1 {
                assert!(
                    super::PLIB_HITS.load(Ordering::Relaxed) > plib_hits_before_a2
                        || a2_calls.iter().any(|&(len, cached)| len == shared.len() && cached),
                    "rebuild must hit .plib or COMPUTE_BLOBS: {a2_calls:?}"
                );
            }
            assert_eq!(count_shared_keys(), stored_blob);
            println!("DISPATCH_CHAIN_RS_CACHE shared_dxil={} sigs={}", shared.len(), sigs_after_a);
        }
    }

    #[test]
    fn a_failed_whole_form_is_not_built_again() {
        use std::ffi::c_void;
        use std::sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Arc, Mutex, OnceLock,
        };
        static CALLS: AtomicU64 = AtomicU64::new(0);
        unsafe extern "system" fn refused(_: *mut c_void, _: *const c_void, _: *const super::GUID, _: *mut *mut c_void) -> super::HRESULT {
            CALLS.fetch_add(1, Ordering::Relaxed);
            crate::util::E_FAIL
        }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let signature = super::pso::dispatch_chain_root_signature(&dev, &[]).unwrap();
            let builder = Arc::new(super::PipelineBuilder {
                device: dev,
                create: refused,
                signature,
                samplers: Vec::new(),
                flags: D3D12_PIPELINE_STATE_FLAG_NONE,
                node: 0,
                record: 0,
            });
            let chain = super::Chain {
                builder,
                tail: Vec::new(),
                words: 4,
                whole_code: Arc::new(vec![0; 16]),
                whole: OnceLock::new(),
                whole_started: AtomicBool::new(false),
                retry_at: Mutex::new(None),
                signature: OnceLock::new(),
                slabs: Default::default(),
                warned: AtomicBool::new(false),
            };
            assert!(chain.whole().is_err());
            assert!(chain.whole().is_err());
            assert_eq!(CALLS.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn a_whole_form_that_ran_out_of_memory_is_tried_once_more_after_a_pause() {
        use std::ffi::c_void;
        use std::sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Arc, Mutex, OnceLock,
        };
        static CALLS: AtomicU64 = AtomicU64::new(0);
        unsafe extern "system" fn out_of_memory(_: *mut c_void, _: *const c_void, _: *const super::GUID, _: *mut *mut c_void) -> super::HRESULT {
            CALLS.fetch_add(1, Ordering::Relaxed);
            crate::util::E_OUTOFMEMORY
        }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let signature = super::pso::dispatch_chain_root_signature(&dev, &[]).unwrap();
            let builder = Arc::new(super::PipelineBuilder {
                device: dev,
                create: out_of_memory,
                signature,
                samplers: Vec::new(),
                flags: D3D12_PIPELINE_STATE_FLAG_NONE,
                node: 0,
                record: 0,
            });
            let chain = super::Chain {
                builder,
                tail: Vec::new(),
                words: 4,
                whole_code: Arc::new(vec![0; 16]),
                whole: OnceLock::new(),
                whole_started: AtomicBool::new(false),
                retry_at: Mutex::new(None),
                signature: OnceLock::new(),
                slabs: Default::default(),
                warned: AtomicBool::new(false),
            };
            assert!(chain.whole().is_err());
            assert!(chain.whole().is_err());
            assert_eq!(CALLS.load(Ordering::Relaxed), 1);
            *chain.retry_at.lock().unwrap() = Some(std::time::Instant::now());
            assert!(chain.whole().is_err());
            assert_eq!(CALLS.load(Ordering::Relaxed), 2);
            *chain.retry_at.lock().unwrap() = Some(std::time::Instant::now());
            assert!(chain.whole().is_err());
            assert_eq!(CALLS.load(Ordering::Relaxed), 2);
        }
    }
}
