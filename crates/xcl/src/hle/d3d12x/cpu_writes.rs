//! Write32/64BitValueBottomOfPipeX into memory the GPU does not see: done by the CPU once the GPU
//! has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Here each queue gets a fence of XCL's: after a submission whose lists carry such writes the
//! queue signals it, and a worker thread does the writes, in submission order, when the GPU is
//! there – what bottom of pipe means.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32_FLOAT};
use windows_sys::Win32::System::Threading as th;

use crate::hle::xg;

/// `XCL_WATER_READBACK=0`: disable GPU->UMA readback of placed ocean height/displacement/buoyancy
/// textures (`256×256` `R32_FLOAT`/`R32G32_FLOAT`, `64×64` `R32G32_FLOAT`, `128×128` `R16G16_UINT`).
pub(super) fn water_readback_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_WATER_READBACK").map_or(true, |v| v.trim() != "0"))
}

/// Whether `desc` matches one of the 4 verified single-mip 2D ocean height/displacement/buoyancy
/// readback textures used by CPU water/boat physics:
/// - `256×256` `R32_FLOAT` (`41`) or `R32G32_FLOAT` (`16`)
/// - `64×64` `R32G32_FLOAT` (`16`)
/// - `128×128` `R16G16_UINT` (`36`)
pub(super) fn is_water_readback_desc(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.MipLevels == 1
        && desc.SampleDesc.Count <= 1
        && matches!(
            (desc.Width, desc.Height, desc.Format),
            (256, 256, DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32G32_FLOAT)
                | (64, 64, DXGI_FORMAT_R32G32_FLOAT)
                | (128, 128, DXGI_FORMAT_R16G16_UINT)
        )
}

#[derive(Clone)]
pub(super) struct WaterTarget {
    pub va: usize,
    pub format: DXGI_FORMAT,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u32,
    pub tex: Arc<xg::Texture>,
}

static WATER_TARGETS: Mutex<Option<HashMap<usize, WaterTarget>>> = Mutex::new(None);
static HAS_WATER_TARGETS: AtomicBool = AtomicBool::new(false);

pub(super) fn has_water_readback() -> bool {
    HAS_WATER_TARGETS.load(Ordering::Acquire)
}

pub(super) fn register_water_readback(res: usize, va: usize, format: DXGI_FORMAT, width: u32, height: u32, tex: Arc<xg::Texture>) {
    if !water_readback_enabled() || res == 0 || va == 0 {
        return;
    }
    let bpe: u32 = match format {
        DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R16G16_UINT => 4,
        DXGI_FORMAT_R32G32_FLOAT => 8,
        _ => return,
    };
    let Some(row_bytes) = width.checked_mul(bpe) else { return };
    // Align to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256 B); all 4 verified formats are exact multiples.
    let row_pitch = row_bytes.next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    WATER_TARGETS.lock().unwrap().get_or_insert_with(HashMap::new).insert(
        res,
        WaterTarget { va, format, width, height, row_pitch, tex },
    );
    HAS_WATER_TARGETS.store(true, Ordering::Release);
}

pub(super) fn unregister_water_readback(res: usize) {
    if !HAS_WATER_TARGETS.load(Ordering::Acquire) {
        return;
    }
    let mut g = WATER_TARGETS.lock().unwrap();
    if let Some(m) = g.as_mut() {
        m.remove(&res);
        if m.is_empty() {
            HAS_WATER_TARGETS.store(false, Ordering::Release);
        }
    }
}

pub(super) fn water_readback_target(res: usize) -> Option<WaterTarget> {
    if !water_readback_enabled() || !HAS_WATER_TARGETS.load(Ordering::Acquire) {
        return None;
    }
    WATER_TARGETS.lock().unwrap().as_ref()?.get(&res).cloned()
}

/// Pooled persistently-mapped `READBACK` buffer returned to `FREE_WATER_RB` on drop once
/// `xcl-cpu-writes` finishes swizzling the heightmap into console memory.
pub(super) struct WaterReadbackBuf {
    pub buf: Option<ID3D12Resource>,
    pub mapped: usize,
    pub bytes: u64,
    pub dev_key: usize,
}

unsafe impl Send for WaterReadbackBuf {}
unsafe impl Sync for WaterReadbackBuf {}

const MAX_POOLED_WATER_RB: usize = 16;
static FREE_WATER_RB: Mutex<Vec<(ID3D12Resource, usize, u64, usize)>> = Mutex::new(Vec::new());

impl Drop for WaterReadbackBuf {
    fn drop(&mut self) {
        let Some(buf) = self.buf.take() else { return };
        let mapped = std::mem::replace(&mut self.mapped, 0);
        let mut free = FREE_WATER_RB.lock().unwrap();
        if free.len() < MAX_POOLED_WATER_RB && mapped != 0 {
            free.push((buf, mapped, self.bytes, self.dev_key));
        } else if mapped != 0 {
            unsafe { buf.Unmap(0, None) };
        }
    }
}

/// Acquire a persistently-mapped `READBACK` buffer of at least `bytes` bytes for `dev`.
pub(super) unsafe fn take_water_readback_buf(dev: &ID3D12Device, bytes: u64) -> Option<Arc<WaterReadbackBuf>> {
    let dev_key = dev.as_raw() as usize;
    {
        let mut free = FREE_WATER_RB.lock().unwrap();
        if let Some(i) = free.iter().position(|(_, _, cap, d)| *d == dev_key && *cap >= bytes) {
            let (buf, mapped, cap, d) = free.swap_remove(i);
            return Some(Arc::new(WaterReadbackBuf { buf: Some(buf), mapped, bytes: cap, dev_key: d }));
        }
    }
    let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let mut rb: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).ok()?;
    let rb = rb?;
    let _ = rb.SetName(windows::core::w!("xcl water heightmap readback"));
    let mut mapped = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut mapped)).ok()?;
    if mapped.is_null() {
        return None;
    }
    Some(Arc::new(WaterReadbackBuf { buf: Some(rb), mapped: mapped as usize, bytes, dev_key }))
}

/// A bottom-of-pipe write into memory outside unified memory, applied by the CPU once the
/// queue's fence reaches the submission that recorded it.
#[derive(Clone)]
pub(super) enum Write {
    /// Immediate dwords to store starting at `va`.
    Immediate(usize, Vec<u32>),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
    /// A 256×256 `R32_FLOAT` / `R32G32_FLOAT` ocean height/displacement texture read back from
    /// VRAM into a persistently mapped READBACK buffer, tiled back into console memory `dst_va`
    /// with `tex.chain.tile_level` once the queue's fence completes.
    ReadbackTexture { dst_va: usize, format: DXGI_FORMAT, row_pitch: u32, tex: Arc<xg::Texture>, buf: Arc<WaterReadbackBuf> },
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    writes: Vec<Write>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence and the last value signalled
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

/// how long the worker waits for one submission before writing anyway (a removed device, a
/// queue that never gets there)
const WAIT_MS: u32 = 10_000;

fn worker() -> &'static Mutex<mpsc::Sender<Job>> {
    static W: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    W.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        let spawned = std::thread::Builder::new().name("xcl-cpu-writes".into()).spawn(move || unsafe {
            let event = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            for job in rx {
                if job.fence.GetCompletedValue() < job.value && job.fence.SetEventOnCompletion(job.value, HANDLE(event as *mut c_void)).is_ok() {
                    th::WaitForSingleObject(event, WAIT_MS);
                    if job.fence.GetCompletedValue() < job.value {
                        static LATE: AtomicU64 = AtomicU64::new(0);
                        let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                        if k.is_power_of_two() {
                            crate::xlog_warn!("D3D12.X: {k} writes after the GPU so far done although their queue had not got there after {WAIT_MS} ms (fence at {}, wanted {}; {} writes)", job.fence.GetCompletedValue(), job.value, job.writes.len());
                        }
                    }
                }
                for w in &job.writes {
                    match w {
                        Write::Immediate(va, words) => write_now(*va, words),
                        Write::ReadbackU64 { va, src, .. } => {
                            let value = (*(*src as *const AtomicU64)).load(Ordering::Acquire);
                            write_u64_now(*va, value);
                        }
                        Write::ReadbackTexture { dst_va, format, row_pitch, tex, buf } => {
                            write_texture_now(*dst_va, *format, *row_pitch, tex, buf);
                        }
                    }
                }
            }
        });
        if let Err(e) = spawned {
            crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
        }
        Mutex::new(tx)
    })
}

/// The dwords at `va`, if the memory is still there and writable.
unsafe fn write_now(va: usize, words: &[u32]) {
    let bytes = 4 * words.len();
    if !crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
            return;
        }
        if m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 || (m.BaseAddress as usize + m.RegionSize) < va + bytes {
            return;
        }
    }
    for (i, &w) in words.iter().enumerate() {
        (*((va + 4 * i) as *const AtomicU32)).store(w, Ordering::Release);
    }
}

/// Whether `[va, va + 8)` is 8-byte aligned, committed, and writable without `PAGE_GUARD`.
unsafe fn u64_writable_now(va: usize) -> bool {
    if va == 0 || !va.is_multiple_of(8) {
        return false;
    }
    let Some(end) = va.checked_add(8) else { return false };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, 8)) {
        return true;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return false;
    }
    m.Protect & PAGE_GUARD == 0
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && (m.BaseAddress as usize).saturating_add(m.RegionSize) >= end
}

/// An 8-byte aligned 64-bit value at `va`, if the memory is still there and writable.
unsafe fn write_u64_now(va: usize, value: u64) {
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).store(value, Ordering::Release);
    }
}

/// Advance an 8-byte aligned 64-bit value at `va` to at least `value` (`fetch_max`), if the memory
/// is still committed and writable. Best-effort check before the atomic write (same TOCTOU
/// trade-off as `write_u64_now`: a concurrent title decommit between check and store is unguarded).
pub(super) unsafe fn max_u64_now(va: usize, value: u64) {
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).fetch_max(value, Ordering::AcqRel);
    }
}

/// Whether the entire byte span `[va, va + bytes)` is committed, writable, and without `PAGE_GUARD`.
unsafe fn range_writable_now(va: usize, bytes: usize) -> bool {
    if va == 0 || bytes == 0 {
        return false;
    }
    let Some(end) = va.checked_add(bytes) else { return false };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        return true;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let mut cur = va;
    while cur < end {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(cur as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
            return false;
        }
        if m.Protect & PAGE_GUARD != 0 || m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 {
            return false;
        }
        let region_end = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if region_end <= cur {
            return false;
        }
        cur = region_end;
    }
    true
}

fn main_surface(tex: &xg::Texture) -> Option<(usize, &xg::Surface)> {
    let plane = tex.planes.get(tex.main)?;
    let xg::PlaneKind::Surface(ref s) = plane.kind else { return None };
    Some((plane.offset as usize, s))
}

/// Swizzle `linear` rows (`row_pitch` bytes per row, 256 rows) into Xbox console tiled layout
/// inside `dst` using the main plane's `Surface::chain.tile_level`.
fn tile_readback_into_slice(dst: &mut [u8], linear: &[u8], row_pitch: u32, tex: &xg::Texture) -> bool {
    let Some((_, surface)) = main_surface(tex) else { return false };
    let Some(l0) = surface.chain.levels.first() else { return false };
    let bpe = surface.bpe as usize;
    let packed_row = l0.width.saturating_mul(bpe);
    let pitch = row_pitch as usize;
    let slice_bytes = surface.chain.slice_bytes;
    if packed_row == 0 || pitch < packed_row || slice_bytes == 0 || dst.len() < slice_bytes {
        return false;
    }
    let Some(src_need) = pitch.checked_mul(l0.height) else { return false };
    if linear.len() < src_need {
        return false;
    }
    if pitch == packed_row {
        surface.chain.tile_level(&mut dst[..slice_bytes], 0, 0, &linear[..src_need]);
    } else {
        let mut packed = Vec::with_capacity(packed_row * l0.height);
        for row in 0..l0.height {
            let start = row * pitch;
            packed.extend_from_slice(&linear[start..start + packed_row]);
        }
        surface.chain.tile_level(&mut dst[..slice_bytes], 0, 0, &packed);
    }
    true
}

/// Write a completed GPU readback of a 256×256 ocean height/displacement texture back into
/// console memory in its Xbox XOR-swizzled layout.
unsafe fn write_texture_now(dst_va: usize, format: DXGI_FORMAT, row_pitch: u32, tex: &xg::Texture, buf: &WaterReadbackBuf) {
    if buf.mapped == 0 {
        return;
    }
    let Some((plane_off, surface)) = main_surface(tex) else { return };
    let Some(target_va) = super::title_textures::mirror_source(dst_va).checked_add(plane_off) else { return };
    let dst_bytes = surface.chain.slice_bytes;
    if dst_bytes == 0 {
        return;
    }
    let _open = crate::hle::kernelx::lazy::ensure(target_va, dst_bytes);
    if !range_writable_now(target_va, dst_bytes) {
        return;
    }
    let linear = std::slice::from_raw_parts(buf.mapped as *const u8, buf.bytes as usize);
    let dst = std::slice::from_raw_parts_mut(target_va as *mut u8, dst_bytes);
    if tile_readback_into_slice(dst, linear, row_pitch, tex) {
        std::sync::atomic::fence(Ordering::Release);
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let n = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || (n.is_power_of_two() && n >= 64) {
            let (w, h) = surface.chain.levels.first().map_or((0, 0), |l| (l.width, l.height));
            crate::xlog_info!("D3D12.X: water heightmap readback #{n} ({w}x{h} fmt {}, -> {target_va:#x}, {dst_bytes} B)", format.0);
        }
    }
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        return true;
    }
    // (held while the protection is asked: a view of a physical page is not guarded again meanwhile)
    let _open = crate::hle::kernelx::lazy::ensure(va, bytes);
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0;
    found
        && m.State == MEM_COMMIT
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

/// `writes` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    if writes.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let job = {
        let mut g = FENCES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = super::real::create_raw_fence(&dev) else { return };
                v.insert(QueueFence { fence, value: 0 })
            }
        };
        entry.value += 1;
        if q.Signal(&entry.fence, entry.value).is_err() {
            return;
        }
        Job { fence: entry.fence.clone(), value: entry.value, writes }
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len()));
    let _ = worker().lock().unwrap().send(job);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::Immediate(va, vec![7, 9])]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU32)).load(Ordering::Acquire) != 7 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }

    #[test]
    fn water_readback_desc_matches_verified_water_textures_only() {
        let base = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_FLOAT,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(is_water_readback_desc(&base));
        assert!(is_water_readback_desc(&D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_R32G32_FLOAT, ..base }));
        assert!(is_water_readback_desc(&D3D12_RESOURCE_DESC { Width: 64, Height: 64, Format: DXGI_FORMAT_R32G32_FLOAT, ..base }));
        assert!(is_water_readback_desc(&D3D12_RESOURCE_DESC { Width: 128, Height: 128, Format: DXGI_FORMAT_R16G16_UINT, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { Width: 64, Height: 64, Format: DXGI_FORMAT_R32_FLOAT, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { Width: 256, Height: 256, Format: DXGI_FORMAT_R16G16_UINT, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { Width: 512, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { MipLevels: 2, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { DepthOrArraySize: 2, ..base }));
        assert!(!is_water_readback_desc(&D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, ..base }));
    }

    #[test]
    fn tile_readback_roundtrips_all_verified_water_textures() {
        for (dim, fmt, bpe) in [
            (256u32, DXGI_FORMAT_R32_FLOAT.0 as u32, 4usize),
            (256u32, DXGI_FORMAT_R32G32_FLOAT.0 as u32, 8usize),
            (64u32, DXGI_FORMAT_R32G32_FLOAT.0 as u32, 8usize),
            (128u32, DXGI_FORMAT_R16G16_UINT.0 as u32, 4usize),
        ] {
            let mut x = [0u8; 0x38];
            x[0..4].copy_from_slice(&(D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32).to_le_bytes());
            x[0x10..0x18].copy_from_slice(&(dim as u64).to_le_bytes());
            x[0x18..0x1C].copy_from_slice(&dim.to_le_bytes());
            x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            x[0x20..0x24].copy_from_slice(&fmt.to_le_bytes());
            x[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            x[0x2C..0x30].copy_from_slice(&0x11Bu32.to_le_bytes()); // console swizzle 64KB_R_X (0x100 | 27)
            let tex = xg::Texture::new(&x).expect("console water texture layout");
            let (_, surface) = main_surface(&tex).expect("main color surface");
            let n = dim as usize;
            let linear: Vec<u8> = (0..(n * n * bpe)).map(|i| ((i * 31 + 17) & 0xFF) as u8).collect();
            let mut tiled = vec![0u8; surface.chain.slice_bytes];
            assert!(tile_readback_into_slice(&mut tiled, &linear, (n * bpe) as u32, &tex));
            assert_eq!(surface.chain.detile_level(&tiled, 0, 0), linear);
        }
    }
}
