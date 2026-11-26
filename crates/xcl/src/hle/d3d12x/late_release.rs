//! The title's last `Release` of a resource, put off until the GPU is past the work in flight.
//!
//! D3D12 on the PC asks that an object outlives the GPU work that uses it. A console title
//! keeps to that by its own count of finished frames – but XCL's submits are not the title's:
//! texture uploads go out with a later submit, tile mappings on a queue of XCL's own. A retail
//! title's final `Release` of a resource then came while a queue still had work on it (debug
//! layer: "CORRUPTION: … is referenced by GPU operations in-flight", id 921).
//!
//! The last reference is held here instead. Each batch of held resources gets a fence value
//! signalled on every queue with work on it; the batch goes once all of them have passed it.
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// a batch is not closed more often than this (a `Signal` a queue each time)
const SEAL_EVERY: Duration = Duration::from_millis(10);
/// how often the thread here looks for batches that can go
const TICK: Duration = Duration::from_millis(4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeferredItem {
    Com(usize),
    Uma { ptr: usize, sparse: bool },
}

struct Batch {
    dead: Vec<DeferredItem>,
    /// (fence, value) a queue: all reached = nothing submitted before the batch closed is left
    marks: Vec<(ID3D12Fence, u64)>,
}

#[derive(Default)]
struct Held {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    open: Vec<DeferredItem>,
    sealed: VecDeque<Batch>,
    last_seal: Option<Instant>,
}

// (the interface pointers are only used under the lock)
unsafe impl Send for Held {}

impl Held {
    fn is_empty(&self) -> bool {
        self.open.is_empty() && self.sealed.is_empty()
    }

    fn needs_seal(&self, seal_now: bool) -> bool {
        !self.open.is_empty() && (seal_now || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY))
    }

    /// Take out the sealed batches every queue is past without touching `open`.
    unsafe fn reap(&mut self) -> Vec<DeferredItem> {
        let mut gone = Vec::new();
        while self.sealed.front().is_some_and(|b| b.marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)) {
            gone.extend(self.sealed.pop_front().unwrap().dead);
        }
        gone
    }

    /// Close the open batch (at most every `SEAL_EVERY`, or `seal_now`) with a fence value on each of `queues`,
    /// and take out the batches every queue is past: the items to release now.
    unsafe fn drain(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> Vec<DeferredItem> {
        if self.needs_seal(seal_now) {
            let mut marks = Vec::with_capacity(queues.len());
            let mut ok = true;
            for q in queues {
                let key = q.as_raw() as usize;
                if !self.fences.contains_key(&key) {
                    let mut dev: Option<ID3D12Device> = None;
                    let made = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| super::real::create_raw_fence(&d).ok());
                    let Some(f) = made else {
                        ok = false;
                        break;
                    };
                    self.fences.insert(key, (f, 0));
                }
                let (fence, value) = self.fences.get_mut(&key).unwrap();
                if q.Signal(&*fence, *value + 1).is_err() {
                    ok = false;
                    break;
                }
                *value += 1;
                marks.push((fence.clone(), *value));
            }
            // (no fence or no signal: the device is gone – the batch stays open, and held)
            if ok {
                self.last_seal = Some(Instant::now());
                let dead = std::mem::take(&mut self.open);
                self.sealed.push_back(Batch { dead, marks });
            }
        }
        self.reap()
    }
}

static HELD: Mutex<Option<Held>> = Mutex::new(None);

/// Whether last releases are held (`XCL_LATE_RELEASE=0` turns it off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_LATE_RELEASE").is_ok_and(|v| v.trim() == "0"))
}

/// how long a last `Release` itself waits for queues with nothing to do to say so
const IDLE_WAIT: Duration = Duration::from_micros(300);

fn ensure_thread() {
    // A thread of its own lets the held ones go: the title may release a resource and then
    // wait for its memory without another submit (an imported heap pins the pages under its
    // resources – unmapping them stood still while one was held "until the next submit").
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-late-release".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            unsafe { drain_known(false) };
        });
    });
}

unsafe fn enqueue(item: DeferredItem) {
    let busy = {
        let mut g = HELD.lock().unwrap();
        let h = g.get_or_insert_with(Held::default);
        let busy = !h.sealed.is_empty();
        h.open.push(item);
        busy
    };
    if busy {
        // A sealed batch is already in flight on the GPU: let the background thread batch
        // this release into the next SEAL_EVERY signal instead of issuing a per-release
        // Signal on every queue and spinning IDLE_WAIT on the caller's thread.
        ensure_thread();
        return;
    }
    let start = Instant::now();
    drain_known(true);
    while start.elapsed() < IDLE_WAIT && HELD.lock().unwrap().as_ref().is_some_and(|h| !h.is_empty()) {
        std::hint::spin_loop();
        drain_known(false);
    }
    if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
        return;
    }
    ensure_thread();
}

/// `res` (an adopted resource) holds one reference that is now this module's. With no work in
/// flight it goes before this returns (the queues pass a fence value at once): a title that
/// frees the memory under a resource right after its last `Release` finds it unpinned, as
/// before. Otherwise the thread here lets it go.
pub unsafe fn hold(res: usize) {
    enqueue(DeferredItem::Com(res));
}

/// Defer freeing a UMA block (`sparse::free` or `uma::Uma::free`) until every queue XCL knows
/// of has passed the work submitted before this call, so in-flight GPU lists finish before the
/// block is reused or decommitted.
pub(super) fn hold_uma(ptr: usize, sparse: bool) {
    if ptr == 0 {
        return;
    }
    unsafe { enqueue(DeferredItem::Uma { ptr, sparse }) };
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones.
pub unsafe fn drain_known(seal_now: bool) {
    let need_queues = {
        let g = HELD.lock().unwrap();
        let Some(h) = g.as_ref() else { return };
        if h.is_empty() {
            return;
        }
        h.needs_seal(seal_now)
    };
    let gone = if need_queues {
        let mut queues: Vec<ID3D12CommandQueue> = Vec::new();
        for raw in super::present::known_queues() {
            let raw = raw as *mut c_void;
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                queues.push(q.clone());
            }
        }
        queues.extend(super::sparse::queue());
        queues.extend(super::vram::queues());
        queues.extend(super::title_textures::upload_queues());
        match HELD.lock().unwrap().as_mut() {
            Some(h) => h.drain(&queues, seal_now),
            None => return,
        }
    } else {
        match HELD.lock().unwrap().as_mut() {
            Some(h) => h.reap(),
            None => return,
        }
    };
    // (outside the lock: a release may end other objects, whose releases come back here)
    for item in gone {
        match item {
            DeferredItem::Com(res) => {
                if super::vram::recycle_ready(res) {
                    continue;
                }
                let ptr = res as *mut c_void;
                let f: unsafe extern "system" fn(*mut c_void) -> u32 = if super::real::is_adopted(ptr) {
                    std::mem::transmute(super::real::orig_fn(ptr, 2))
                } else {
                    std::mem::transmute(*(*(ptr as *const *const usize)).add(2))
                };
                f(ptr);
            }
            DeferredItem::Uma { ptr, sparse } => {
                if sparse {
                    super::sparse::free(ptr);
                } else if let Some(u) = crate::gpu::uma::get() {
                    u.free(ptr);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    /// A held resource stays while a queue has not got past the work submitted before it was
    /// released, and goes once the queue has.
    #[test]
    fn a_held_resource_goes_when_the_queue_is_past_its_batch() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            // the queue stands still, as with work in flight: it waits for a fence nobody has signalled
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let mut res: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let raw = res.unwrap().into_raw() as usize;

            let mut held = Held::default();
            held.open.push(DeferredItem::Com(raw));
            let queues = [queue.clone()];
            assert!(held.drain(&queues, false).is_empty(), "the queue has not passed the batch's fence value");
            std::thread::sleep(Duration::from_millis(100));
            assert!(held.drain(&queues, false).is_empty(), "still in flight: still held");
            assert!(!held.is_empty());

            gate.Signal(1).unwrap();
            let start = Instant::now();
            let mut gone = Vec::new();
            while gone.is_empty() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
                gone = held.drain(&queues, false);
            }
            assert_eq!(gone, vec![DeferredItem::Com(raw)], "released once the queue is past the batch");
            assert!(held.is_empty());
            drop(ID3D12Resource::from_raw(raw as *mut c_void));
        }
    }

    /// With no queue at all nothing can be in flight: the next drain lets the batch go.
    #[test]
    fn without_queues_a_batch_goes_at_once() {
        let mut held = Held::default();
        held.open.push(DeferredItem::Com(0x1000));
        assert_eq!(unsafe { held.drain(&[], false) }, vec![DeferredItem::Com(0x1000)]);
        assert!(held.is_empty());
    }

    /// `reap()` only drains completed sealed batches and never prematurely releases `open` items.
    #[test]
    fn reap_keeps_open_items_and_respects_seal_interval() {
        let mut held = Held::default();
        held.last_seal = Some(Instant::now());
        held.open.push(DeferredItem::Com(0x2000));
        assert!(!held.needs_seal(false), "within SEAL_EVERY: batching defers sealing");
        assert!(held.needs_seal(true), "explicit seal_now seals immediately");
        assert!(unsafe { held.reap() }.is_empty());
        assert_eq!(held.open, vec![DeferredItem::Com(0x2000)]);
    }

    /// Deferred UMA blocks retire in FIFO order alongside COM releases and wait for active GPU queues.
    #[test]
    fn hold_uma_preserves_order_and_frees_on_idle_drain() {
        let mut held = Held::default();
        held.open.push(DeferredItem::Com(0x1000));
        held.open.push(DeferredItem::Uma { ptr: 0x2000_0000, sparse: false });
        held.open.push(DeferredItem::Uma { ptr: 0x3000_0000, sparse: true });
        assert_eq!(
            unsafe { held.drain(&[], false) },
            vec![
                DeferredItem::Com(0x1000),
                DeferredItem::Uma { ptr: 0x2000_0000, sparse: false },
                DeferredItem::Uma { ptr: 0x3000_0000, sparse: true },
            ]
        );
        assert!(held.is_empty());

        // Real UMA block deferred through `hold_uma` is returned once no GPU queues are busy.
        let Some(u) = crate::gpu::uma::get() else { return };
        let before = u.used();
        let ptr = u.alloc(0x1_0000, 0x1_0000).expect("UMA alloc") as usize;
        assert!(u.used() >= before + 0x1_0000);
        hold_uma(ptr, false);
        // With no active queues in unit tests, `hold_uma` drains synchronously and frees the block.
        assert_eq!(u.used(), before, "deferred UMA block should be freed once queues are idle");
    }
}
