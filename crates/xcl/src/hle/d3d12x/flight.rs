//! A flight recorder of XCL's GPU side, on with `XCL_ACTIVITY` (or `XCL_FLIGHT=1`): what one run
//! of a title has to tell when the device is lost or the title stops, without another run.
//!
//! - **Queues.** Every fence XCL signals on a queue after its work (`signaled`: the title's
//!   submits through `alloc_pool`, the sparse windows' queue and its read-back queue, the video
//!   memory pages' queue, texture uploads, the AMD keep-awake queue) is polled every 50 ms: the
//!   value submitted, the value done and when it last moved. A queue with work that has not moved
//!   for a second is logged at once (`flight: … has not finished …`), and on a device removal or
//!   a crash all queues are listed, the one that stopped first first. A device removed with no
//!   command buffer in flight (what AMD's crash analysis showed for a retail title) leaves only
//!   the queues' own operations – tile mappings, waits – and this says which queue stopped.
//! - **Counters**, a `flight (2 s)` line beside `profile (2 s)`: tile mappings by caller (calls,
//!   tiles, time) and resources made by kind (calls, time), and the adapter's video memory
//!   against its budget (`OVER` when past it) with the system's free memory and commit.
//! - **Events**, the last 2 048 (submits, tile mappings, big resources), printed with the queues.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("XCL_FLIGHT").map_or(std::env::var_os("XCL_ACTIVITY").is_some(), |v| v.trim() != "0")
    })
}

fn start() -> Instant {
    static T0: OnceLock<Instant> = OnceLock::new();
    *T0.get_or_init(Instant::now)
}

fn now_ms() -> u64 {
    start().elapsed().as_millis() as u64
}

fn thread_id() -> u32 {
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

// ---- events -------------------------------------------------------------------------------

#[derive(Clone)]
struct Event {
    ms: u64,
    thread: u32,
    what: &'static str,
    a: u64,
    b: u64,
    c: u64,
}

const RING: usize = 2048;
static EVENTS: Mutex<VecDeque<Event>> = Mutex::new(VecDeque::new());

/// Notes `what` with three numbers (their meaning is the caller's: see `dump`'s lines).
pub fn event(what: &'static str, a: u64, b: u64, c: u64) {
    if !on() {
        return;
    }
    let e = Event { ms: now_ms(), thread: thread_id(), what, a, b, c };
    let mut g = EVENTS.lock().unwrap();
    if g.len() >= RING {
        g.pop_front();
    }
    g.push_back(e);
}

// ---- queues -------------------------------------------------------------------------------

struct Tracked {
    name: &'static str,
    queue: usize,
    fence: ID3D12Fence,
    submitted: u64,
    completed: u64,
    last_submit_ms: u64,
    last_progress_ms: u64,
    stalled: bool,
}
// (the fence is used under the lock and by the poll thread only)
unsafe impl Send for Tracked {}

static QUEUES: Mutex<Option<HashMap<usize, Tracked>>> = Mutex::new(None);

/// `queue` was given `Signal(fence, value)` after its work (`name`: whose queue it is).
pub fn signaled(name: &'static str, queue: usize, fence: &ID3D12Fence, value: u64) {
    if !on() {
        return;
    }
    ensure_poller();
    let now = now_ms();
    let key = windows::core::Interface::as_raw(fence) as usize;
    {
        let mut g = QUEUES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let t = m.entry(key).or_insert_with(|| Tracked {
            name,
            queue,
            fence: fence.clone(),
            submitted: 0,
            completed: 0,
            last_submit_ms: now,
            last_progress_ms: now,
            stalled: false,
        });
        if value > t.submitted {
            // a queue that had nothing to do starts its wait now
            if t.submitted <= t.completed {
                t.last_progress_ms = now;
            }
            t.submitted = value;
        }
        t.last_submit_ms = now;
    }
    event(name, queue as u64, value, 0);
}

/// How long a queue with work may stay without finishing any before it is logged.
const STALL_MS: u64 = 1000;

fn ensure_poller() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-flight".into()).spawn(|| loop {
            std::thread::sleep(Duration::from_millis(50));
            poll();
        });
    });
}

fn poll() {
    let now = now_ms();
    // (logged after the lock: submits note their fences under it)
    let mut lines = Vec::new();
    // how long the longest stalled queue has not moved (ms)
    let mut longest = 0u64;
    {
    let mut g = QUEUES.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    for t in m.values_mut() {
        let c = unsafe { t.fence.GetCompletedValue() };
        if c == u64::MAX {
            // the device is gone: what the queues held stays as it was for `dump`
            continue;
        }
        if t.submitted > c {
            longest = longest.max(now.saturating_sub(t.last_progress_ms));
        }
        if c > t.completed {
            t.completed = c;
            t.last_progress_ms = now;
            if t.stalled {
                t.stalled = false;
                lines.push((false, format!("flight: {} (queue {:#x}) moves again: done {} of {}", t.name, t.queue, c, t.submitted)));
            }
        }
        if t.submitted > t.completed && !t.stalled && now.saturating_sub(t.last_progress_ms) >= STALL_MS {
            t.stalled = true;
            lines.push((
                true,
                format!(
                    "flight: {} (queue {:#x}) has not finished value {} for {} ms (submitted up to {}, last submit {} ms ago)",
                    t.name,
                    t.queue,
                    t.completed + 1,
                    now - t.last_progress_ms,
                    t.submitted,
                    now.saturating_sub(t.last_submit_ms)
                ),
            ));
        }
    }
    }
    for (warn, l) in lines {
        if warn {
            crate::xlog_warn!("{l}");
        } else {
            crate::xlog_info!("{l}");
        }
    }
    // A GPU that has not moved for 3 s: which dispatches started and did not finish (the markers
    // are CPU memory, read without the driver), and at 8 s where every thread is. Here and not
    // in the activity thread: in a retail title's hang its DXGI queries blocked in the driver,
    // and the report never came.
    if longest >= 3_000 {
        super::culprit::on_watchdog_release();
    }
    static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if longest >= 8_000 && std::env::var("XCL_STALL_DUMP").is_ok_and(|v| v.trim() == "1") && !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::xlog_info!("flight: the GPU has not moved for {longest} ms: thread dump");
        crate::eh::dump_threads();
    }
}

// ---- counters -----------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Count {
    calls: u64,
    units: u64,
    ns: u64,
    longest_ns: u64,
}

static TILE_MAPS: Mutex<Vec<(&'static str, Count)>> = Mutex::new(Vec::new());
static MADE: Mutex<Vec<(&'static str, Count)>> = Mutex::new(Vec::new());

fn add(list: &Mutex<Vec<(&'static str, Count)>>, key: &'static str, units: u64, d: Duration) {
    let ns = d.as_nanos().min(u64::MAX as u128) as u64;
    let mut g = list.lock().unwrap();
    let c = match g.iter_mut().find(|(k, _)| *k == key) {
        Some((_, c)) => c,
        None => {
            g.push((key, Count::default()));
            &mut g.last_mut().unwrap().1
        }
    };
    c.calls += 1;
    c.units += units;
    c.ns += ns;
    c.longest_ns = c.longest_ns.max(ns);
}

/// An `UpdateTileMappings` of `caller` on `queue`: `tiles` tiles, the call took `d`.
pub fn tile_map(caller: &'static str, queue: usize, tiles: u64, d: Duration) {
    if !on() {
        return;
    }
    add(&TILE_MAPS, caller, tiles, d);
    event("tile map", queue as u64, tiles, d.as_micros() as u64);
}

/// A resource or heap made through `kind` (a creation call, `bytes` when known), in `d`.
pub fn made(kind: &'static str, bytes: u64, d: Duration) {
    if !on() {
        return;
    }
    add(&MADE, kind, bytes, d);
    if bytes >= 8 << 20 || d >= Duration::from_millis(20) {
        event(kind, bytes, d.as_micros() as u64, 0);
    }
}

fn take(list: &Mutex<Vec<(&'static str, Count)>>) -> Vec<(&'static str, Count)> {
    std::mem::take(&mut *list.lock().unwrap())
}

fn mb(b: u64) -> u64 {
    b >> 20
}

/// The adapter's video memory against its budget, and the system's memory.
fn budget_text() -> String {
    let mut s = String::new();
    if let Some((local, non_local)) = super::real::video_budget() {
        let part = |name: &str, i: &windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO| {
            format!(
                "{name} {} of {} MB{}",
                mb(i.CurrentUsage),
                mb(i.Budget),
                if i.Budget > 0 && i.CurrentUsage > i.Budget { " OVER" } else { "" }
            )
        };
        s = format!("video memory {}, {}", part("local", &local), part("non-local", &non_local));
    }
    unsafe {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 {
            s.push_str(&format!(
                "; system memory free {} of {} MB, commit free {} of {} MB",
                mb(m.ullAvailPhys),
                mb(m.ullTotalPhys),
                mb(m.ullAvailPageFile),
                mb(m.ullTotalPageFile)
            ));
        }
    }
    s
}

/// The `flight (2 s)` line: tile mappings and resources made since the last line, the budget.
pub fn line() -> String {
    let list = |v: Vec<(&'static str, Count)>, unit: &str| -> String {
        if v.is_empty() {
            return "none".into();
        }
        v.iter()
            .map(|(k, c)| {
                format!("{k} {} calls, {} {unit}, {} ms (longest {} ms)", c.calls, c.units, c.ns / 1_000_000, c.longest_ns / 1_000_000)
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let maps = list(take(&TILE_MAPS), "tiles");
    let made = take(&MADE);
    let made = if made.is_empty() {
        "none".to_string()
    } else {
        made.iter()
            .map(|(k, c)| format!("{k} {} ({} MB, {} ms, longest {} ms)", c.calls, mb(c.units), c.ns / 1_000_000, c.longest_ns / 1_000_000))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!("flight (2 s): tile mappings: {maps}; made: {made}; {}", budget_text())
}

// ---- the dump -----------------------------------------------------------------------------

/// Everything the recorder holds, into the log (once): on a device removal or a crash. A crash
/// leaves the video memory budget out: the title may have stopped inside the driver, holding its
/// locks, and asking DXGI then could hang the crash report.
pub fn dump(why: &str) {
    static DONE: AtomicBool = AtomicBool::new(false);
    if !on() || DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    let now = now_ms();
    // (a crash can come with these locks held: then without that part)
    if let Ok(g) = QUEUES.try_lock() {
        if let Some(m) = g.as_ref() {
            let mut v: Vec<&Tracked> = m.values().collect();
            // the queue with work that stopped first, first
            v.sort_by_key(|t| (t.submitted <= t.completed, t.last_progress_ms));
            crate::xlog_error!("flight: {why} at {now} ms – the queues XCL signals (the one that stopped first first):");
            for t in v {
                crate::xlog_error!(
                    "flight:   {} queue {:#x}: done {} of {} ({}), last moved {} ms ago, last submit {} ms ago",
                    t.name,
                    t.queue,
                    t.completed,
                    t.submitted,
                    if t.submitted > t.completed { "WORK PENDING" } else { "idle" },
                    now.saturating_sub(t.last_progress_ms),
                    now.saturating_sub(t.last_submit_ms)
                );
            }
        }
    }
    if let Ok(g) = EVENTS.try_lock() {
        let skip = g.len().saturating_sub(400);
        crate::xlog_error!("flight: the last {} events (ms, thread, what, numbers):", g.len() - skip);
        for e in g.iter().skip(skip) {
            crate::xlog_error!("flight:   {} t{} {} {:#x} {} {}", e.ms, e.thread, e.what, e.a, e.b, e.c);
        }
    }
    if why != "crash" {
        crate::xlog_error!("flight: {}", budget_text());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counters sum calls, units and time by key and start again after a line.
    #[test]
    fn counters_sum_by_key_and_start_again() {
        let l: Mutex<Vec<(&'static str, Count)>> = Mutex::new(Vec::new());
        add(&l, "vram", 10, Duration::from_millis(3));
        add(&l, "vram", 5, Duration::from_millis(7));
        add(&l, "sparse", 1, Duration::from_millis(1));
        let v = take(&l);
        let vram = v.iter().find(|(k, _)| *k == "vram").unwrap().1;
        assert_eq!((vram.calls, vram.units), (2, 15));
        assert_eq!(vram.longest_ns, 7_000_000);
        assert!(take(&l).is_empty());
    }
}
