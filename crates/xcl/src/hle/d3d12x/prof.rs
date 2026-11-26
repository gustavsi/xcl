//! Where the title's threads spend their time in XCL: a `profile (2 s)` line beside the
//! `activity (2 s)` one (`XCL_ACTIVITY`). A console title's calls into the graphics driver take
//! next to no time; here a call can wait for a pipeline still being built, for XCL's uploads and
//! tile mappings before a submission, or for the GPU. Each kind of wait is timed where it happens
//! (`span`), summed over all the threads, so a frame rate near 0 shows what holds it: the line
//! gives each kind's time in the 2 s (over 2 000 ms: several threads at once), its calls and the
//! longest one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Kind {
    /// a draw waiting for its graphics pipeline (`pso_async::resolve_wait_in`)
    GraphicsPipelineWait,
    /// a dispatch or a submission waiting for a compute pipeline, its build on the title's
    /// thread included (`pso_async`)
    ComputePipelineWait,
    /// `ExecuteCommandLists` / `KickoffX`: XCL's uploads and tile mappings, then the driver's
    Submit,
    /// `CreatePlacedResourceX`: the PC's reserved resource and its tile mappings
    Placement,
    /// `WaitFrameEventX`: the title waiting for its frame pacing (the GPU, a flip)
    FrameEvent,
    /// `PresentX`
    Present,
    /// the driver compiling a pipeline, on any thread (workers too)
    DriverCompile,
    /// in a submit: the write-watch look for the pages the CPU wrote (`title_textures::refresh_once`)
    SubmitWatch,
    /// in a submit: those pages copied to the sparse windows' video memory (`sparse::before_execute`)
    SubmitSparse,
    /// in a submit: textures in title memory uploaded again (`title_textures::upload_all`)
    SubmitTextures,
    /// in a submit: the title's lists expanded into their PC segments (put-off dispatches recorded)
    SubmitRecord,
    /// in a submit: the PC `ExecuteCommandLists` (the driver)
    SubmitDriver,
    /// in a submit: after the driver's call (scratch copies, Write64 signals, readbacks, CPU writes)
    SubmitAfter,
}

const KINDS: usize = 13;
const NAMES: [&str; KINDS] = [
    "graphics pipeline waits",
    "compute pipeline waits",
    "submits",
    "placements",
    "frame-event waits",
    "presents",
    "driver compiles (all threads)",
    "[submit: write-watch",
    "sparse copies",
    "texture uploads",
    "list segments",
    "driver",
    "after]",
];

static NS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static CALLS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
/// the longest single one since the last line
static LONGEST: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// Adds `d` of `kind`.
pub fn add(kind: Kind, d: Duration) {
    let k = kind as usize;
    let ns = d.as_nanos().min(u64::MAX as u128) as u64;
    NS[k].fetch_add(ns, Ordering::Relaxed);
    CALLS[k].fetch_add(1, Ordering::Relaxed);
    LONGEST[k].fetch_max(ns, Ordering::Relaxed);
}

/// Times from now to its drop as `kind`.
pub struct Span(Kind, Instant);

pub fn span(kind: Kind) -> Span {
    Span(kind, Instant::now())
}

impl Drop for Span {
    fn drop(&mut self) {
        add(self.0, self.1.elapsed());
    }
}

/// Totals so far: (ns, calls) of each kind.
#[derive(Clone, Copy)]
pub struct Totals([(u64, u64); KINDS]);

pub fn totals() -> Totals {
    Totals(std::array::from_fn(|k| (NS[k].load(Ordering::Relaxed), CALLS[k].load(Ordering::Relaxed))))
}

/// The line for what was added between `before` and `now` (the longest ones start again).
pub fn line(before: Totals, now: Totals, presents: u64, secs: f64) -> String {
    let mut parts = Vec::new();
    for k in 0..KINDS {
        let ns = now.0[k].0.saturating_sub(before.0[k].0);
        let calls = now.0[k].1.saturating_sub(before.0[k].1);
        let longest = LONGEST[k].swap(0, Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        parts.push(format!("{} {} ms ({calls}, longest {} ms)", NAMES[k], ns / 1_000_000, longest / 1_000_000));
    }
    let frame = if presents > 0 { format!("{:.0} ms a frame", secs * 1000.0 / presents as f64) } else { "no frame".to_string() };
    let (_, failed, queued) = super::pso_async::counts();
    let (running, done) = crate::gpu::recomp::dxc::compile_counts();
    format!(
        "profile ({secs:.0} s): {frame}; {}; pipelines queued {queued}, failed {failed}; DXC {running} running, {done} done",
        if parts.is_empty() { "no waits".to_string() } else { parts.join(", ") }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span adds its time and a call to its kind; the line gives the difference since the
    /// totals before, and leaves out kinds with no call.
    #[test]
    fn spans_add_up_by_kind_in_the_line() {
        let before = totals();
        add(Kind::Placement, Duration::from_millis(30));
        add(Kind::Placement, Duration::from_millis(20));
        {
            let _s = span(Kind::FrameEvent);
        }
        let now = totals();
        let k = Kind::Placement as usize;
        assert!(now.0[k].0 - before.0[k].0 >= 50_000_000);
        assert!(now.0[k].1 - before.0[k].1 >= 2);
        let l = line(before, now, 4, 2.0);
        assert!(l.contains("500 ms a frame"), "{l}");
        assert!(l.contains("placements "), "{l}");
        assert!(l.contains("frame-event waits "), "{l}");
    }
}
