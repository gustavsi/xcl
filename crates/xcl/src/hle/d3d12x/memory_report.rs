//! Where XCL's memory goes: the parts it knows of, against what Windows and DXGI report (#2688,
//! workstream 1). One snapshot holds the process's commit, the adapter's local (video) and
//! non-local (system) memory in use against their budgets, and XCL's owners of each. What the
//! owners don't cover is shown as its own part: the title's own resources, the driver, and
//! whatever XCL doesn't count yet.

/// Which memory a part takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// the adapter's own memory (DXGI local segment)
    Video,
    /// system memory the GPU reads (DXGI non-local segment: upload and readback heaps)
    NonLocal,
    /// the process's commit
    Commit,
}

/// One owner's bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub owner: &'static str,
    pub kind: Kind,
    pub bytes: u64,
}

/// What DXGI reports for one segment group.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Segment {
    pub usage: u64,
    pub budget: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// the process's committed bytes (`PrivateUsage`; None where it reads 0: Wine leaves it so)
    pub commit: Option<u64>,
    /// None: no device or DXGI did not answer
    pub local: Option<Segment>,
    pub non_local: Option<Segment>,
    pub parts: Vec<Part>,
}

impl Snapshot {
    /// The bytes of the parts of `kind`.
    pub fn counted(&self, kind: Kind) -> u64 {
        self.parts.iter().filter(|p| p.kind == kind).map(|p| p.bytes).sum()
    }

    /// What the total of `kind` holds besides the parts XCL counts (negative when the parts are
    /// more than the total: a part counted on the wrong side, or memory not made yet).
    pub fn other(&self, kind: Kind) -> Option<i64> {
        let total = match kind {
            Kind::Video => self.local?.usage,
            Kind::NonLocal => self.non_local?.usage,
            Kind::Commit => self.commit?,
        };
        Some(total as i64 - self.counted(kind) as i64)
    }

    /// One log line: each total, its budget, its parts and what is left (MB).
    pub fn line(&self) -> String {
        let mb = |b: u64| b >> 20;
        let smb = |b: i64| b >> 20;
        let parts = |kind: Kind| {
            let mut s: Vec<String> = self.parts.iter().filter(|p| p.kind == kind).map(|p| format!("{} {}", p.owner, mb(p.bytes))).collect();
            if let Some(o) = self.other(kind) {
                s.push(format!("other {}", smb(o)));
            }
            s.join(", ")
        };
        let segment = |name: &str, seg: Option<Segment>, kind: Kind| match seg {
            Some(g) => format!("{name} {} of a {} budget ({})", mb(g.usage), mb(g.budget), parts(kind)),
            None => format!("{name} unknown ({})", parts(kind)),
        };
        format!(
            // (not "memory: N MB committed": the line `scripts/smoke.ps1` reads stays as it is)
            "memory by owner (MB): commit {} ({}); {}; {}",
            self.commit.map_or("unknown".into(), |c| mb(c).to_string()),
            parts(Kind::Commit),
            segment("video memory", self.local, Kind::Video),
            segment("non-local", self.non_local, Kind::NonLocal)
        )
    }
}

/// The process's committed bytes, or None where `PrivateUsage` reads 0 (Wine leaves it so; #1097
/// and #2362 read the commit there another way).
fn process_commit() -> Option<u64> {
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
        Some(c.PrivateUsage as u64).filter(|&n| n != 0)
    }
}

/// What DXGI reports for the device's adapter: (local, non-local), or None without a device.
fn video_memory() -> Option<(Segment, Segment)> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::ID3D12Device;
    use windows::Win32::Graphics::Dxgi::*;
    let raw = super::real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).ok()?;
        let query = |group| {
            let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter.QueryVideoMemoryInfo(0, group, &mut i).ok().map(|_| Segment { usage: i.CurrentUsage, budget: i.Budget })
        };
        Some((query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?, query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?))
    }
}

/// The commit Windows charges for the process's D3D12 heaps: what DXGI counts in both segments,
/// less the coherent window D3D12 imported (`imported`: CPU memory, already committed as such);
/// 0 when unknown.
fn heap_charge(local: Option<Segment>, non_local: Option<Segment>, imported: u64) -> u64 {
    match (local, non_local) {
        (Some(l), Some(n)) => (l.usage + n.usage).saturating_sub(imported),
        _ => 0,
    }
}

/// Where DXGI counts memory the GPU reads from system memory: the non-local segment, or on an
/// adapter with one segment (no non-local budget: an integrated GPU) the local one.
fn uma_segment(non_local: Option<Segment>) -> Kind {
    if non_local.is_some_and(|n| n.budget > 0) {
        Kind::NonLocal
    } else {
        Kind::Video
    }
}

/// Now.
pub fn snapshot() -> Snapshot {
    let (pages, page_pools) = super::vram::page_bytes();
    let (arenas, _) = super::vram::arena_bytes();
    let (pools, mapped) = super::sparse::pool_bytes();
    let (_, committed_textures) = super::title_textures::committed_bytes();
    let (local, non_local) = video_memory().map_or((None, None), |(l, n)| (Some(l), Some(n)));
    let uma = crate::gpu::uma::get().map_or(0, |u| u.size() as u64);
    // (what of it D3D12 has imported, which DXGI counts: none if the import failed)
    let imported = super::real::uma_imported_bytes();
    let b = |n: usize| n as u64;
    let parts = vec![
        // the title's memory in the coherent window: committed once, the GPU reads it there
        Part { owner: "UMA window", kind: Kind::Commit, bytes: uma },
        // a title page in a sparse window is committed on the CPU as well as mapped to a tile
        Part { owner: "sparse CPU copy", kind: Kind::Commit, bytes: b(super::sparse::committed_bytes()) },
        // Windows charges the process's D3D12 heaps to its commit as well, local and non-local
        // alike (1 GiB heap: +1 GiB of commit, on an RTX 4070 Ti and an RDNA2 iGPU) – all DXGI
        // counts but the coherent window, which is CPU memory imported and committed once
        Part { owner: "D3D12 heaps", kind: Kind::Commit, bytes: heap_charge(local, non_local, imported) },
        // the console shaders pipelines not built yet keep (`real::intern_blob`)
        Part { owner: "pipeline shader copies", kind: Kind::Commit, bytes: b(super::real::pipeline_blob_bytes().1) },
        Part { owner: "target pages", kind: Kind::Video, bytes: b(pages) },
        Part { owner: "target pools free", kind: Kind::Video, bytes: b(page_pools.saturating_sub(pages)) },
        Part { owner: "arenas", kind: Kind::Video, bytes: b(arenas) },
        Part { owner: "sparse tiles", kind: Kind::Video, bytes: b(mapped) },
        Part { owner: "sparse pools free", kind: Kind::Video, bytes: b(pools.saturating_sub(mapped)) },
        Part { owner: "sparse pools retired", kind: Kind::Video, bytes: b(super::sparse::retired_bytes()) },
        Part { owner: "committed textures", kind: Kind::Video, bytes: committed_textures },
        // the GPU reads the coherent window (imported, `OpenExistingHeapFromAddress`) as system
        // memory: DXGI counts it in the non-local segment of a card (an RTX 4070 Ti and an
        // RX 7800 XT), and in the local one of an integrated GPU, whose only segment that is
        // (no non-local budget)
        Part { owner: "UMA window", kind: uma_segment(non_local), bytes: imported },
    ];
    Snapshot { commit: process_commit(), local, non_local, parts }
}

/// The process's address space, walked (`VirtualQuery`): its committed private memory outside
/// the unified-memory block by allocation, the largest ones, and the rest by size; mapped and
/// image memory. What `snapshot`'s "other" holds, for finding its owner (the title's heaps, the
/// driver's, XCL's).
pub fn walk_line() -> String {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE};
    let u = crate::gpu::uma::get();
    let t0 = std::time::Instant::now();
    let (mut uma, mut mapped, mut image, mut regions) = (0u64, 0u64, 0u64, 0usize);
    let mut by_base: std::collections::HashMap<usize, (u64, u32, u32)> = std::collections::HashMap::new();
    let mut at = 0x1_0000usize;
    while at < 0x7FFF_FFFE_0000 {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const _, &mut m, std::mem::size_of_val(&m)) } == 0 {
            break;
        }
        regions += 1;
        let (base, size) = (m.BaseAddress as usize, m.RegionSize);
        if m.State == MEM_COMMIT {
            let n = size as u64;
            if u.is_some_and(|u| u.contains(base)) {
                uma += n;
            } else if m.Type == MEM_PRIVATE {
                let e = by_base.entry(m.AllocationBase as usize).or_insert((0, 0, m.Protect));
                e.0 += n;
                e.1 += 1;
            } else if m.Type == MEM_MAPPED {
                mapped += n;
            } else if m.Type == MEM_IMAGE {
                image += n;
            }
        }
        match base.checked_add(size) {
            Some(next) if next > at => at = next,
            _ => break,
        }
    }
    let mut all: Vec<(usize, u64, u32, u32)> = by_base.into_iter().map(|(b, (n, r, p))| (b, n, r, p)).collect();
    all.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let private: u64 = all.iter().map(|a| a.1).sum();
    let mb = |b: u64| b >> 20;
    let top: Vec<String> = all.iter().take(8).map(|&(b, n, r, p)| format!("{b:#x} {} ({r} regions, protect {p:#x})", mb(n))).collect();
    // the rest by size: (< 1 MB, < 16 MB, < 256 MB, larger) → (allocations, MB)
    let mut sizes = [(0usize, 0u64); 4];
    for &(_, n, _, _) in all.iter().skip(8) {
        let k = match n {
            n if n < 1 << 20 => 0,
            n if n < 16 << 20 => 1,
            n if n < 256 << 20 => 2,
            _ => 3,
        };
        sizes[k].0 += 1;
        sizes[k].1 += n;
    }
    format!(
        "memory walk (MB, {} regions, {} ms): unified memory {}; private elsewhere {} in {} allocations – largest {}; the rest under 1 MB {} in {}, under 16 MB {} in {}, under 256 MB {} in {}, larger {} in {}; mapped {}, images {}",
        regions,
        t0.elapsed().as_millis(),
        mb(uma),
        mb(private),
        all.len(),
        top.join(", "),
        mb(sizes[0].1),
        sizes[0].0,
        mb(sizes[1].1),
        sizes[1].0,
        mb(sizes[2].1),
        sizes[2].0,
        mb(sizes[3].1),
        sizes[3].0,
        mb(mapped),
        mb(image)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rest of each total is its total less its parts, a part is counted on its side only,
    /// and the line shows every part, each total against its budget, and the rest.
    #[test]
    fn the_rest_is_the_total_less_the_parts() {
        const M: u64 = 1 << 20;
        let s = Snapshot {
            commit: Some(10_000 * M),
            local: Some(Segment { usage: 9_000 * M, budget: 11_000 * M }),
            non_local: Some(Segment { usage: 300 * M, budget: 16_000 * M }),
            parts: vec![
                Part { owner: "UMA window", kind: Kind::Commit, bytes: 4_096 * M },
                Part { owner: "target pages", kind: Kind::Video, bytes: 5_000 * M },
                Part { owner: "arenas", kind: Kind::Video, bytes: 1_000 * M },
                Part { owner: "upload staging", kind: Kind::NonLocal, bytes: 400 * M },
            ],
        };
        assert_eq!(s.counted(Kind::Video), 6_000 * M);
        assert_eq!(s.other(Kind::Video), Some((3_000 * M) as i64));
        assert_eq!(s.other(Kind::Commit), Some((5_904 * M) as i64));
        // more counted than DXGI reports: a negative rest, not a wrap
        assert_eq!(s.other(Kind::NonLocal), Some(-((100 * M) as i64)));
        assert_eq!(
            s.line(),
            "memory by owner (MB): commit 10000 (UMA window 4096, other 5904); video memory 9000 of a 11000 budget (target pages 5000, arenas 1000, other 3000); non-local 300 of a 16000 budget (upload staging 400, other -100)"
        );
        // the coherent window: non-local on a card, local on a GPU of one segment
        assert_eq!(uma_segment(Some(Segment { usage: 0, budget: 16_000 * M })), Kind::NonLocal);
        assert_eq!(uma_segment(Some(Segment { usage: 0, budget: 0 })), Kind::Video);
        assert_eq!(uma_segment(None), Kind::Video);
        // the heaps' charge to the commit: both segments less the coherent window
        let (l, n) = (Some(Segment { usage: 9_000 * M, budget: 1 }), Some(Segment { usage: 4_500 * M, budget: 1 }));
        assert_eq!(heap_charge(l, n, 4_096 * M), 9_404 * M);
        assert_eq!(heap_charge(l, None, 4_096 * M), 0);
        assert_eq!(heap_charge(Some(Segment { usage: 100 * M, budget: 1 }), n, 8_000 * M), 0, "never below 0");
        // a window whose import failed is not in DXGI's usage: nothing of it is taken off
        assert_eq!(heap_charge(Some(Segment { usage: 3_000 * M, budget: 1 }), Some(Segment { usage: 0, budget: 1 }), 0), 3_000 * M);
        let unknown = Snapshot { commit: None, local: None, non_local: None, ..s };
        assert_eq!(unknown.other(Kind::Video), None);
        // a commit that reads 0 (Wine) is unknown, not 0 with a negative rest
        assert_eq!(unknown.other(Kind::Commit), None);
        assert!(unknown.line().starts_with("memory by owner (MB): commit unknown (UMA window 4096)"), "{}", unknown.line());
        assert!(unknown.line().contains("video memory unknown (target pages 5000, arenas 1000)"), "{}", unknown.line());
    }

    /// On a device the snapshot has DXGI's totals with a budget, the process's commit, and every
    /// owner.
    #[test]
    fn a_snapshot_on_the_device_has_its_budget_and_owners() {
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let s = snapshot();
        // (an integrated GPU has one segment: its non-local budget is 0)
        assert!(s.non_local.is_some(), "{}", s.line());
        assert!(s.local.expect("local video memory").budget > 0, "{}", s.line());
        // (`PrivateUsage` reads 0 under Wine: the commit is unknown there, and the line says so)
        match s.commit {
            Some(c) => assert!(c > 0),
            None => assert!(s.line().contains("commit unknown"), "{}", s.line()),
        }
        for owner in ["UMA window", "sparse CPU copy", "D3D12 heaps", "pipeline shader copies", "target pages", "target pools free", "arenas", "sparse tiles", "sparse pools free", "sparse pools retired", "committed textures"] {
            assert!(s.parts.iter().any(|p| p.owner == owner), "{owner} missing: {}", s.line());
        }
    }
}
