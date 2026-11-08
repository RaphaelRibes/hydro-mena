//! End-to-end pipeline orchestration + benchmark harness.
//!
//! Every stage writes a CSV a plotting layer or a GUI can read directly:
//!
//! | file | content |
//! |---|---|
//! | `report.json` | all scalars + timings (single source of truth) |
//! | `similarity.csv` | S = abs(r), n x n (only with `--write-similarity`) |
//! | `adjacency.csv` | thresholded 0/1 graph on the kept nodes |
//! | `edges.csv` | `source,target,weight` for network plots |
//! | `nodes.csv` | per-OTU module, role, zi, Pi, degree, MM, strongest trait |
//! | `rmt_scan.csv` | one row per scanned threshold (chi2, dof, crit, KS) |
//! | `spectrum.csv` | eigenvalues + unfolded spacings of the kept graph |
//! | `degree_dist.csv` | degree distribution for the scale-free fit |
//! | `modules.csv` | module size + variance explained |
//! | `eigengenes.csv` | E_b(sample) |
//! | `fig5_r.csv`, `fig5_p.csv` | module x trait correlations and p-values |
use crate::eigengene::{eigengenes, trait_significance};
use crate::io;
use crate::linalg::{self, Mat};
use crate::modules::{analyze_modules_with, ModuleAnalysis};
use crate::network::{self, Topology};
use crate::rmt::{rmt_threshold, RmtResult};
use crate::traits::mantel_connectivity_vs_traits;
use anyhow::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Serialize)]
pub struct RunReport {
    pub n_otus: usize,
    pub n_samples: usize,
    pub n_traits: usize,
    pub gpu_used: bool,
    pub input: InputMeta,
    pub rmt: RmtReport,
    pub topology: TopoReport,
    pub modules: ModuleReport,
    pub eigengene_var_explained: Vec<f64>,
    pub fig5_r: Vec<Vec<f64>>, // modules x traits
    pub fig5_p: Vec<Vec<f64>>,
    pub fig5_q: Vec<Vec<f64>>,
    pub mantel: MantelReport,
    pub timing_s: Timing,
}

#[derive(Serialize)]
pub struct InputMeta {
    pub abundance: String,
    pub traits: String,
    pub seed: u64,
    pub permutations: usize,
    pub fixed_threshold: Option<f64>,
}

#[derive(Serialize)]
pub struct RmtReport {
    pub st: f64,
    pub converged: bool,
    /// no threshold passed the GOF; `st` is a diagnostic fallback
    pub fallback: bool,
    pub alpha: f64,
    pub n_points: usize,
    pub n_kept: usize,
    pub edges: usize,
    pub scan: Vec<crate::rmt::ScanPoint>,
}

#[derive(Serialize)]
pub struct TopoReport {
    pub n: usize,
    pub edges: usize,
    pub density: f64,
    pub avg_k: f64,
    pub gd: f64,
    pub connected_frac: f64,
    pub n_components: usize,
    pub giant_frac: f64,
    pub avg_cc: f64,
    pub modularity: f64,
    pub n_modules: usize,
    pub r2_powerlaw: f64,
    pub gamma: f64,
    pub r2_hierarchy: f64,
}

#[derive(Serialize)]
pub struct ModuleReport {
    pub modularity: f64,
    pub n_modules: usize,
    pub sizes: Vec<usize>,
    pub role_counts: std::collections::HashMap<String, usize>,
}

#[derive(Serialize)]
pub struct MantelReport {
    pub r: f64,
    pub p: f64,
    pub n_perm: usize,
    pub r_null_mean: f64,
}

#[derive(Serialize)]
pub struct Timing {
    pub standardize_s: f64,
    pub similarity_s: f64,
    pub rmt_s: f64,
    pub topology_s: f64,
    pub modules_s: f64,
    pub eigengene_s: f64,
    pub mantel_s: f64,
    pub export_s: f64,
    pub total_s: f64,
}

/// Everything the caller can tune, so `run_pipeline` keeps a stable arity.
#[derive(Clone, Debug)]
pub struct RunOptions {
    pub out_dir: PathBuf,
    pub use_gpu: bool,
    pub n_perm: usize,
    pub seed: u64,
    /// skip the RMT scan and use this similarity threshold
    pub fixed_st: Option<f64>,
    /// lowest similarity threshold the RMT scan considers
    pub rmt_start: f64,
    /// significance level of the Poisson GOF
    pub alpha: f64,
    /// write the full n x n similarity matrix (can be huge)
    pub write_similarity: bool,
    /// write the per-run CSVs (disabled for benchmarking)
    pub write_artifacts: bool,
    /// provenance labels recorded in report.json
    pub abundance_path: String,
    pub traits_path: String,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("out"),
            use_gpu: false,
            n_perm: 999,
            seed: 0,
            fixed_st: None,
            rmt_start: 0.30,
            alpha: 0.05,
            write_similarity: false,
            write_artifacts: true,
            abundance_path: String::new(),
            traits_path: String::new(),
        }
    }
}

pub struct RunArtifacts {
    pub report: RunReport,
    pub adjacency: Mat,
    pub kept_names: Vec<String>,
}

/// Similarity matrix on GPU when asked and available, otherwise rayon CPU.
/// The GPU result is *used*; the previous version discarded it and recomputed
/// the whole matrix on the CPU ("[gpu] similarity computed" followed by a
/// silent CPU pass), so `--gpu` reported `gpu_used=true` while burning both.
fn similarity(z: &Mat, use_gpu: bool) -> (Mat, bool) {
    #[cfg(feature = "gpu")]
    {
        if use_gpu {
            match crate::gpu::similarity_gpu(z) {
                Ok(s) => return (s, true),
                Err(e) => eprintln!("[gpu] similarity failed ({e}); falling back to CPU"),
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        if use_gpu {
            eprintln!("[gpu] binary built without the gpu feature; using CPU");
        }
    }
    (linalg::similarity_from_standardized(z), false)
}

pub fn run_pipeline(
    abundance: &Mat,
    otu_names: &[String],
    sample_names: &[String],
    traits: &Mat,
    trait_names: &[String],
    opts: &RunOptions,
) -> Result<RunArtifacts> {
    if opts.write_artifacts {
        std::fs::create_dir_all(&opts.out_dir)?;
    }
    let t0 = Instant::now();

    let t = Instant::now();
    let z = linalg::standardize_rows(abundance);
    let standardize_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (s, gpu_used) = similarity(&z, opts.use_gpu);
    let similarity_s = t.elapsed().as_secs_f64();
    if opts.write_artifacts && opts.write_similarity {
        io::write_csv_matrix(&opts.out_dir.join("similarity.csv"), &s, otu_names, otu_names, "OTU")?;
    }

    let t = Instant::now();
    let min_nodes = (abundance.nrows() / 10).max(10);
    let rmt = match opts.fixed_st {
        Some(st) => crate::rmt::at_threshold(&s, st),
        None => rmt_threshold(&s, opts.rmt_start, min_nodes, opts.alpha),
    };
    let rmt_s = t.elapsed().as_secs_f64();
    eprintln!(
        "[rmt] st = {:.3} converged={}{} ({} scan points, kept {}/{}, E={}, {:.2}s)",
        rmt.st,
        rmt.converged,
        if rmt.fallback { " FALLBACK" } else { "" },
        rmt.log.len(),
        rmt.kept.len(),
        abundance.nrows(),
        rmt.edges,
        rmt_s
    );

    // restrict to kept nodes
    let kept = &rmt.kept;
    let p = kept.len();
    let mut a = Mat::zeros((p, p));
    for (x, &i) in kept.iter().enumerate() {
        for (y, &j) in kept.iter().enumerate() {
            a[(x, y)] = rmt.adjacency[(i, j)];
        }
    }
    let kept_names: Vec<String> = kept.iter().map(|&i| otu_names[i].clone()).collect();
    let zk = Mat::from_shape_fn((p, abundance.ncols()), |(i, j)| z[(kept[i], j)]);

    let t = Instant::now();
    let (topo, partition) = network::topology_parts(&a);
    let topology_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mod_analysis: ModuleAnalysis = analyze_modules_with(
        &network::adjacency_list(&a),
        topo.modularity,
        partition,
    );
    let modules_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let tz = linalg::standardize_rows(&traits.t().to_owned()).t().to_owned();
    let eig = eigengenes(&zk, &mod_analysis.module_of, mod_analysis.n_modules, &tz);
    let gs = trait_significance(&zk, traits);
    let eigengene_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    // BH-FDR over the whole module x trait family: starring raw p-values across
    // ~1,480 tests (20 modules x 74 traits) is multiplicative cherry-picking.
    let p_flat: Vec<f64> = eig.trait_p.iter().copied().collect();
    let q_flat = crate::stats::bh_qvalues(&p_flat);
    let fig5_q: Vec<Vec<f64>> = {
        let cols = eig.trait_p.ncols();
        (0..eig.trait_p.nrows())
            .map(|b| (0..cols).map(|h| q_flat[b * cols + h]).collect())
            .collect()
    };
    let mantel = mantel_connectivity_vs_traits(&a, &zk, traits, opts.n_perm, opts.seed);
    let mantel_s = t.elapsed().as_secs_f64();

    let mut role_counts = std::collections::HashMap::new();
    for r in &mod_analysis.roles {
        *role_counts.entry(r.role.to_string()).or_insert(0) += 1;
    }

    let module_names: Vec<String> = (0..mod_analysis.n_modules)
        .map(|b| format!("module_{}", b + 1))
        .collect();
    let fig5_r: Vec<Vec<f64>> = (0..eig.trait_r.nrows())
        .map(|b| (0..eig.trait_r.ncols()).map(|h| eig.trait_r[(b, h)]).collect())
        .collect();
    let fig5_p: Vec<Vec<f64>> = (0..eig.trait_p.nrows())
        .map(|b| (0..eig.trait_p.ncols()).map(|h| eig.trait_p[(b, h)]).collect())
        .collect();

    let report = RunReport {
        n_otus: abundance.nrows(),
        n_samples: abundance.ncols(),
        n_traits: traits.ncols(),
        gpu_used,
        input: InputMeta {
            abundance: opts.abundance_path.clone(),
            traits: opts.traits_path.clone(),
            seed: opts.seed,
            permutations: opts.n_perm,
            fixed_threshold: opts.fixed_st,
        },
        rmt: RmtReport {
            st: rmt.st,
            converged: rmt.converged,
            fallback: rmt.fallback,
            alpha: opts.alpha,
            n_points: rmt.log.len(),
            n_kept: p,
            edges: rmt.edges,
            scan: rmt.log.clone(),
        },
        topology: TopoReport {
            n: topo.n,
            edges: topo.edges,
            density: topo.density,
            avg_k: topo.avg_k,
            gd: topo.gd,
            connected_frac: topo.connected_frac,
            n_components: topo.n_components,
            giant_frac: topo.giant_frac,
            avg_cc: topo.avg_cc,
            modularity: topo.modularity,
            n_modules: topo.n_modules,
            r2_powerlaw: topo.r2_powerlaw,
            gamma: topo.gamma,
            r2_hierarchy: topo.r2_hierarchy,
        },
        modules: ModuleReport {
            modularity: mod_analysis.modularity,
            n_modules: mod_analysis.n_modules,
            sizes: mod_analysis.members.iter().map(|m| m.len()).collect(),
            role_counts,
        },
        eigengene_var_explained: eig.var_explained.clone(),
        fig5_r,
        fig5_p,
        fig5_q,
        mantel: MantelReport {
            r: mantel.r,
            p: mantel.p,
            n_perm: mantel.n_perm,
            r_null_mean: mantel.r_null_mean,
        },
        timing_s: Timing {
            standardize_s,
            similarity_s,
            rmt_s,
            topology_s,
            modules_s,
            eigengene_s,
            mantel_s,
            export_s: 0.0,
            total_s: 0.0,
        },
    };

    if !opts.write_artifacts {
        let mut report = report;
        report.timing_s.total_s = t0.elapsed().as_secs_f64();
        return Ok(RunArtifacts { report, adjacency: a, kept_names });
    }

    let t = Instant::now();
    write_artifacts(
        &opts.out_dir,
        &a,
        &kept_names,
        &s,
        otu_names,
        sample_names,
        trait_names,
        &rmt,
        &topo,
        &mod_analysis,
        &eig,
        &gs,
        &module_names,
        opts.write_similarity,
    )?;
    let export_s = t.elapsed().as_secs_f64();

    let mut report = report;
    report.timing_s.export_s = export_s;
    report.timing_s.total_s = t0.elapsed().as_secs_f64();
    io::write_json(&opts.out_dir.join("report.json"), &report)?;

    Ok(RunArtifacts { report, adjacency: a, kept_names })
}

#[allow(clippy::too_many_arguments)]
fn write_artifacts(
    out: &Path,
    a: &Mat,
    names: &[String],
    s: &Mat,
    otu_names: &[String],
    sample_names: &[String],
    trait_names: &[String],
    rmt: &RmtResult,
    topo: &Topology,
    ma: &ModuleAnalysis,
    eig: &crate::eigengene::EigengeneResult,
    gs: &Mat,
    module_names: &[String],
    write_similarity: bool,
) -> Result<()> {
    if write_similarity {
        io::write_csv_matrix(&out.join("similarity.csv"), s, otu_names, otu_names, "OTU")?;
    }
    io::write_csv_matrix(&out.join("adjacency.csv"), a, names, names, "OTU")?;
    io::write_csv_matrix(&out.join("eigengenes.csv"), &eig.E, module_names, sample_names, "module")?;
    io::write_csv_matrix(&out.join("fig5_r.csv"), &eig.trait_r, module_names, trait_names, "module")?;
    io::write_csv_matrix(&out.join("fig5_p.csv"), &eig.trait_p, module_names, trait_names, "module")?;
    {
        let cols = eig.trait_p.ncols();
        let flat: Vec<f64> = eig.trait_p.iter().copied().collect();
        let q = crate::stats::bh_qvalues(&flat);
        let qm = Mat::from_shape_fn((eig.trait_p.nrows(), cols), |(b, h)| q[b * cols + h]);
        io::write_csv_matrix(&out.join("fig5_q.csv"), &qm, module_names, trait_names, "module")?;
    }

    // edges (weight = |r| when the similarity matrix is available)
    {
        let mut w = csv::Writer::from_path(out.join("edges.csv"))?;
        w.write_record(["source", "target", "module_source", "module_target"])?;
        let n = a.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)] > 0.0 {
                    w.write_record([
                        names[i].clone(),
                        names[j].clone(),
                        format!("{}", ma.module_of[i] + 1),
                        format!("{}", ma.module_of[j] + 1),
                    ])?;
                }
            }
        }
        w.flush()?;
    }

    // per-OTU table: everything a network viewer or a scatter plot needs
    {
        let n = a.nrows();
        let mut w = csv::Writer::from_path(out.join("nodes.csv"))?;
        w.write_record([
            "OTU", "module", "zi", "Pi", "degree", "role", "MM_eigengene", "gs_max", "gs_trait",
        ])?;
        for r in &ma.roles {
            let i = r.node;
            let (mut gs_max, mut gs_trait) = (f64::NAN, String::new());
            for h in 0..gs.ncols() {
                let v = gs[(i, h)];
                if gs_max.is_nan() || v > gs_max {
                    gs_max = v;
                    gs_trait = trait_names.get(h).cloned().unwrap_or_default();
                }
            }
            let _ = n;
            w.write_record([
                names[i].clone(),
                format!("{}", r.module + 1),
                format!("{:.4}", r.zi),
                format!("{:.4}", r.pi),
                format!("{:.0}", r.connectivity),
                r.role.to_string(),
                format!("{:.4}", eig.mmie[i]),
                format!("{:.4}", gs_max),
                gs_trait,
            ])?;
        }
        w.flush()?;
    }

    // RMT scan trace
    {
        let mut w = csv::Writer::from_path(out.join("rmt_scan.csv"))?;
        w.write_record(["st", "p", "edges", "chi2", "dof", "crit", "poisson", "ks"])?;
        for pt in &rmt.log {
            w.write_record([
                format!("{:.4}", pt.st),
                pt.p.to_string(),
                pt.edges.to_string(),
                format!("{:.4}", pt.chi2),
                pt.dof.to_string(),
                format!("{:.4}", pt.crit),
                pt.poisson.to_string(),
                format!("{:.4}", pt.ks),
            ])?;
        }
        w.flush()?;
    }

    // spectrum of the kept graph
    {
        let mut w = csv::Writer::from_path(out.join("spectrum.csv"))?;
        w.write_record(["index", "eigenvalue", "spacing"])?;
        for (i, lam) in rmt.eigenvalues.iter().enumerate() {
            w.write_record([
                i.to_string(),
                format!("{:.6}", lam),
                rmt.spacings.get(i).map(|v| format!("{v:.6}")).unwrap_or_default(),
            ])?;
        }
        w.flush()?;
    }

    // degree distribution for the scale-free fit
    {
        let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for r in &ma.roles {
            *counts.entry(r.connectivity as usize).or_insert(0) += 1;
        }
        let n = a.nrows().max(1) as f64;
        let mut w = csv::Writer::from_path(out.join("degree_dist.csv"))?;
        w.write_record(["k", "count", "pk"])?;
        for (k, c) in counts {
            w.write_record([k.to_string(), c.to_string(), format!("{:.6}", c as f64 / n)])?;
        }
        w.flush()?;
    }

    // module table
    {
        let mut w = csv::Writer::from_path(out.join("modules.csv"))?;
        w.write_record(["module", "size", "var_explained"])?;
        for b in 0..ma.n_modules {
            w.write_record([
                format!("{}", b + 1),
                ma.members[b].len().to_string(),
                format!("{:.4}", eig.var_explained.get(b).copied().unwrap_or(f64::NAN)),
            ])?;
        }
        w.flush()?;
    }
    let _ = topo;
    Ok(())
}

// ------------------------------------------------------------------ benchmarks
pub struct BenchRow {
    pub name: String,
    pub mode: String,  // cpu | gpu
    pub stage: String, // pipeline | similarity
    pub n_otus: usize,
    pub n_samples: usize,
    pub seconds: f64,
    pub repeats: usize,
    pub maxdiff: f64, // gpu vs cpu correctness (NAN for cpu rows)
    pub detail: String,
}

/// Benchmark the real `run_pipeline` (artifacts disabled) instead of the copy of
/// the pipeline this module used to carry, so the numbers cannot drift from the
/// code that users run.
pub fn bench_all(out_dir: &Path) -> Result<Vec<BenchRow>> {
    let _ = out_dir;
    let mut rows = Vec::new();
    let sizes = [
        (60usize, 12usize),
        (120, 12),
        (240, 24),
        (480, 24),
        (960, 32),
        (1920, 48),
    ];
    for &(n, m) in &sizes {
        let d = crate::simulate::simulate(n, m, 8, 4, 16, 0.45, 42);
        let mut best: Option<(f64, crate::pipeline::Timing, String)> = None;
        for _ in 0..3 {
            let opts = RunOptions {
                n_perm: 99,
                seed: 42,
                write_artifacts: false,
                ..Default::default()
            };
            let art = run_pipeline(
                &d.abundance,
                &d.otu_names,
                &d.sample_names,
                &d.traits,
                &d.trait_names,
                &opts,
            )?;
            let total = art.report.timing_s.total_s;
            let t = &art.report.timing_s;
            let detail = format!(
                "st={:.2} conv={} n={} E={} M={:.3} mods={}",
                art.report.rmt.st,
                art.report.rmt.converged,
                art.report.topology.n,
                art.report.topology.edges,
                art.report.topology.modularity,
                art.report.topology.n_modules
            );
            if best.as_ref().map(|(b, _, _)| total < *b).unwrap_or(true) {
                best = Some((
                    total,
                    Timing {
                        standardize_s: t.standardize_s,
                        similarity_s: t.similarity_s,
                        rmt_s: t.rmt_s,
                        topology_s: t.topology_s,
                        modules_s: t.modules_s,
                        eigengene_s: t.eigengene_s,
                        mantel_s: t.mantel_s,
                        export_s: t.export_s,
                        total_s: t.total_s,
                    },
                    detail,
                ));
            }
        }
        let (total, ts, detail) = best.unwrap();
        rows.push(BenchRow {
            name: format!("pipeline n={n} m={m}"),
            mode: "cpu".into(),
            stage: "pipeline".into(),
            n_otus: n,
            n_samples: m,
            seconds: total,
            repeats: 3,
            maxdiff: f64::NAN,
            detail: format!(
                "{detail} [min-of-3; stages: std={:.3} sim={:.3} rmt={:.3} topo={:.3} mod={:.3} eig={:.3} mantel={:.3}]",
                ts.standardize_s, ts.similarity_s, ts.rmt_s, ts.topology_s, ts.modules_s, ts.eigengene_s, ts.mantel_s
            ),
        });
    }

    // GPU vs CPU similarity at the same sizes (parity + speed)
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            for &(n, m) in &[
                (240usize, 24usize),
                (480, 24),
                (960, 32),
                (1920, 48),
                (3000, 64),
                (5000, 64),
                (8000, 96),
                (12000, 128),
            ] {
                let d = crate::simulate::simulate(n, m, 8, 4, 16, 0.45, 7);
                let z = linalg::standardize_rows(&d.abundance);
                let mut t_cpu = f64::INFINITY;
                let mut cpu = linalg::similarity_from_standardized(&z);
                for _ in 0..3 {
                    let t = Instant::now();
                    cpu = linalg::similarity_from_standardized(&z);
                    t_cpu = t_cpu.min(t.elapsed().as_secs_f64());
                }
                let mut t_gpu = f64::INFINITY;
                let mut maxd = f64::NAN;
                for rep in 0..3 {
                    let t = Instant::now();
                    match crate::gpu::similarity_gpu(&z) {
                        Ok(g) => {
                            t_gpu = t_gpu.min(t.elapsed().as_secs_f64());
                            if rep == 2 {
                                maxd = 0.0;
                                for i in 0..n {
                                    for j in 0..n {
                                        maxd = maxd.max((g[(i, j)] - cpu[(i, j)]).abs());
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[gpu] failed n={n}: {e}");
                            break;
                        }
                    }
                }
                rows.push(BenchRow {
                    name: format!("similarity n={n} m={m}"),
                    mode: "cpu".into(),
                    stage: "similarity".into(),
                    n_otus: n,
                    n_samples: m,
                    seconds: t_cpu,
                    repeats: 3,
                    maxdiff: f64::NAN,
                    detail: "rayon CPU min-of-3".into(),
                });
                rows.push(BenchRow {
                    name: format!("similarity n={n} m={m}"),
                    mode: "gpu".into(),
                    stage: "similarity".into(),
                    n_otus: n,
                    n_samples: m,
                    seconds: t_gpu,
                    repeats: 3,
                    maxdiff: maxd,
                    detail: format!(
                        "wgpu/Vulkan min-of-3, speedup vs cpu = {:.2}x, maxdiff = {maxd:.2e}",
                        t_cpu / t_gpu.max(1e-12)
                    ),
                });
            }
        } else {
            eprintln!("[gpu] no adapter; skipping GPU rows");
        }
    }
    #[cfg(not(feature = "gpu"))]
    eprintln!("[gpu] built without gpu feature; GPU rows skipped (build with --features gpu)");
    Ok(rows)
}
