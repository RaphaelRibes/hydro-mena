//! End-to-end pipeline orchestration + benchmark harness.
//!
//! Every stage writes a CSV a plotting layer or a GUI can read directly:
//!
//! | file | content |
//! |---|---|
//! | `report.json` | all scalars + timings (single source of truth) |
//! | `similarity.csv` | S = abs(r), n x n (only with `--write-similarity`; raw signed r when `--signed`) |
//! | `adjacency.csv` | thresholded 0/1 graph on the kept nodes |
//! | `edges.csv` | `source,target,module_source,module_target,sign` (sign = +1/-1, see `--signed`) |
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
use crate::modules::{
    analyze_modules_with, bootstrap_stability, detect_modules, BootReport, ModuleAnalysis,
    ModulesAlgo,
};
use crate::network;
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
    /// edges whose raw Pearson r is negative (competition), counted on the
    /// kept graph; 0 unless `--signed` was requested
    pub n_negative_edges: usize,
    /// bootstrap module stability (Jaccard over resampled samples); absent
    /// unless `RunOptions::boot > 0`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boot: Option<BootReport>,
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
    /// true when `--signed` kept the sign of r (negative edges recorded)
    pub signed: bool,
    /// bootstrap replicates for module stability (samples resampled with
    /// replacement, Jaccard per module); 0 = off
    pub boot: usize,
}

#[derive(Serialize)]
pub struct RmtReport {
    pub st: f64,
    pub converged: bool,
    /// no threshold passed the GOF: `st` is NaN and no adjacency is built
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
    /// Clauset MLE power-law fit of the degree tail (gamma, xmin, ks, p)
    pub clauset_gamma: f64,
    pub clauset_xmin: f64,
    pub clauset_ks: f64,
    pub clauset_p: f64,
    /// observed-vs-null z-scores (degree-preserving randomisations); absent
    /// unless `RunOptions::null_samples > 0`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub null: Option<crate::nullmodel::NullReport>,
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
    pub null_s: f64,
    pub modules_s: f64,
    pub eigengene_s: f64,
    pub mantel_s: f64,
    pub export_s: f64,
    pub total_s: f64,
    /// bootstrap module-stability pass (inside the module stage)
    pub boot_s: f64,
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
    /// spectral-test toggles (ablated in the B1 study)
    pub rmt_cfg: crate::rmt::RmtConfig,
    /// apply a per-OTU centred log-ratio transform before standardising.
    /// Abundance tables are compositional, so only ratios between taxa carry
    /// information; CLR removes the constant-sum artefact that otherwise shows
    /// up as spurious correlation between unrelated OTUs.
    pub clr: bool,
    /// pseudocount added before the log in the CLR transform (0 is allowed but
    /// any exact zero in the table then clamps to a tiny positive value)
    pub clr_pseudocount: f64,
    /// module-detection optimiser: Louvain (default) or greedy (ablation)
    pub modules_algo: ModulesAlgo,
    /// keep the sign of Pearson r instead of |r| (report §10 item 5). The RMT
    /// stage still thresholds |r| — negative weights would break Deng Eq. 3 —
    /// so the graph, topology and modularity are identical to the default
    /// mode; the sign survives only as edge metadata (`edges.csv`) plus
    /// `n_negative_edges` in report.json.
    pub signed: bool,
    /// degree-preserving null replicates for topology z-scores; 0 = off
    pub null_samples: usize,
    /// bootstrap replicates for module stability (samples resampled with
    /// replacement, Jaccard per module); 0 = off
    pub boot: usize,
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
            alpha: 0.01,
            write_similarity: false,
            write_artifacts: true,
            rmt_cfg: crate::rmt::RmtConfig::default(),
            clr: false,
            clr_pseudocount: 1.0,
            modules_algo: ModulesAlgo::Louvain,
            signed: false,
            boot: 0,
            null_samples: 0,
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
///
/// `signed` selects the raw Pearson r instead of |r|. The GPU kernel computes
/// |r| (see `gpu.rs`), so a signed request is served on the CPU and says so.
/// Similarity and thresholded adjacency at a fixed `st`, with no RMT scan and
/// no node pruning: the shared core of a `--st` run and the bootstrap
/// replicates, so both see exactly the same graph construction.
///
/// Returns `(S, A)` on the full OTU set (`S = |r|` of the row-standardised
/// table, `A_ij = 1` when `S_ij >= st`). `abundance` must already carry the
/// pipeline's preprocessing: [`run_pipeline`] applies `--clr` once and passes
/// the transformed table here, so the bootstrap inherits the same transform.
pub fn adjacency_at(abundance: &Mat, st: f64) -> (Mat, Mat) {
    let z = linalg::standardize_rows(abundance);
    let s = linalg::similarity_from_standardized(&z);
    let a = crate::rmt::adjacency(&s, st);
    (s, a)
}

fn similarity(z: &Mat, use_gpu: bool, signed: bool) -> (Mat, bool) {
    if signed {
        if use_gpu {
            eprintln!("[gpu] --signed needs the raw r; the GPU kernel only computes |r|, so using CPU");
        }
        return (linalg::similarity_signed(z), false);
    }
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

    // Compositionality fix, applied to the table between loading and
    // standardising, i.e. where a preprocessing transform belongs. CLR output is
    // already centred per row (each OTU row sums to zero) and `standardize_rows`
    // re-centres and rescales every row anyway, so this later pass only divides
    // each CLR row by its own standard deviation: it cannot undo the log-ratio
    // transform, and it keeps the paper's transform -> standardise -> correlate
    // shape intact instead of special-casing CLR to skip a step.
    let abundance: std::borrow::Cow<'_, Mat> = if opts.clr {
        std::borrow::Cow::Owned(linalg::clr_transform(abundance, opts.clr_pseudocount))
    } else {
        std::borrow::Cow::Borrowed(abundance)
    };
    let abundance = abundance.as_ref();

    let t = Instant::now();
    let z = linalg::standardize_rows(abundance);
    let standardize_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (s_raw, gpu_used) = similarity(&z, opts.use_gpu, opts.signed);
    let similarity_s = t.elapsed().as_secs_f64();
    // Signed mode (report §10 item 5): `s_raw` holds r, not |r|. Deng Eq. 3 and
    // the Poisson spacing GOF are only defined for non-negative weights, so the
    // RMT scan, the adjacency, the topology and the modularity all keep running
    // on |r| — i.e. exactly the paper-faithful graph of the default mode, whose
    // threshold semantics are unchanged. The sign is carried as edge metadata
    // (`sign` column in edges.csv, `n_negative_edges` in report.json) instead
    // of as a negative weight in the spectrum. `s_raw` itself is what goes to
    // similarity.csv in signed mode, so the artifact keeps the full r.
    let s_abs_buf;
    let s_abs: &Mat = if opts.signed {
        s_abs_buf = s_raw.mapv(f64::abs);
        &s_abs_buf
    } else {
        &s_raw
    };
    if opts.write_artifacts && opts.write_similarity {
        io::write_csv_matrix(&opts.out_dir.join("similarity.csv"), &s_raw, otu_names, otu_names, "OTU")?;
    }

    let t = Instant::now();
    let min_nodes = (abundance.nrows() / 10).max(10);
    let rmt = match opts.fixed_st {
        Some(st) => crate::rmt::at_threshold(s_abs, st),
        None => rmt_threshold(s_abs, opts.rmt_start, min_nodes, opts.alpha, &opts.rmt_cfg),
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

    if !rmt.estimable {
        eprintln!(
            "[rmt] NOT ESTIMABLE — {} candidates evaluated, none accepted. \
             No adjacency, no network, no Fig-5: pass --st <value> to force a threshold.",
            rmt.log.len()
        );
        let report = RunReport {
            n_otus: abundance.nrows(),
            n_samples: abundance.ncols(),
            n_traits: traits.ncols(),
            gpu_used,
            n_negative_edges: 0,
            boot: None,
            input: InputMeta {
                abundance: opts.abundance_path.clone(),
                traits: opts.traits_path.clone(),
                seed: opts.seed,
                permutations: opts.n_perm,
                fixed_threshold: opts.fixed_st,
                signed: opts.signed,
            boot: opts.boot,
            },
            rmt: RmtReport {
                st: f64::NAN,
                converged: false,
                fallback: true,
                alpha: opts.alpha,
                n_points: rmt.log.len(),
                n_kept: 0,
                edges: 0,
                scan: rmt.log.clone(),
            },
            topology: TopoReport {
                n: 0, edges: 0, density: f64::NAN, avg_k: f64::NAN, gd: f64::NAN,
                connected_frac: f64::NAN, n_components: 0, giant_frac: f64::NAN,
                avg_cc: f64::NAN, modularity: f64::NAN, n_modules: 0,
                r2_powerlaw: f64::NAN, gamma: f64::NAN, r2_hierarchy: f64::NAN,
                clauset_gamma: f64::NAN, clauset_xmin: f64::NAN,
                clauset_ks: f64::NAN, clauset_p: f64::NAN,
                // no adjacency exists, so there is nothing to randomise
                null: None,
            },
            modules: ModuleReport {
                modularity: f64::NAN, n_modules: 0, sizes: Vec::new(),
                role_counts: std::collections::HashMap::new(),
            },
            eigengene_var_explained: Vec::new(),
            fig5_r: Vec::new(),
            fig5_p: Vec::new(),
            fig5_q: Vec::new(),
            mantel: MantelReport { r: f64::NAN, p: f64::NAN, n_perm: opts.n_perm, r_null_mean: f64::NAN },
            timing_s: Timing {
                standardize_s,
                similarity_s,
                rmt_s,
                topology_s: f64::NAN,
                null_s: f64::NAN,
                modules_s: f64::NAN,
                eigengene_s: f64::NAN,
                mantel_s: f64::NAN,
                export_s: 0.0,
                total_s: t0.elapsed().as_secs_f64(),
                boot_s: f64::NAN,
            },
        };
        if opts.write_artifacts {
            write_scan_csv(&opts.out_dir, &rmt.log)?;
            io::write_json(&opts.out_dir.join("report.json"), &report)?;
        }
        return Ok(RunArtifacts { report, adjacency: Mat::zeros((0, 0)), kept_names: Vec::new() });
    }

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

    // Negative edges on exactly the graph edges.csv enumerates (the kept
    // subgraph at |r| >= st): a pair is negative when its raw r < 0. Computed
    // here, once, so report.json and edges.csv cannot disagree.
    let n_negative_edges = if opts.signed {
        let mut c = 0usize;
        for x in 0..p {
            for y in (x + 1)..p {
                if a[(x, y)] > 0.0 && s_raw[(kept[x], kept[y])] < 0.0 {
                    c += 1;
                }
            }
        }
        c
    } else {
        0
    };

    let t = Instant::now();
    let (topo, _) = network::topology_parts(&a);
    let topology_s = t.elapsed().as_secs_f64();

    // Null models: observed topology vs `b` degree-preserving randomisations.
    // Off by default (0) because each replicate costs a full rewiring + the
    // statistics on it; when on, the z-scores land in report.json under
    // `topology.null`.
    let t = Instant::now();
    let null = if opts.null_samples > 0 {
        Some(crate::nullmodel::null_zscores(&a, opts.null_samples, opts.seed))
    } else {
        None
    };
    let null_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    // Partition from the selected optimiser (Louvain by default, greedy as the
    // ablation), then roles on exactly that partition: re-optimising inside
    // `analyze_modules_with` would let the reported modularity describe a
    // different partition than the roles and eigengenes do.
    let det = detect_modules(&a, opts.modules_algo);
    let mod_analysis: ModuleAnalysis =
        analyze_modules_with(&network::adjacency_list(&a), det.modularity, det.module_of);
    let modules_s = t.elapsed().as_secs_f64();

    let t_boot = Instant::now();
    let boot = if opts.boot > 0 {
        let sample_cols: Vec<usize> = (0..abundance.ncols()).collect();
        Some(bootstrap_stability(
            abundance,
            &sample_cols,
            rmt.st,
            opts.modules_algo,
            opts.boot,
            opts.seed,
        ))
    } else {
        None
    };
    let boot_s = t_boot.elapsed().as_secs_f64();

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
        n_negative_edges,
        boot,
        input: InputMeta {
            abundance: opts.abundance_path.clone(),
            traits: opts.traits_path.clone(),
            seed: opts.seed,
            permutations: opts.n_perm,
            fixed_threshold: opts.fixed_st,
            signed: opts.signed,
            boot: opts.boot,
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
            // reported from the partition actually used below, not from the
            // greedy partition `topology_parts` builds internally
            modularity: det.modularity,
            n_modules: det.n_modules,
            r2_powerlaw: topo.r2_powerlaw,
            gamma: topo.gamma,
            r2_hierarchy: topo.r2_hierarchy,
            clauset_gamma: topo.clauset_gamma,
            clauset_xmin: topo.clauset_xmin,
            clauset_ks: topo.clauset_ks,
            clauset_p: topo.clauset_p,
            null,
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
            null_s,
            boot_s,
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
        &s_raw,
        otu_names,
        sample_names,
        trait_names,
        &rmt,
        &mod_analysis,
        &eig,
        &gs,
        &module_names,
        opts.write_similarity,
        opts.signed,
    )?;
    let export_s = t.elapsed().as_secs_f64();

    let mut report = report;
    report.timing_s.export_s = export_s;
    report.timing_s.total_s = t0.elapsed().as_secs_f64();
    io::write_json(&opts.out_dir.join("report.json"), &report)?;

    Ok(RunArtifacts { report, adjacency: a, kept_names })
}


/// One row per candidate threshold, including the ones that never produced a
/// statistic. This file is the audit trail: it must explain a failure, not hide it.
fn write_scan_csv(out: &Path, log: &[crate::rmt::ScanPoint]) -> Result<()> {
    let mut w = csv::Writer::from_path(out.join("rmt_scan.csv"))?;
    w.write_record([
        "st", "p", "edges", "chi2", "dof", "crit", "poisson", "ks", "brody_beta", "ks_goe",
        "giant_frac", "status",
    ])?;
    for pt in log {
        w.write_record([
            format!("{:.4}", pt.st),
            pt.p.to_string(),
            pt.edges.to_string(),
            format!("{:.4}", pt.chi2),
            pt.dof.to_string(),
            format!("{:.4}", pt.crit),
            pt.poisson.to_string(),
            format!("{:.4}", pt.ks),
            // Brody spacing-ratio fit (beta = 1 Poisson, 2 GOE) and the KS
            // distance to the GOE CDF: the same point can be Poisson-compatible
            // on chi^2 and still lean GOE on beta, so both are written out and
            // neither is used to override the accepted/rejected status.
            format!("{:.4}", pt.brody_beta),
            format!("{:.4}", pt.ks_goe),
            format!("{:.4}", pt.giant_frac),
            pt.status.clone(),
        ])?;
    }
    w.flush()?;
    Ok(())
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
    ma: &ModuleAnalysis,
    eig: &crate::eigengene::EigengeneResult,
    gs: &Mat,
    module_names: &[String],
    write_similarity: bool,
    signed: bool,
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

    // edges. `sign` is always written: +1 when only |r| was kept (default
    // mode, where every edge is a positive association), and the sign of the
    // raw Pearson r under `--signed` (-1 = anticorrelated pair). 0 cannot
    // occur on a written edge: the edge exists only when |r| >= st > 0.
    {
        let mut w = csv::Writer::from_path(out.join("edges.csv"))?;
        w.write_record(["source", "target", "module_source", "module_target", "sign"])?;
        let n = a.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)] > 0.0 {
                    let sign = if signed && s[(rmt.kept[i], rmt.kept[j])] < 0.0 { -1 } else { 1 };
                    w.write_record([
                        names[i].clone(),
                        names[j].clone(),
                        format!("{}", ma.module_of[i] + 1),
                        format!("{}", ma.module_of[j] + 1),
                        sign.to_string(),
                    ])?;
                }
            }
        }
        w.flush()?;
    }

    // per-OTU table: everything a network viewer or a scatter plot needs
    {
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

    write_scan_csv(out, &rmt.log)?;

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
                        null_s: t.null_s,
                        modules_s: t.modules_s,
                        eigengene_s: t.eigengene_s,
                        mantel_s: t.mantel_s,
                        export_s: t.export_s,
                        total_s: t.total_s,
                        boot_s: t.boot_s,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate;

    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("mena_pipe_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn read_csv_rows(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
        let mut rdr = csv::Reader::from_path(path).expect("open csv");
        let header: Vec<String> = rdr.headers().expect("header").iter().map(str::to_string).collect();
        let rows = rdr
            .records()
            .map(|rec| rec.expect("record").iter().map(str::to_string).collect())
            .collect();
        (header, rows)
    }

    /// Simulated table plus one exact mirror of the first OTU: Pearson r of
    /// that pair is -1, so a signed run must contain at least one negative
    /// edge no matter how the rest of the data correlates.
    fn mirrored_input() -> (Mat, Vec<String>, Vec<String>, Mat, Vec<String>) {
        let d = simulate::simulate(80, 12, 3, 2, 6, 0.45, 11);
        let (n, m) = (d.abundance.nrows(), d.abundance.ncols());
        let mut ab = Mat::zeros((n + 1, m));
        ab.slice_mut(ndarray::s![..n, ..]).assign(&d.abundance);
        for j in 0..m {
            ab[(n, j)] = d.abundance[(0, m - 1 - j)];
        }
        let mut names = d.otu_names.clone();
        names.push("mirror_OTU".to_string());
        (ab, names, d.sample_names, d.traits, d.trait_names)
    }

    #[test]
    fn signed_mode_marks_negative_edges_and_reports_the_count() {
        let (ab, names, samples, traits, trait_names) = mirrored_input();

        let out_signed = scratch("signed");
        let opts_signed = RunOptions {
            out_dir: out_signed.clone(),
            n_perm: 19,
            seed: 4,
            fixed_st: Some(0.55),
            signed: true,
            ..Default::default()
        };
        let art = run_pipeline(&ab, &names, &samples, &traits, &trait_names, &opts_signed).unwrap();
        assert!(art.report.topology.edges > 0, "fixed st kept no edges");
        assert!(art.report.input.signed);

        let (header, rows) = read_csv_rows(&out_signed.join("edges.csv"));
        assert_eq!(header.last().map(String::as_str), Some("sign"), "header {header:?}");
        assert!(rows.iter().all(|r| r.len() == header.len()));
        let neg = rows.iter().filter(|r| r[4] == "-1").count();
        assert!(neg >= 1, "the mirrored OTU pair must be a negative edge");
        assert_eq!(neg, art.report.n_negative_edges, "report count vs edges.csv");
        assert!(rows.iter().all(|r| r[4] == "1" || r[4] == "-1"));

        // the field must survive serialisation, not just live in the struct
        let js: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out_signed.join("report.json")).unwrap()).unwrap();
        assert_eq!(js["n_negative_edges"].as_u64().unwrap() as usize, neg);
        assert_eq!(js["input"]["signed"].as_bool().unwrap(), true);

        // default mode: same graph (signed is metadata, not a different graph),
        // every edge +1, zero negative edges
        let out_plain = scratch("plain");
        let opts_plain = RunOptions {
            out_dir: out_plain.clone(),
            n_perm: 19,
            seed: 4,
            fixed_st: Some(0.55),
            signed: false,
            boot: 0,
            ..Default::default()
        };
        let art0 = run_pipeline(&ab, &names, &samples, &traits, &trait_names, &opts_plain).unwrap();
        let (header0, rows0) = read_csv_rows(&out_plain.join("edges.csv"));
        assert_eq!(header0, header);
        assert_eq!(art0.report.n_negative_edges, 0);
        assert!(rows0.iter().all(|r| r[4] == "1"));
        let pairs = |rs: &Vec<Vec<String>>| -> Vec<(String, String)> {
            rs.iter().map(|r| (r[0].clone(), r[1].clone())).collect()
        };
        assert_eq!(pairs(&rows), pairs(&rows0), "signed mode changed the edge set");

        let _ = std::fs::remove_dir_all(&out_signed);
        let _ = std::fs::remove_dir_all(&out_plain);
    }
}
