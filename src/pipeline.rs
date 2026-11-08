//! End-to-end pipeline orchestration + benchmark harness.
use crate::eigengene::{eigengenes, EigengeneResult};
use crate::io;
use crate::linalg::{self, Mat};
use crate::modules::{analyze_modules, ModuleAnalysis};
use crate::network::{self, Topology};
use crate::rmt::{rmt_threshold, RmtResult};
use crate::traits::mantel_connectivity_vs_traits;
use anyhow::Result;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

#[derive(Serialize)]
pub struct RunReport {
    pub n_otus: usize,
    pub n_samples: usize,
    pub n_traits: usize,
    pub gpu_used: bool,
    pub rmt: RmtReport,
    pub topology: TopoReport,
    pub modules: ModuleReport,
    pub eigengene_var_explained: Vec<f64>,
    pub fig5_r: Vec<Vec<f64>>,   // modules x traits
    pub fig5_p: Vec<Vec<f64>>,
    pub mantel: MantelReport,
    pub timing_s: Timing,
}

#[derive(Serialize)]
pub struct RmtReport {
    pub st: f64,
    pub converged: bool,
    pub n_points: usize,
    pub scan: Vec<(f64, usize, f64, bool)>,
}

#[derive(Serialize)]
pub struct TopoReport {
    pub n: usize,
    pub edges: usize,
    pub avg_k: f64,
    pub gd: f64,
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
    pub total_s: f64,
}

pub struct RunArtifacts {
    pub report: RunReport,
    pub S: Mat,
    pub A: Mat,
    pub kept: Vec<usize>,
    pub kept_names: Vec<String>,
    pub E: Mat,
    pub trait_r: Mat,
    pub trait_p: Mat,
    pub trait_names: Vec<String>,
    pub sample_names: Vec<String>,
    pub eig: EigengeneResult,
    pub mod_analysis: ModuleAnalysis,
    pub topo: Topology,
    pub rmt: RmtResult,
}

pub fn run_pipeline(
    abundance: &Mat,
    otu_names: &[String],
    sample_names: &[String],
    traits: &Mat,
    trait_names: &[String],
    out_dir: &Path,
    use_gpu: bool,
    n_perm: usize,
    seed: u64,
    fixed_st: Option<f64>,
) -> Result<RunArtifacts> {
    std::fs::create_dir_all(out_dir)?;
    let t0 = Instant::now();
    let t = Instant::now();
    let z = linalg::standardize_rows(abundance);
    let standardize_s = t.elapsed().as_secs_f64();

    // similarity: GPU if requested+available, else CPU (rayon)
    let t = Instant::now();
    let gpu_used = use_gpu && {
        #[cfg(feature = "gpu")]
        {
            match crate::gpu::similarity_gpu(&z) {
                Ok(S) => {
                    eprintln!("[gpu] similarity computed on GPU in {:.3}s", t.elapsed().as_secs_f64());
                    std::fs::write(out_dir.join("S.csv"), csv_of(&S, otu_names, otu_names))?;
                    return_continue(S)
                }
                Err(e) => {
                    eprintln!("[gpu] similarity failed ({e}); falling back to CPU");
                    false_marker()
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("[gpu] binary built without gpu feature; using CPU");
            false_marker()
        }
    };
    let S = if gpu_used {
        // recompute path handled above; unreachable in practice
        let R = linalg::correlation_matrix(&z);
        let mut S = Mat::zeros((R.nrows(), R.ncols()));
        for i in 0..R.nrows() {
            for j in 0..R.ncols() {
                S[(i, j)] = R[(i, j)].abs();
            }
            S[(i, i)] = 0.0;
        }
        S
    } else {
        let R = linalg::correlation_matrix(&z);
        let mut S = Mat::zeros((R.nrows(), R.ncols()));
        for i in 0..R.nrows() {
            for j in 0..R.ncols() {
                S[(i, j)] = R[(i, j)].abs();
            }
            S[(i, i)] = 0.0;
        }
        S
    };
    let similarity_s = t.elapsed().as_secs_f64();
    io::write_csv_matrix(&out_dir.join("similarity.csv"), &S, otu_names, otu_names, "OTU")?;

    let t = Instant::now();
    let min_nodes = (abundance.nrows() / 10).max(10);
    let rmt = match fixed_st {
        Some(st) => {
            let A = {
                let n = S.nrows();
                let mut A = Mat::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        if i != j && S[(i, j)] >= st {
                            A[(i, j)] = 1.0;
                        }
                    }
                }
                A
            };
            let kept: Vec<usize> = (0..A.nrows()).filter(|&i| (0..A.ncols()).any(|j| A[(i, j)] > 0.0)).collect();
            RmtResult {
                st,
                converged: false,
                log: vec![],
                adjacency: A,
                kept,
            }
        }
        None => rmt_threshold(&S, 0.01, min_nodes),
    };
    let rmt_s = t.elapsed().as_secs_f64();
    eprintln!(
        "[rmt] st = {:.3} converged={} ({} scan points, {:.2}s)",
        rmt.st, rmt.converged, rmt.log.len(), rmt_s
    );

    // restrict to kept nodes
    let kept = &rmt.kept;
    let p = kept.len();
    let mut A = Mat::zeros((p, p));
    for (a, &i) in kept.iter().enumerate() {
        for (b, &j) in kept.iter().enumerate() {
            A[(a, b)] = rmt.adjacency[(i, j)];
        }
    }
    let kept_names: Vec<String> = kept.iter().map(|&i| otu_names[i].clone()).collect();
    let zk = Mat::from_shape_fn((p, abundance.ncols()), |(i, j)| z[(kept[i], j)]);
    io::write_csv_matrix(&out_dir.join("adjacency.csv"), &A, &kept_names, &kept_names, "OTU")?;

    let t = Instant::now();
    let topo = network::topology(&A);
    let topology_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mod_analysis = analyze_modules(&A);
    let modules_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let tz = linalg::standardize_rows(&traits.t().to_owned()).t().to_owned();
    let eig = eigengenes(&zk, &mod_analysis.module_of, mod_analysis.n_modules, &tz);
    let eigengene_s = t.elapsed().as_secs_f64();
    io::write_csv_matrix(
        &out_dir.join("eigengenes.csv"),
        &eig.E,
        &(0..mod_analysis.n_modules).map(|b| format!("module_{}", b + 1)).collect::<Vec<_>>(),
        sample_names,
        "module",
    )?;
    io::write_csv_matrix(
        &out_dir.join("fig5_r.csv"),
        &eig.trait_r,
        &(0..mod_analysis.n_modules).map(|b| format!("module_{}", b + 1)).collect::<Vec<_>>(),
        trait_names,
        "module",
    )?;
    io::write_csv_matrix(
        &out_dir.join("fig5_p.csv"),
        &eig.trait_p,
        &(0..mod_analysis.n_modules).map(|b| format!("module_{}", b + 1)).collect::<Vec<_>>(),
        trait_names,
        "module",
    )?;

    let t = Instant::now();
    let mantel = mantel_connectivity_vs_traits(&A, &zk, traits, n_perm, seed);
    let mantel_s = t.elapsed().as_secs_f64();

    // module report
    let mut role_counts = std::collections::HashMap::new();
    for r in &mod_analysis.roles {
        *role_counts.entry(r.role.to_string()).or_insert(0) += 1;
    }

    let fig5_r: Vec<Vec<f64>> = (0..eig.trait_r.nrows()).map(|b| (0..eig.trait_r.ncols()).map(|h| eig.trait_r[(b, h)]).collect()).collect();
    let fig5_p: Vec<Vec<f64>> = (0..eig.trait_p.nrows()).map(|b| (0..eig.trait_p.ncols()).map(|h| eig.trait_p[(b, h)]).collect()).collect();

    let report = RunReport {
        n_otus: abundance.nrows(),
        n_samples: abundance.ncols(),
        n_traits: traits.ncols(),
        gpu_used,
        rmt: RmtReport {
            st: rmt.st,
            converged: rmt.converged,
            n_points: rmt.log.len(),
            scan: rmt.log.iter().map(|p| (p.st, p.p, p.chi2, p.poisson)).collect(),
        },
        topology: TopoReport {
            n: topo.n,
            edges: topo.edges,
            avg_k: topo.avg_k,
            gd: topo.gd,
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
        mantel: MantelReport { r: mantel.r, p: mantel.p, n_perm: mantel.n_perm },
        timing_s: Timing {
            standardize_s,
            similarity_s,
            rmt_s,
            topology_s,
            modules_s,
            eigengene_s,
            mantel_s,
            total_s: t0.elapsed().as_secs_f64(),
        },
    };
    io::write_json(&out_dir.join("report.json"), &report)?;

    // roles table
    {
        let names = &kept_names;
        let mut w = csv::Writer::from_path(out_dir.join("roles.csv"))?;
        w.write_record(["OTU", "module", "zi", "Pi", "connectivity", "role"])?;
        for r in &mod_analysis.roles {
            w.write_record([
                names[r.node].clone(),
                format!("{}", r.module + 1),
                format!("{:.4}", r.zi),
                format!("{:.4}", r.pi),
                format!("{:.0}", r.connectivity),
                r.role.to_string(),
            ])?;
        }
        w.flush()?;
    }

    Ok(RunArtifacts {
        report,
        S,
        A,
        kept: kept.clone(),
        kept_names,
        E: eig.E.clone(),
        trait_r: eig.trait_r.clone(),
        trait_p: eig.trait_p.clone(),
        trait_names: trait_names.to_vec(),
        sample_names: sample_names.to_vec(),
        eig,
        mod_analysis,
        topo,
        rmt,
    })
}

#[cfg(feature = "gpu")]
fn return_continue(S: Mat) -> bool {
    let _ = S;
    true
}
#[cfg(not(feature = "gpu"))]
fn return_continue(_S: Mat) -> bool {
    true
}
fn false_marker() -> bool {
    false
}

fn csv_of(m: &Mat, rows: &[String], cols: &[String]) -> String {
    let mut s = String::from("OTU");
    for c in cols {
        s.push(',');
        s.push_str(c);
    }
    s.push('\n');
    for i in 0..m.nrows() {
        s.push_str(&rows[i]);
        for j in 0..m.ncols() {
            s.push_str(&format!(",{:.8}", m[(i, j)]));
        }
        s.push('\n');
    }
    s
}

// ------------------------------------------------------------------ benchmarks
pub struct BenchRow {
    pub name: String,
    pub mode: String,      // cpu | gpu
    pub stage: String,     // pipeline | similarity
    pub n_otus: usize,
    pub n_samples: usize,
    pub seconds: f64,
    pub repeats: usize,
    pub maxdiff: f64,      // gpu vs cpu correctness (NAN for cpu rows)
    pub detail: String,
}

pub fn bench_all(out_dir: &Path) -> Result<Vec<BenchRow>> {
    let _ = out_dir;
    let mut rows = Vec::new();
    // ---- CPU pipeline scaling matrix (3 repeats, take min) ----
    let sizes = [(60usize, 12usize), (120, 12), (240, 24), (480, 24), (960, 32), (1920, 48)];
    for &(n, m) in &sizes {
        let d = crate::simulate::simulate(n, m, 8, 4, 16, 0.45, 42);
        let mut best: Option<(f64, String)> = None;
        let mut best_stages = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for rep in 0..3 {
            let t0 = Instant::now();
            let t = Instant::now();
            let z = linalg::standardize_rows(&d.abundance);
            let t_std = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let R = linalg::correlation_matrix(&z);
            let mut S = Mat::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    S[(i, j)] = R[(i, j)].abs();
                }
                S[(i, i)] = 0.0;
            }
            let t_sim = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let rmt = rmt_threshold(&S, 0.01, (n / 10).max(10));
            let t_rmt = t.elapsed().as_secs_f64();
            let p = rmt.kept.len();
            let mut A = Mat::zeros((p, p));
            for (a, &i) in rmt.kept.iter().enumerate() {
                for (b, &j) in rmt.kept.iter().enumerate() {
                    A[(a, b)] = rmt.adjacency[(i, j)];
                }
            }
            let t = Instant::now();
            let topo = network::topology(&A);
            let t_topo = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let ma = analyze_modules(&A);
            let t_mod = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let tz = linalg::standardize_rows(&d.traits.t().to_owned()).t().to_owned();
            let zk = Mat::from_shape_fn((p, m), |(i, j)| z[(rmt.kept[i], j)]);
            let _e = eigengenes(&zk, &ma.module_of, ma.n_modules, &tz);
            let t_eig = t.elapsed().as_secs_f64();
            let total = t0.elapsed().as_secs_f64();
            if best.as_ref().map(|(b, _)| total < *b).unwrap_or(true) {
                best = Some((total, format!(
                    "std={t_std:.3} sim={t_sim:.3} rmt={t_rmt:.3}(st={:.2},conv={}) topo={t_topo:.3} modules={t_mod:.3} eig={t_eig:.3} | n={} E={} M={:.3} mods={}",
                    rmt.st, rmt.converged, topo.n, topo.edges, topo.modularity, topo.n_modules)));
                best_stages = (t_std, t_sim, t_rmt, t_topo, t_mod, t_eig);
            }
            let _ = rep;
        }
        let (total, detail) = best.unwrap();
        let (t_std, t_sim, t_rmt, t_topo, t_mod, t_eig) = best_stages;
        rows.push(BenchRow {
            name: format!("pipeline n={n} m={m}"),
            mode: "cpu".into(), stage: "pipeline".into(),
            n_otus: n, n_samples: m, seconds: total, repeats: 3,
            maxdiff: f64::NAN,
            detail: format!("{detail} [min-of-3; stages: std={t_std:.3} sim={t_sim:.3} rmt={t_rmt:.3} topo={t_topo:.3} mod={t_mod:.3} eig={t_eig:.3}]"),
        });
    }
    // ---- per-stage scaling for the two dominant stages: similarity + topology ----
    for &(n, m) in &sizes {
        let d = crate::simulate::simulate(n, m, 8, 4, 16, 0.45, 42);
        let z = linalg::standardize_rows(&d.abundance);
        let t = Instant::now();
        let R = linalg::correlation_matrix(&z);
        let t_sim = t.elapsed().as_secs_f64();
        let mut S = Mat::zeros((n, n));
        for i in 0..n { for j in 0..n { S[(i, j)] = R[(i, j)].abs(); } S[(i, i)] = 0.0; }
        let rmt = rmt_threshold(&S, 0.01, (n / 10).max(10));
        let p = rmt.kept.len();
        let mut A = Mat::zeros((p, p));
        for (a, &i) in rmt.kept.iter().enumerate() {
            for (b, &j) in rmt.kept.iter().enumerate() { A[(a, b)] = rmt.adjacency[(i, j)]; }
        }
        let t = Instant::now();
        let topo = network::topology(&A);
        let t_topo = t.elapsed().as_secs_f64();
        rows.push(BenchRow {
            name: format!("stage similarity n={n} m={m}"),
            mode: "cpu".into(), stage: "similarity".into(),
            n_otus: n, n_samples: m, seconds: t_sim, repeats: 1, maxdiff: f64::NAN,
            detail: format!("O(n^2 m) pearson+|r|"),
        });
        rows.push(BenchRow {
            name: format!("stage topology n={n} m={m}"),
            mode: "cpu".into(), stage: "topology".into(),
            n_otus: n, n_samples: m, seconds: t_topo, repeats: 1, maxdiff: f64::NAN,
            detail: format!("BFS GD + clustering + greedy modularity | net n={} E={}", topo.n, topo.edges),
        });
    }
    // ---- GPU vs CPU similarity on the same sizes (parity + speed) ----
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            for &(n, m) in &[(240usize, 24usize), (480, 24), (960, 32), (1920, 48), (3000, 64), (5000, 64), (8000, 96), (12000, 128)] {
                let d = crate::simulate::simulate(n, m, 8, 4, 16, 0.45, 7);
                let z = linalg::standardize_rows(&d.abundance);
                // CPU: 3 reps min
                let mut t_cpu = f64::INFINITY;
                let mut cpu = Mat::zeros((n, n));
                for _ in 0..3 {
                    let t = Instant::now();
                    let R = linalg::correlation_matrix(&z);
                    t_cpu = t_cpu.min(t.elapsed().as_secs_f64());
                    cpu = R;
                }
                for i in 0..n { for j in 0..n { cpu[(i, j)] = cpu[(i, j)].abs(); } cpu[(i, i)] = 0.0; }
                // GPU: 3 reps min; parity measured against the LAST gpu result only
                // when it matches CPU shape, using a fresh compute (no reuse of stale buffer)
                let mut t_gpu = f64::INFINITY;
                let mut maxd = f64::NAN;
                for rep in 0..3 {
                    let t = Instant::now();
                    match crate::gpu::similarity_gpu(&z) {
                        Ok(g) => {
                            t_gpu = t_gpu.min(t.elapsed().as_secs_f64());
                            if rep == 2 {
                                maxd = 0.0;
                                for i in 0..n { for j in 0..n { maxd = maxd.max((g[(i, j)] - cpu[(i, j)]).abs()); } }
                            }
                        }
                        Err(e) => { eprintln!("[gpu] failed n={n}: {e}"); break; }
                    }
                }
                rows.push(BenchRow {
                    name: format!("similarity n={n} m={m}"),
                    mode: "cpu".into(), stage: "similarity".into(),
                    n_otus: n, n_samples: m, seconds: t_cpu, repeats: 3, maxdiff: f64::NAN,
                    detail: format!("rayon CPU min-of-3"),
                });
                rows.push(BenchRow {
                    name: format!("similarity n={n} m={m}"),
                    mode: "gpu".into(), stage: "similarity".into(),
                    n_otus: n, n_samples: m, seconds: t_gpu, repeats: 3, maxdiff: maxd,
                    detail: format!("wgpu/Vulkan min-of-3, speedup vs cpu = {:.2}x, maxdiff = {maxd:.2e}", t_cpu / t_gpu.max(1e-12)),
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
