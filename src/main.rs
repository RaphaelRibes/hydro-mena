//! hydro-mena CLI — RMT-based Molecular Ecological Network Analysis
//! (Deng et al. 2012, BMC Bioinformatics 13:113).
//!
//! Subcommands:
//!   simulate   generate synthetic OTU abundance + traits CSVs
//!   run        full pipeline on abundance.csv + traits.csv -> artifacts + report.json
//!   steps      similarity + RMT scan only (diagnostics)
//!   bench      benchmark the real pipeline (and GPU similarity with --features gpu)
use anyhow::Result;
use clap::{Parser, Subcommand};
use hydro_mena::pipeline::RunOptions;
use hydro_mena::{io, pipeline, simulate};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hydro-mena", version, about = "RMT-based molecular ecological network analysis (MENA)")]
struct Cli {
    /// Use GPU (wgpu) for the similarity matrix when built with --features gpu
    #[arg(long, global = true)]
    gpu: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate synthetic data with planted modules correlated to traits
    Simulate {
        #[arg(long, default_value = "120")]
        otus: usize,
        #[arg(long, default_value = "12")]
        samples: usize,
        #[arg(long, default_value = "5")]
        modules: usize,
        #[arg(long, default_value = "4")]
        traits: usize,
        #[arg(long, default_value = "8")]
        background: usize,
        #[arg(long, default_value = "0.45")]
        noise: f64,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
    },
    /// Run the full MENA pipeline
    Run {
        #[arg(long)]
        abundance: PathBuf,
        #[arg(long)]
        traits: PathBuf,
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
        #[arg(long, default_value = "999")]
        permutations: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Fixed similarity threshold st (skip the RMT scan), e.g. 0.52 to match the 2025 repo
        #[arg(long)]
        st: Option<f64>,
        /// Lowest threshold the RMT scan considers (default 0.30)
        #[arg(long, default_value = "0.30")]
        rmt_start: f64,
        /// Significance level of the Poisson GOF in the RMT scan
        #[arg(long, default_value = "0.05")]
        alpha: f64,
        /// Also write the full n x n similarity matrix (large for many OTUs)
        #[arg(long)]
        write_similarity: bool,
    },
    /// Similarity + RMT scan diagnostics only
    Steps {
        #[arg(long)]
        abundance: PathBuf,
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
        #[arg(long, default_value = "0.30")]
        rmt_start: f64,
    },
    /// Benchmark the pipeline (CPU, and GPU similarity when built with the gpu feature)
    Bench {
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
        #[arg(long, default_value = "all")]
        which: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Simulate {
            otus,
            samples,
            modules,
            traits,
            background,
            noise,
            seed,
            out_dir,
        } => {
            std::fs::create_dir_all(&out_dir)?;
            let d = simulate::simulate(otus, samples, modules, traits, background, noise, seed);
            io::write_abundance_csv(&out_dir.join("abundance.csv"), &d.abundance, &d.otu_names, &d.sample_names)?;
            io::write_traits_csv(&out_dir.join("traits.csv"), &d.traits, &d.sample_names, &d.trait_names)?;
            // planted partition, so a recovered network can be scored against known truth
            {
                let mut w = csv::Writer::from_path(out_dir.join("planted.csv"))?;
                w.write_record(["OTU", "planted_module"])?;
                for (name, m) in d.otu_names.iter().zip(d.module_of.iter()) {
                    w.write_record([name.clone(), format!("{}", m + 1)])?;
                }
                w.flush()?;
            }
            println!(
                "wrote {} ({} OTUs x {} samples) and traits.csv ({} traits) to {}",
                out_dir.join("abundance.csv").display(),
                otus,
                samples,
                traits,
                out_dir.display()
            );
        }
        Cmd::Run {
            abundance,
            traits,
            out_dir,
            permutations,
            seed,
            st,
            rmt_start,
            alpha,
            write_similarity,
        } => {
            let (ab, otu_names, sample_names) = io::read_abundance_csv(&abundance)?;
            let (tr, _sn, trait_names) = io::read_traits_csv(&traits, &sample_names)?;
            let opts = RunOptions {
                out_dir: out_dir.clone(),
                use_gpu: cli.gpu,
                n_perm: permutations,
                seed,
                fixed_st: st,
                rmt_start,
                alpha,
                write_similarity,
                write_artifacts: true,
                abundance_path: abundance.display().to_string(),
                traits_path: traits.display().to_string(),
            };
            let art = pipeline::run_pipeline(&ab, &otu_names, &sample_names, &tr, &trait_names, &opts)?;
            let r = &art.report;
            println!("\n=== MENA report ===");
            println!("OTUs={} samples={} traits={} gpu={}", r.n_otus, r.n_samples, r.n_traits, r.gpu_used);
            println!(
                "RMT threshold st={:.3} converged={} fallback={} ({} scan pts, kept {}/{}, E={})",
                r.rmt.st, r.rmt.converged, r.rmt.fallback, r.rmt.n_points, r.rmt.n_kept, r.n_otus, r.rmt.edges
            );
            let t = &r.topology;
            println!(
                "network: n={} E={} density={:.4} avgK={:.2} GD={:.2} (connected {:.1}%, {} components) avgCC={:.3}",
                t.n, t.edges, t.density, t.avg_k, t.gd, 100.0 * t.connected_frac, t.n_components, t.avg_cc
            );
            println!(
                "modularity M={:.3} modules={} R2_power={:.2} gamma={:.2} R2_hier={:.2}",
                t.modularity, t.n_modules, t.r2_powerlaw, t.gamma, t.r2_hierarchy
            );
            println!("modules: sizes={:?} roles={:?}", r.modules.sizes, r.modules.role_counts);
            println!(
                "eigengene var explained: {:?}",
                r.eigengene_var_explained.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
            );
            println!("fig5 r (modules x traits):");
            for (b, row) in r.fig5_r.iter().enumerate() {
                let ps: Vec<String> = r.fig5_p[b].iter().map(|p| format!("{p:.3}")).collect();
                println!(
                    "  module_{}: r={:?} p={:?}",
                    b + 1,
                    row.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>(),
                    ps
                );
            }
            println!(
                "mantel connectivity-vs-GS: r={:.3} (null {:.3}) p={:.4} ({} perms)",
                r.mantel.r, r.mantel.r_null_mean, r.mantel.p, r.mantel.n_perm
            );
            println!("timing: total={:.2}s {:?}", r.timing_s.total_s, serde_json::to_value(&r.timing_s)?);
            println!("artifacts in {}", out_dir.display());
        }
        Cmd::Steps { abundance, out_dir, rmt_start } => {
            let (ab, otu_names, _sample_names) = io::read_abundance_csv(&abundance)?;
            std::fs::create_dir_all(&out_dir)?;
            let z = hydro_mena::linalg::standardize_rows(&ab);
            let s = hydro_mena::linalg::similarity_from_standardized(&z);
            let n = s.nrows();
            io::write_csv_matrix(&out_dir.join("similarity.csv"), &s, &otu_names, &otu_names, "OTU")?;
            let rmt = hydro_mena::rmt::rmt_threshold(&s, rmt_start, (n / 10).max(10), 0.01);
            println!(
                "st={:.3} converged={} fallback={} kept={}/{}",
                rmt.st,
                rmt.converged,
                rmt.fallback,
                rmt.kept.len(),
                n
            );
            io::write_json(&out_dir.join("rmt_scan.json"), &rmt.log)?;
        }
        Cmd::Bench { out_dir, which } => {
            std::fs::create_dir_all(&out_dir)?;
            let rows = pipeline::bench_all(&out_dir)?;
            let mut w = csv::Writer::from_path(out_dir.join("bench.csv"))?;
            w.write_record(["name", "mode", "stage", "n_otus", "n_samples", "seconds", "repeats", "maxdiff", "detail"])?;
            println!("\n=== benchmarks ({which}) ===");
            for r in &rows {
                println!(
                    "{:<34} {:<4} {:<10} n={:<5} m={:<3} {:>8.4}s  {}",
                    r.name, r.mode, r.stage, r.n_otus, r.n_samples, r.seconds, r.detail
                );
                w.write_record([
                    &r.name,
                    &r.mode,
                    &r.stage,
                    &r.n_otus.to_string(),
                    &r.n_samples.to_string(),
                    &format!("{:.6}", r.seconds),
                    &r.repeats.to_string(),
                    &if r.maxdiff.is_nan() { String::new() } else { format!("{:.3e}", r.maxdiff) },
                    &r.detail,
                ])?;
            }
            w.flush()?;
            println!("wrote {}", out_dir.join("bench.csv").display());
        }
    }
    Ok(())
}
