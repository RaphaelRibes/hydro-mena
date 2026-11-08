//! CSV/JSON I/O for abundance tables, trait tables and result artifacts.
use crate::linalg::Mat;
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::path::Path;

pub fn write_abundance_csv(path: &Path, m: &Mat, row_names: &[String], col_names: &[String]) -> Result<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record(std::iter::once("OTU".to_string()).chain(col_names.iter().cloned()))?;
    for i in 0..m.nrows() {
        let mut rec = vec![row_names[i].clone()];
        for j in 0..m.ncols() {
            rec.push(format!("{:.10}", m[(i, j)]));
        }
        w.write_record(rec)?;
    }
    w.flush()?;
    Ok(())
}

pub fn read_abundance_csv(path: &Path) -> Result<(Mat, Vec<String>, Vec<String>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let header = rdr.headers()?.clone();
    let col_names: Vec<String> = header.iter().skip(1).map(|s| s.to_string()).collect();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut row_names = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        row_names.push(rec[0].to_string());
        let mut row = Vec::with_capacity(col_names.len());
        for j in 1..rec.len() {
            row.push(rec[j].parse::<f64>().context("non-numeric abundance")?);
        }
        rows.push(row);
    }
    let n = rows.len();
    let m = col_names.len();
    let data = Mat::from_shape_fn((n, m), |(i, j)| rows[i][j]);
    Ok((data, row_names, col_names))
}

pub fn write_traits_csv(path: &Path, t: &Mat, sample_names: &[String], trait_names: &[String]) -> Result<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record(std::iter::once("sample".to_string()).chain(trait_names.iter().cloned()))?;
    for i in 0..t.nrows() {
        let mut rec = vec![sample_names[i].clone()];
        for j in 0..t.ncols() {
            rec.push(format!("{:.10}", t[(i, j)]));
        }
        w.write_record(rec)?;
    }
    w.flush()?;
    Ok(())
}

pub fn read_traits_csv(path: &Path, samples: &[String]) -> Result<(Mat, Vec<String>, Vec<String>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let header = rdr.headers()?.clone();
    let trait_names: Vec<String> = header.iter().skip(1).map(|s| s.to_string()).collect();
    // Index by sample name instead of a linear `contains` scan per row, and
    // reject a file that does not cover every abundance column. The old version
    // silently returned a shorter matrix, and every downstream correlation then
    // paired OTU rows with the wrong samples (pearson() truncates to the shorter
    // vector, so nothing raised).
    let mut by_sample: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for rec in rdr.records() {
        let rec = rec?;
        let id = rec[0].to_string();
        let mut row = Vec::with_capacity(trait_names.len());
        for j in 1..rec.len() {
            row.push(
                rec[j]
                    .parse::<f64>()
                    .with_context(|| format!("non-numeric trait value in sample {id}"))?,
            );
        }
        if row.len() != trait_names.len() {
            anyhow::bail!(
                "{}: sample {id} has {} values but the header declares {} traits",
                path.display(),
                row.len(),
                trait_names.len()
            );
        }
        by_sample.insert(id, row);
    }
    let missing: Vec<&String> = samples.iter().filter(|s| !by_sample.contains_key(*s)).collect();
    if !missing.is_empty() {
        let head: Vec<&str> = missing.iter().take(5).map(|s| s.as_str()).collect();
        anyhow::bail!(
            "{}: {} of {} abundance samples have no trait row (e.g. {:?}); \\
             the trait table must cover every sample column",
            path.display(),
            missing.len(),
            samples.len(),
            head
        );
    }
    let m = trait_names.len();
    let data = Mat::from_shape_fn((samples.len(), m), |(i, j)| by_sample[&samples[i]][j]);
    Ok((data, samples.to_vec(), trait_names))
}

pub fn write_json<T: Serialize>(path: &Path, obj: &T) -> Result<()> {
    let s = serde_json::to_string_pretty(obj)?;
    fs::write(path, s).with_context(|| format!("write {}", path.display()))
}

pub fn write_csv_matrix(path: &Path, m: &Mat, row_names: &[String], col_names: &[String], first_col: &str) -> Result<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record(std::iter::once(first_col.to_string()).chain(col_names.iter().cloned()))?;
    for i in 0..m.nrows() {
        let mut rec = vec![row_names[i].clone()];
        for j in 0..m.ncols() {
            rec.push(format!("{:.8}", m[(i, j)]));
        }
        w.write_record(rec)?;
    }
    w.flush()?;
    Ok(())
}
