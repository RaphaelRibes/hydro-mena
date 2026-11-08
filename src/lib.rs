//! hydro-mena — RMT-based Molecular Ecological Network Analysis pipeline
//! (Deng et al., BMC Bioinformatics 2012, 13:113).
//!
//! Compute core in Rust (CPU multithreaded via rayon; optional GPU via wgpu).
//! Plotting is done by the thin Python wrapper (graphs only), not here.

pub mod linalg;
pub mod stats;
pub mod rmt;
pub mod network;
pub mod nullmodel;
pub mod modules;
pub mod eigengene;
pub mod traits;
pub mod simulate;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod pipeline;
pub mod io;
