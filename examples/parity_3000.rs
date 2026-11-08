use hydro_mena::gpu::similarity_gpu;
use hydro_mena::linalg;
fn main() {
    for n in [480usize, 960, 1920, 3000] {
        let m = 64;
        let d = hydro_mena::simulate::simulate(n, m, 8, 4, 16, 0.45, 7);
        let z = linalg::standardize_rows(&d.abundance);
        let mut cpu = linalg::correlation_matrix(&z);
        for i in 0..n { for j in 0..n { cpu[(i,j)] = cpu[(i,j)].abs(); } cpu[(i,i)] = 0.0; }
        let g = similarity_gpu(&z).unwrap();
        let mut maxd = 0.0f64; let (mut bi, mut bj) = (0,0);
        let mut bad = 0usize;
        for i in 0..n { for j in 0..n { let dd = (g[(i,j)] - cpu[(i,j)]).abs(); if dd > 1e-5 { bad += 1; } if dd > maxd { maxd = dd; bi = i; bj = j; } } }
        println!("n={n}: maxdiff={maxd:.3e} at ({bi},{bj}) gpu={:.4} cpu={:.4} bad(>1e-5)={bad}/{}", g[(bi,bj)], cpu[(bi,bj)], n*n);
    }
}
