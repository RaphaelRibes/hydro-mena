use mena::gpu::similarity_gpu;
use mena::linalg;
fn main() {
    for n in [2048usize, 2100, 2200, 2300, 2400, 2500, 2600, 2800, 3000] {
        let m = 8;
        let d = mena::simulate::simulate(n, m, 4, 2, 8, 0.45, 7);
        let z = linalg::standardize_rows(&d.abundance);
        let mut cpu = linalg::correlation_matrix(&z);
        for i in 0..n { for j in 0..n { cpu[(i,j)] = cpu[(i,j)].abs(); } cpu[(i,i)] = 0.0; }
        match similarity_gpu(&z) {
            Ok(g) => {
                let mut maxd = 0.0f64; let mut bad = 0usize;
                for i in 0..n { for j in 0..n { let dd = (g[(i,j)] - cpu[(i,j)]).abs(); if dd > 1e-5 { bad += 1; } if dd > maxd { maxd = dd; } } }
                println!("n={n}: maxdiff={maxd:.3e} bad={bad}/{} ({:.1}%)", n*n, 100.0*bad as f64/(n*n) as f64);
            }
            Err(e) => println!("n={n}: ERR {e}"),
        }
    }
}
