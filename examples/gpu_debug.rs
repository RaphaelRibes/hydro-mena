use mena::gpu::similarity_gpu;
use mena::linalg;
use ndarray::Array2;
fn main() {
    let z = Array2::from_shape_vec((4,3), vec![
        1.0, -1.0, 0.0,
        1.0, 0.0, -1.0,
        -1.0, 1.0, 0.0,
        0.0, 1.0, -1.0,
    ]).unwrap();
    let cpu = linalg::correlation_matrix(&z);
    let mut cpu_abs = cpu.clone();
    for i in 0..4 { for j in 0..4 { cpu_abs[(i,j)] = cpu[(i,j)].abs(); } cpu_abs[(i,i)] = 0.0; }
    println!("cpu |R|: {:?}", cpu_abs);
    let g = similarity_gpu(&z).unwrap();
    println!("gpu S: {:?}", g);
    let mut maxd = 0.0f64;
    for i in 0..4 { for j in 0..4 { maxd = maxd.max((g[(i,j)] - cpu_abs[(i,j)]).abs()); } }
    println!("maxdiff vs |R| = {maxd:e}");
}
