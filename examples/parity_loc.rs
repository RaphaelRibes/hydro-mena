use mena::gpu::similarity_gpu;
use mena::linalg;
fn main() {
    let n = 2100usize; let m = 8;
    let d = mena::simulate::simulate(n, m, 4, 2, 8, 0.45, 7);
    let z = linalg::standardize_rows(&d.abundance);
    let mut cpu = linalg::correlation_matrix(&z);
    for i in 0..n { for j in 0..n { cpu[(i,j)] = cpu[(i,j)].abs(); } cpu[(i,i)] = 0.0; }
    let g = similarity_gpu(&z).unwrap();
    // find first bad index in scan order
    let mut first_bad = None;
    for idx in 0..(n*n) {
        let i = idx / n; let j = idx % n;
        if (g[(i,j)] - cpu[(i,j)]).abs() > 1e-5 { first_bad = Some((idx, i, j, g[(i,j)], cpu[(i,j)])); break; }
    }
    println!("first_bad = {:?}", first_bad);
    // count bad per row-block of 65535*64 = 4194240 idx
    let blk = 65535usize * 64;
    let mut counts = vec![0usize; (n*n + blk - 1)/blk + 1];
    for idx in 0..(n*n) {
        let i = idx / n; let j = idx % n;
        if (g[(i,j)] - cpu[(i,j)]).abs() > 1e-5 { counts[idx/blk] += 1; }
    }
    for (b, c) in counts.iter().enumerate() { if *c > 0 { println!("block {b} (idx {}..{}): {} bad", b*blk, (b+1)*blk, c); } }
}
