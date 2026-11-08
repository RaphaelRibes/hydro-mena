//! Optional GPU compute via wgpu (Vulkan/RADV on this desktop). Currently
//! accelerates the similarity-matrix construction (pairwise |Pearson r|) and
//! the adjacency eigenvalue scan's matrix thresholding, the two O(n^2 m)
//! hot spots. Falls back to CPU transparently when no adapter is available.
use crate::linalg::Mat;
use anyhow::{anyhow, Result};
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .is_some()
    })
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_available() -> bool {
    false
}

/// GPU pairwise absolute Pearson correlation of rows of `z` (already
/// standardized). Returns S = |R| with zero diagonal. Requires `gpu` feature;
/// on any failure returns Err so caller can fall back to CPU.
#[cfg(feature = "gpu")]
pub fn similarity_gpu(z: &Mat) -> Result<Mat> {
    let n = z.nrows();
    let m = z.ncols();
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow!("no GPU adapter"))?;
        let info = adapter.get_info();
        eprintln!("[gpu] using adapter: {} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 1 << 30, // 1 GiB: n=16000 -> 1 GiB f32 output
                        max_compute_workgroups_per_dimension: 65535,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| anyhow!("device: {e}"))?;


        // per-row mean/sd for exact Pearson in shader
        let n_ = z.nrows(); let m_ = z.ncols();
        let mut mean = vec![0.0f32; n_];
        let mut sd = vec![0.0f32; n_];
        for i in 0..n_ {
            let mut s = 0.0f64; for j in 0..m_ { s += z[(i,j)]; }
            mean[i] = (s / m_ as f64) as f32;
            let mut v = 0.0f64; for j in 0..m_ { let d = z[(i,j)] - s/m_ as f64; v += d*d; }
            sd[i] = ((v / (m_-1) as f64).sqrt()) as f32;
        }
        let mut z_data: Vec<f32> = z.iter().map(|&v| v as f32).collect();
        let mut params: Vec<f32> = Vec::with_capacity(2*n_);
        for i in 0..n_ { params.push(mean[i]); params.push(sd[i]); }
        let z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("z"),
            contents: bytemuck::cast_slice(&z_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // chunk rows so the storage buffer stays under the binding limit
        let max_rows = (134_217_728u64 / (4 * n as u64)).max(1) as usize; // rows per chunk
        let s_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("s"),
            size: (max_rows * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (max_rows * n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: z_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p_buf.as_entire_binding() },
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bind],
            push_constant_ranges: &[],
        });
        let pipe_layout = &pipe_layout; // pipelines built per chunk below use this ref
        let mut S = Mat::zeros((n, n));
        let chunk = max_rows.min(n);
        for row0 in (0..n).step_by(chunk) {
            let nchunk = (n - row0).min(chunk);
            // rebuild shader for this chunk (ROW0/NCHUNK baked in)
            let tiles_x = ((n as u32) + 7) / 8;
            let tiles_y = ((nchunk as u32) + 7) / 8;
            let total_tiles = tiles_x * tiles_y;
            let wgx = ((total_tiles + 63) / 64).min(65535);
            let shader_src = format!(
                r#"
@group(0) @binding(0) var<storage, read> z : array<f32>;
@group(0) @binding(1) var<storage, read_write> s : array<f32>; // nchunk * n
@group(0) @binding(2) var<storage, read> params : array<f32>;
const N : u32 = {n}u;
const M : u32 = {m}u;
const WGX : u32 = {wgx}u;
const TILES_X : u32 = {tiles_x}u;
const ROW0 : u32 = {row0}u;
const NCHUNK : u32 = {nchunk}u;
const TILE : u32 = 8u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{
    let tile = gid.x + gid.y * (WGX * 64u);
    if (tile >= TILES_X * ((NCHUNK + TILE - 1u) / TILE)) {{ return; }}
    let tx = tile % TILES_X;
    let ty = tile / TILES_X;
    let i0 = ROW0 + ty * TILE;
    let j0 = tx * TILE;
    for (var di : u32 = 0u; di < TILE; di = di + 1u) {{
        let i = i0 + di;
        if (i >= ROW0 + NCHUNK) {{ break; }}
        let mi = params[2u * i];
        let si = params[2u * i + 1u];
        for (var dj : u32 = 0u; dj < TILE; dj = dj + 1u) {{
            let j = j0 + dj;
            if (j >= N) {{ break; }}
            if (i == j) {{ s[(i - ROW0) * N + j] = 0.0; continue; }}
            let mj = params[2u * j];
            let sj = params[2u * j + 1u];
            var acc : f32 = 0.0;
            for (var k : u32 = 0u; k < M; k = k + 1u) {{
                acc = acc + (z[i * M + k] - mi) * (z[j * M + k] - mj);
            }}
            s[(i - ROW0) * N + j] = abs(acc / ((f32(M) - 1.0) * si * sj));
        }}
    }}
}}
"#,
                n = n, m = m, wgx = wgx, tiles_x = tiles_x, row0 = row0 as u32, nchunk = nchunk as u32
            );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("similarity_chunk"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cp_chunk"),
                layout: Some(pipe_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                let wg_x = wgx;
                let wg_y = (((total_tiles + 63) / 64) + wg_x - 1) / wg_x;
                let wg_y = wg_y.min(65535);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            let chunk_bytes = (nchunk * n * 4) as u64;
            encoder.copy_buffer_to_buffer(&s_buf, 0, &out_buf, 0, chunk_bytes);
            queue.submit(Some(encoder.finish()));
            let slice = out_buf.slice(..chunk_bytes);
            let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            {
                let data = slice.get_mapped_range();
                let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                for i in 0..nchunk {
                    for j in 0..n {
                        S[(row0 + i, j)] = floats[i * n + j] as f64;
                    }
                }
            }
            out_buf.unmap();
        }
        Ok(S)
    })
}


#[cfg(not(feature = "gpu"))]
pub fn similarity_gpu(_z: &Mat) -> Result<Mat> {
    Err(anyhow!("built without gpu feature"))
}
