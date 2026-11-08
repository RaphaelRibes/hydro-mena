//! Optional GPU compute via wgpu (Vulkan/RADV on this desktop). Currently
//! accelerates the similarity-matrix construction (pairwise |Pearson r|) and
//! the adjacency eigenvalue scan's matrix thresholding, the two O(n^2 m)
//! hot spots. Falls back to CPU transparently when no adapter is available.
//!
//! The similarity kernel is a tiled SGEMM-style dot product: 16x16 output
//! tiles per 16x16 workgroup, `vec4<f32>` shared-memory staging (64 floats per
//! k-slice), and one pipeline reused for every row-chunk. Rows are
//! mean-centred / sd-scaled and pre-multiplied by 1/sqrt(m-1) on the host, so
//! the shader accumulates a plain dot product and writes `|r|` directly.
//!
//! Adapter, device and pipeline are created once per process and cached: the
//! Vulkan device setup costs ~0.3 s, which used to be paid on every call and
//! dominated every matrix below n~10000.
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

/// Tile size must match `TILE`/`workgroup_size` in [`SIMILARITY_WGSL`].
#[cfg(feature = "gpu")]
const TILE: usize = 16;

#[cfg(feature = "gpu")]
const SIMILARITY_WGSL: &str = r#"
struct Dims {
    n : u32,       // full matrix dimension (columns of z, rows of s)
    kvec : u32,    // z row stride in vec4 units (m padded to a multiple of 4)
    row0 : u32,    // first row of this chunk
    nchunk : u32,  // rows in this chunk
}

@group(0) @binding(0) var<storage, read> z : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;

const TILE : u32 = 16u;
const KSTEP : u32 = 16u; // vec4 per shared-memory k-slice (64 floats)

var<workgroup> za : array<vec4<f32>, 256>;
var<workgroup> zb : array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
    let n = dims.n;
    let kvec = dims.kvec;
    let row0 = dims.row0;
    let nchunk = dims.nchunk;
    let i0 = row0 + wid.y * TILE;
    let j0 = wid.x * TILE;
    let row = i0 + lid.y;
    let col = j0 + lid.x;

    var acc = vec4<f32>(0.0);
    var kk : u32 = 0u;
    loop {
        if (kk >= kvec) { break; }
        // Stage one 16-row x 64-float slice of A (rows i) and B (rows j).
        let ca = kk + lid.x;
        if (row < row0 + nchunk && ca < kvec) {
            za[lid.y * TILE + lid.x] = z[row * kvec + ca];
        } else {
            za[lid.y * TILE + lid.x] = vec4<f32>(0.0);
        }
        let jrow = j0 + lid.x;
        let cb = kk + lid.y;
        if (jrow < n && cb < kvec) {
            zb[lid.x * TILE + lid.y] = z[jrow * kvec + cb];
        } else {
            zb[lid.x * TILE + lid.y] = vec4<f32>(0.0);
        }
        workgroupBarrier();
        for (var u : u32 = 0u; u < TILE; u = u + 1u) {
            acc = acc + za[lid.y * TILE + u] * zb[lid.x * TILE + u];
        }
        workgroupBarrier();
        kk = kk + KSTEP;
    }

    if (row < row0 + nchunk && col < n) {
        var v = abs(acc.x + acc.y + acc.z + acc.w);
        if (row == col) { v = 0.0; }
        s[(row - row0) * n + col] = v;
    }
}
"#;

/// Device + queue + the single compiled similarity pipeline, created once per
/// process. `wgpu` handles are `Send + Sync`, so a `OnceLock` is enough.
#[cfg(feature = "gpu")]
struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    /// Largest storage-buffer binding this adapter/device will accept; the
    /// row-chunk budget is clamped to it so a 128 MiB-limited adapter works.
    max_bind: u64,
}

#[cfg(feature = "gpu")]
fn gpu_ctx() -> Result<&'static GpuCtx> {
    static CTX: std::sync::OnceLock<GpuCtx> = std::sync::OnceLock::new();
    if let Some(ctx) = CTX.get() {
        return Ok(ctx);
    }
    let (device, queue, max_bind) = pollster::block_on(async {
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
        let alim = adapter.limits();
        let max_bind = (alim.max_storage_buffer_binding_size as u64).min(1 << 30);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: max_bind as u32,
                        max_compute_workgroups_per_dimension: alim
                            .max_compute_workgroups_per_dimension
                            .min(65535),
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| anyhow!("device: {e}"))?;
        Ok::<_, anyhow::Error>((device, queue, max_bind))
    })?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("similarity"),
        source: wgpu::ShaderSource::Wgsl(SIMILARITY_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            },
        ],
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cp"),
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let _ = CTX.set(GpuCtx { device, queue, bgl, pipeline, max_bind });
    Ok(CTX.get().expect("GPU context initialised"))
}

/// GPU pairwise absolute Pearson correlation of rows of `z` (already
/// standardized). Returns S = |R| with zero diagonal. Requires `gpu` feature;
/// on any failure returns Err so caller can fall back to CPU.
#[cfg(feature = "gpu")]
pub fn similarity_gpu(z: &Mat) -> Result<Mat> {
    let n = z.nrows();
    let m = z.ncols();
    if n == 0 || m < 2 {
        return Ok(Mat::zeros((n, n)));
    }
    let ctx = gpu_ctx()?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    // Host-side preparation, in f64: centre/scale each row, then fold the
    // 1/(m-1) Pearson denominator into the row so the shader is a pure dot
    // product. Rows are padded to a multiple of 4 floats for vec4 loads.
    let m4 = (m + 3) & !3;
    let kvec = m4 / 4;
    let inv_scale = 1.0 / (m as f64 - 1.0).sqrt();
    let mut z_data = vec![0.0f32; n * m4];
    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..m {
            sum += z[(i, k)];
        }
        let mean = sum / m as f64;
        let mut var = 0.0f64;
        for k in 0..m {
            let d = z[(i, k)] - mean;
            var += d * d;
        }
        let sd = (var / (m as f64 - 1.0)).sqrt();
        if sd > 0.0 {
            let row = &mut z_data[i * m4..i * m4 + m];
            for (k, dst) in row.iter_mut().enumerate() {
                *dst = (((z[(i, k)] - mean) / sd) * inv_scale) as f32;
            }
        } // else: constant row stays all-zero, matching standardize_rows
    }

    let z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("z"),
        contents: bytemuck::cast_slice(&z_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Row-chunks keep every storage buffer under the 128 MiB default binding
    // limit (or the adapter's own limit if that is smaller). All chunks share
    // one pipeline and are recorded into a single command buffer (one submit,
    // one poll for the whole matrix).
    const BUDGET: u64 = 128 * 1024 * 1024;
    let budget = BUDGET.min(ctx.max_bind);
    let max_rows = (budget / (4 * n as u64)).max(1) as usize;
    let chunk_rows = max_rows.min(n);
    let chunk_bytes = (chunk_rows * n * 4) as u64;

    let s_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("s"),
        size: chunk_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let tiles_x = ((n + TILE - 1) / TILE) as u32;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut out_bufs: Vec<(wgpu::Buffer, usize, usize, u64)> = Vec::new();
    for row0 in (0..n).step_by(chunk_rows) {
        let nchunk = (n - row0).min(chunk_rows);
        let tiles_y = ((nchunk + TILE - 1) / TILE) as u32;
        let dims = [n as u32, kvec as u32, row0 as u32, nchunk as u32];
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &ctx.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: z_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&ctx.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(tiles_x, tiles_y, 1);
        }
        let bytes = (nchunk * n * 4) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&s_buf, 0, &out_buf, 0, bytes);
        out_bufs.push((out_buf, row0, nchunk, bytes));
    }
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    for (buf, ..) in &out_bufs {
        let tx = tx.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    }
    device.poll(wgpu::Maintain::Wait);
    for _ in 0..out_bufs.len() {
        rx.recv()
            .map_err(|_| anyhow!("map channel closed"))?
            .map_err(|e| anyhow!("map failed: {e:?}"))?;
    }

    let mut s = Mat::zeros((n, n));
    for (buf, row0, nchunk, bytes) in &out_bufs {
        {
            let data = buf.slice(..*bytes).get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            let mut view = s.slice_mut(ndarray::s![*row0..*row0 + *nchunk, ..]);
            for (dst, &src) in view.iter_mut().zip(floats.iter()) {
                *dst = src as f64;
            }
        }
        buf.unmap();
    }
    Ok(s)
}

#[cfg(not(feature = "gpu"))]
pub fn similarity_gpu(_z: &Mat) -> Result<Mat> {
    Err(anyhow!("built without gpu feature"))
}
