//! The GPU half, through wgpu (Direct3D 12 or Vulkan on Windows, Metal on a
//! Mac): the depth refinement kernels (`shaders/depth.wgsl`) and the
//! two-eye warp and packing (`shaders/warp.wgsl`).

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use half::f16;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    linear: wgpu::Sampler,
    guide: wgpu::ComputePipeline,
    bilateral: wgpu::ComputePipeline,
    encode: wgpu::ComputePipeline,
    yuv_to_rgb: wgpu::ComputePipeline,
    mesh: wgpu::RenderPipeline,
    pack: wgpu::ComputePipeline,
    grid: (wgpu::Buffer, wgpu::Buffer, u32),
}

/// The dense warp grid's vertices per axis (`StereoWarpResources.denseGridColumns/Rows`).
const GRID_COLUMNS: usize = 769;
const GRID_ROWS: usize = 433;
const MSAA: u32 = 4;

fn compute(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU for the warp: {e}"))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vergence-convert"),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        }))
        .map_err(|e| format!("the GPU refused a device: {e}"))?;
        let depth = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/depth.wgsl").into()),
        });
        let warp = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/warp.wgsl").into()),
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mesh = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &warp,
                entry_point: Some("mesh_vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 }],
                })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                // lessEqual: the farthest geometry sits exactly at the cleared 1.
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false },
            fragment: Some(wgpu::FragmentState {
                module: &warp,
                entry_point: Some("mesh_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let mut positions: Vec<f32> = Vec::with_capacity(GRID_COLUMNS * GRID_ROWS * 2);
        for j in 0..GRID_ROWS {
            for i in 0..GRID_COLUMNS {
                positions.push(i as f32 / (GRID_COLUMNS - 1) as f32);
                positions.push(j as f32 / (GRID_ROWS - 1) as f32);
            }
        }
        let mut indices: Vec<u32> = Vec::with_capacity((GRID_COLUMNS - 1) * (GRID_ROWS - 1) * 6);
        for j in 0..GRID_ROWS - 1 {
            for i in 0..GRID_COLUMNS - 1 {
                let a = (j * GRID_COLUMNS + i) as u32;
                let c = a + GRID_COLUMNS as u32;
                indices.extend_from_slice(&[a, c, a + 1, a + 1, c, c + 1]);
            }
        }
        let vertices = buffer_with(&device, "grid", bytemuck::cast_slice(&positions), wgpu::BufferUsages::VERTEX);
        let index_buffer = buffer_with(&device, "grid indices", bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX);
        Ok(Gpu {
            guide: compute(&device, &depth, "guide_luma"),
            bilateral: compute(&device, &depth, "joint_bilateral"),
            encode: compute(&device, &depth, "temporal_encode"),
            yuv_to_rgb: compute(&device, &warp, "yuv_to_rgb"),
            pack: compute(&device, &warp, "pack_picture"),
            mesh,
            grid: (vertices, index_buffer, indices.len() as u32),
            linear,
            adapter_name,
            device,
            queue,
        })
    }

    fn texture(&self, label: &str, w: usize, h: usize, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, samples: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn upload(&self, texture: &wgpu::Texture, data: &[u8], bytes_per_pixel: usize) {
        let size = texture.size();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bytes_per_pixel as u32),
                rows_per_image: None,
            },
            size,
        );
    }

    /// Submits and waits, then reads `buffer` (mapped whole).
    fn read(&self, encoder: wgpu::CommandEncoder, buffer: &wgpu::Buffer, into: &mut Vec<u8>) -> Result<(), String> {
        self.queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = send.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| format!("the GPU stopped: {e}"))?;
        receive.recv().map_err(|e| e.to_string())?.map_err(|e| format!("reading back from the GPU: {e}"))?;
        {
            let view = buffer.get_mapped_range(..).map_err(|e| format!("reading back from the GPU: {e}"))?;
            into.clear();
            into.extend_from_slice(&view);
        }
        buffer.unmap();
        Ok(())
    }
}

fn buffer_with(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage })
}

fn uniform<T: Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    buffer_with(device, label, bytemuck::bytes_of(value), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
}

fn plain_buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4), usage, mapped_at_creation: false })
}

fn dispatch(pass: &mut wgpu::ComputePass, w: u32, h: u32) {
    pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
}

fn bind(device: &wgpu::Device, pipeline_layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> =
        entries.iter().enumerate().map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: pipeline_layout, entries: &entries })
}

// MARK: - Depth refinement

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GuideParams {
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BilateralParams {
    direction: [f32; 2],
    radius: i32,
    sigma: f32,
    sigma_luma: f32,
    pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EncodeParams {
    range: [f32; 4],
    weights: [f32; 4],
    scales: [f32; 4],
    biases: [f32; 4],
    last: [f32; 4],
}

/// `DepthConverter`'s joint bilateral at model resolution (radius 12, σ 5) and
/// guided ×2 upsample (radius 6, σ 3), both with luma σ 0.06.
const BILATERAL: (i32, f32) = (12, 5.0);
const UPSAMPLE: (i32, f32) = (6, 3.0);
const SIGMA_LUMA: f32 = 0.06;
pub const UPSAMPLE_FACTOR: usize = 2;

/// One conversion's depth textures: the video frame at decode size, the raw
/// map, the refined maps and guides (two of each, the previous frame's kept for
/// the anchor fit) and the ring of five refined maps at twice the model's size.
pub struct DepthRefiner<'g> {
    gpu: &'g Gpu,
    pub depth_size: (usize, usize),
    pub hi_size: (usize, usize),
    video: wgpu::Texture,
    raw: wgpu::Texture,
    scratch: wgpu::Texture,
    mids: [wgpu::Texture; 2],
    guides: [wgpu::Texture; 2],
    hi_guide: wgpu::Texture,
    hi_scratch: wgpu::Texture,
    ring: Vec<wgpu::Texture>,
    guide_params: wgpu::Buffer,
    bilateral_params: [wgpu::Buffer; 4],
    encode_params: wgpu::Buffer,
    lo_readback: wgpu::Buffer,
    lo_row_bytes: usize,
    encoded: wgpu::Buffer,
    encoded_readback: wgpu::Buffer,
    encoded_stride_words: usize,
    bytes: Vec<u8>,
}

/// What `refine` reads back: this frame's refined map and luma guide at model resolution.
pub struct Refined {
    pub map: Vec<f32>,
    pub guide: Vec<f32>,
}

impl<'g> DepthRefiner<'g> {
    pub fn new(gpu: &'g Gpu, decoded: (usize, usize), depth: (usize, usize)) -> DepthRefiner<'g> {
        use wgpu::TextureFormat::{R16Float, Rgba16Float};
        use wgpu::TextureUsages as U;
        let device = &gpu.device;
        let (dw, dh) = depth;
        let hi = (dw * UPSAMPLE_FACTOR, dh * UPSAMPLE_FACTOR);
        let storage = U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC | U::COPY_DST;
        let lo = |label: &str| gpu.texture(label, dw, dh, Rgba16Float, storage, 1);
        let high = |label: &str| gpu.texture(label, hi.0, hi.1, Rgba16Float, storage, 1);
        let (scale, offset) = crate::maths::letterbox(decoded.0, decoded.1, dw, dh);
        let bilateral = |direction: [f32; 2], (radius, sigma): (i32, f32)| {
            uniform(device, "bilateral", &BilateralParams { direction, radius, sigma, sigma_luma: SIGMA_LUMA, pad: [0.0; 3] })
        };
        let lo_row_bytes = (dw * 8).next_multiple_of(256);
        let stride_words = hi.0.div_ceil(4);
        let encoded_size = (stride_words * hi.1 * 4) as u64;
        DepthRefiner {
            video: gpu.texture("video", decoded.0, decoded.1, Rgba16Float, U::TEXTURE_BINDING | U::COPY_DST, 1),
            raw: gpu.texture("raw depth", dw, dh, R16Float, U::TEXTURE_BINDING | U::COPY_DST, 1),
            scratch: lo("scratch"),
            mids: [lo("mid a"), lo("mid b")],
            guides: [lo("guide a"), lo("guide b")],
            hi_guide: high("hi guide"),
            hi_scratch: high("hi scratch"),
            ring: (0..5).map(|_| high("ring")).collect(),
            guide_params: uniform(device, "guide", &GuideParams { uv_scale: scale, uv_offset: offset }),
            bilateral_params: [bilateral([1.0, 0.0], BILATERAL), bilateral([0.0, 1.0], BILATERAL),
                               bilateral([1.0, 0.0], UPSAMPLE), bilateral([0.0, 1.0], UPSAMPLE)],
            encode_params: uniform(device, "encode", &EncodeParams::zeroed()),
            lo_readback: plain_buffer(device, "lo readback", (lo_row_bytes * dh * 2) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            lo_row_bytes,
            encoded: plain_buffer(device, "encoded", encoded_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            encoded_readback: plain_buffer(device, "encoded readback", encoded_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            encoded_stride_words: stride_words,
            depth_size: depth,
            hi_size: hi,
            bytes: Vec::new(),
            gpu,
        }
    }

    fn view(t: &wgpu::Texture) -> wgpu::TextureView {
        t.create_view(&Default::default())
    }

    /// Refines `raw` (model resolution, its own units) with the decoded frame
    /// `video` (RGBA half floats at decode size) as the guide, into ring slot
    /// `slot`, writing model-resolution map and guide pair `pair` (the other
    /// pair keeps the previous inferred frame's). Returns that map and guide,
    /// read back for the anchor fit.
    pub fn refine(&mut self, raw: &[f16], video: &[f16], pair: usize, slot: usize) -> Result<Refined, String> {
        let gpu = self.gpu;
        let device = &gpu.device;
        gpu.upload(&self.raw, bytemuck::cast_slice(raw), 2);
        gpu.upload(&self.video, bytemuck::cast_slice(video), 8);
        let (dw, dh) = self.depth_size;
        let (hw, hh) = self.hi_size;
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            let guide_layout = gpu.guide.get_bind_group_layout(0);
            pass.set_pipeline(&gpu.guide);
            for (target, w, h) in [(&self.guides[pair], dw, dh), (&self.hi_guide, hw, hh)] {
                let group = bind(device, &guide_layout, &[
                    wgpu::BindingResource::TextureView(&Self::view(&self.video)),
                    wgpu::BindingResource::TextureView(&Self::view(target)),
                    wgpu::BindingResource::Sampler(&gpu.linear),
                    self.guide_params.as_entire_binding(),
                ]);
                pass.set_bind_group(0, &group, &[]);
                dispatch(&mut pass, w as u32, h as u32);
            }
            let layout = gpu.bilateral.get_bind_group_layout(0);
            pass.set_pipeline(&gpu.bilateral);
            // (source, guide, target, parameters, size)
            type Step<'t> = (&'t wgpu::Texture, &'t wgpu::Texture, &'t wgpu::Texture, usize, (usize, usize));
            let steps: [Step; 4] = [
                (&self.raw, &self.guides[pair], &self.scratch, 0, (dw, dh)),
                (&self.scratch, &self.guides[pair], &self.mids[pair], 1, (dw, dh)),
                (&self.mids[pair], &self.hi_guide, &self.hi_scratch, 2, (hw, hh)),
                (&self.hi_scratch, &self.hi_guide, &self.ring[slot], 3, (hw, hh)),
            ];
            for (src, guide, dst, params, (w, h)) in steps {
                let group = bind(device, &layout, &[
                    wgpu::BindingResource::TextureView(&Self::view(src)),
                    wgpu::BindingResource::TextureView(&Self::view(guide)),
                    wgpu::BindingResource::TextureView(&Self::view(dst)),
                    wgpu::BindingResource::Sampler(&gpu.linear),
                    self.bilateral_params[params].as_entire_binding(),
                ]);
                pass.set_bind_group(0, &group, &[]);
                dispatch(&mut pass, w as u32, h as u32);
            }
        }
        for (i, texture) in [&self.mids[pair], &self.guides[pair]].into_iter().enumerate() {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.lo_readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: (i * self.lo_row_bytes * dh) as u64,
                        bytes_per_row: Some(self.lo_row_bytes as u32),
                        rows_per_image: Some(dh as u32),
                    },
                },
                wgpu::Extent3d { width: dw as u32, height: dh as u32, depth_or_array_layers: 1 },
            );
        }
        let mut bytes = std::mem::take(&mut self.bytes);
        gpu.read(encoder, &self.lo_readback, &mut bytes)?;
        let plane = |i: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(dw * dh);
            for y in 0..dh {
                let row = &bytes[i * self.lo_row_bytes * dh + y * self.lo_row_bytes..];
                for x in 0..dw {
                    out.push(f16::from_le_bytes([row[x * 8], row[x * 8 + 1]]).to_f32());
                }
            }
            out
        };
        let refined = Refined { map: plane(0), guide: plane(1) };
        self.bytes = bytes;
        Ok(refined)
    }

    /// Ring slot `to` gets slot `from`'s map (a static frame repeats its depth).
    pub fn repeat(&self, from: usize, to: usize) {
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let info = |texture| wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All };
        encoder.copy_texture_to_texture(info(&self.ring[from]), info(&self.ring[to]), self.ring[from].size());
        self.gpu.queue.submit([encoder.finish()]);
    }

    /// The temporally smoothed map for one frame, as 8-bit luma at twice the
    /// model's size (`depthTemporalEncode`). `slots` are the ring slots of the
    /// window's five frames, `weights` normalised, `affines` each slot's
    /// units → the centre's; `lo`/`span` the centre's robust range.
    pub fn encode(&mut self, slots: [usize; 5], weights: [f32; 5], affines: [(f32, f32); 5], lo: f32, span: f32,
                  sigma: f32) -> Result<Vec<u8>, String> {
        let gpu = self.gpu;
        let (hw, hh) = self.hi_size;
        let params = EncodeParams {
            range: [lo, 1.0 / span, 1.0 / (2.0 * sigma * sigma), self.encoded_stride_words as f32],
            weights: [weights[0], weights[1], weights[2], weights[3]],
            scales: [affines[0].0, affines[1].0, affines[2].0, affines[3].0],
            biases: [affines[0].1, affines[1].1, affines[2].1, affines[3].1],
            last: [weights[4], affines[4].0, affines[4].1, hw as f32],
        };
        gpu.queue.write_buffer(&self.encode_params, 0, bytemuck::bytes_of(&params));
        let views: Vec<wgpu::TextureView> = slots.iter().map(|&s| Self::view(&self.ring[s])).collect();
        let layout = gpu.encode.get_bind_group_layout(0);
        let group = bind(&gpu.device, &layout, &[
            wgpu::BindingResource::TextureView(&views[0]),
            wgpu::BindingResource::TextureView(&views[1]),
            wgpu::BindingResource::TextureView(&views[2]),
            wgpu::BindingResource::TextureView(&views[3]),
            wgpu::BindingResource::TextureView(&views[4]),
            wgpu::BindingResource::Sampler(&gpu.linear),
            self.encode_params.as_entire_binding(),
            self.encoded.as_entire_binding(),
        ]);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gpu.encode);
            pass.set_bind_group(0, &group, &[]);
            dispatch(&mut pass, self.encoded_stride_words as u32, hh as u32);
        }
        encoder.copy_buffer_to_buffer(&self.encoded, 0, &self.encoded_readback, 0, self.encoded.size());
        let mut bytes = std::mem::take(&mut self.bytes);
        gpu.read(encoder, &self.encoded_readback, &mut bytes)?;
        let stride = self.encoded_stride_words * 4;
        let mut out = Vec::with_capacity(hw * hh);
        for y in 0..hh {
            out.extend_from_slice(&bytes[y * stride..y * stride + hw]);
        }
        self.bytes = bytes;
        Ok(out)
    }
}

// MARK: - The warp

/// Kr and Kb of a YCbCr matrix ffmpeg names.
pub fn matrix_coefficients(matrix: &str) -> (f32, f32) {
    match matrix {
        "bt2020nc" | "bt2020c" | "bt2020_ncl" | "bt2020_cl" => (0.2627, 0.0593),
        "smpte170m" | "bt470bg" | "bt601" | "fcc" => (0.299, 0.114),
        "smpte240m" => (0.212, 0.087),
        _ => (0.2126, 0.0722),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColourParams {
    k: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshUniforms {
    depth_strength: f32,
    convergence: f32,
    eye_sign: f32,
    value_scale: f32,
    value_bias: f32,
    knee_lo: f32,
    knee_hi: f32,
    pad: f32,
    depth_uv_scale: [f32; 2],
    depth_uv_offset: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    size: [u32; 4],
    words: [u32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct PictureFormat {
    pub eye: (usize, usize),
    /// The packed picture's size (half layouts: the eye's size).
    pub picture: (usize, usize),
    pub side_by_side: bool,
    /// 8 or 10.
    pub bits: u32,
    pub kr_kb: (f32, f32),
    pub full_range: bool,
}

/// One depth frame for the warp: 8-bit values at `size`, mapped to display
/// depth by `value_scale`/`value_bias`, letterboxed by `uv`.
pub struct WarpDepth<'a> {
    pub values: &'a [u8],
    pub size: (usize, usize),
    pub value_scale: f32,
    pub value_bias: f32,
    pub uv: ([f32; 2], [f32; 2]),
}

pub struct Warper<'g> {
    gpu: &'g Gpu,
    format: PictureFormat,
    planes: [wgpu::Texture; 3],
    source: wgpu::Texture,
    depth: Option<wgpu::Texture>,
    msaa: wgpu::Texture,
    z: wgpu::Texture,
    eyes: [wgpu::Texture; 2],
    mesh_uniforms: [wgpu::Buffer; 2],
    colour: wgpu::Buffer,
    pack_params: wgpu::Buffer,
    picture: wgpu::Buffer,
    /// Two, so one frame reads back while the next is drawn.
    readback: [wgpu::Buffer; 2],
    next_slot: usize,
    pending: Option<Pending>,
    luma_words: usize,
    chroma_offset_words: usize,
}

impl<'g> Warper<'g> {
    pub fn new(gpu: &'g Gpu, format: PictureFormat) -> Warper<'g> {
        use wgpu::TextureFormat::{Depth32Float, R8Uint, R16Uint, Rgba16Float};
        use wgpu::TextureUsages as U;
        let device = &gpu.device;
        let (ew, eh) = format.eye;
        let plane_format = if format.bits > 8 { R16Uint } else { R8Uint };
        let plane = |label: &str, w: usize, h: usize| gpu.texture(label, w, h, plane_format, U::TEXTURE_BINDING | U::COPY_DST, 1);
        let (pw, ph) = format.picture;
        // NV12 rows are padded to whole words; P010 rows (2 bytes a sample, even widths) already are.
        let row_bytes = if format.bits > 8 { pw * 2 } else { pw.next_multiple_of(4) };
        let luma_words = row_bytes / 4;
        let picture_bytes = (row_bytes * ph + row_bytes * ph / 2) as u64;
        let colour = ColourParams { k: [format.kr_kb.0, format.kr_kb.1, format.bits as f32, if format.full_range { 1.0 } else { 0.0 }] };
        let mesh_uniform = || uniform(device, "mesh", &MeshUniforms::zeroed());
        Warper {
            planes: [plane("y", ew, eh), plane("u", ew / 2, eh / 2), plane("v", ew / 2, eh / 2)],
            source: gpu.texture("source", ew, eh, Rgba16Float, U::STORAGE_BINDING | U::TEXTURE_BINDING, 1),
            depth: None,
            msaa: gpu.texture("msaa", ew, eh, Rgba16Float, U::RENDER_ATTACHMENT, MSAA),
            z: gpu.texture("z", ew, eh, Depth32Float, U::RENDER_ATTACHMENT, MSAA),
            eyes: [0, 1].map(|_| gpu.texture("eye", ew, eh, Rgba16Float, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING, 1)),
            mesh_uniforms: [mesh_uniform(), mesh_uniform()],
            colour: uniform(device, "colour", &colour),
            pack_params: uniform(device, "pack", &PackParams {
                size: [pw as u32, ph as u32, if format.side_by_side { 0 } else { 1 }, 0],
                words: [format.bits, luma_words as u32, luma_words as u32, (luma_words * ph) as u32],
            }),
            picture: plain_buffer(device, "picture", picture_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            readback: [0, 1].map(|_| plain_buffer(device, "picture readback", picture_bytes,
                                                  wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)),
            next_slot: 0,
            pending: None,
            luma_words,
            chroma_offset_words: luma_words * ph,
            format,
            gpu,
        }
    }

    /// Draws both eyes of one frame (`planes`: Y, Cb, Cr as the decoder wrote
    /// them, 1 or 2 bytes a sample) and packs them. Returns the frame before it,
    /// whose read-back overlapped this one, as NV12 (8-bit) or P010 (10-bit)
    /// without row padding; `finish` returns the last. Without `depth`, both
    /// eyes are the frame.
    pub fn render(&mut self, planes: [&[u8]; 3], depth: Option<&WarpDepth>, strength: f32, convergence: f32)
                  -> Result<Option<Vec<u8>>, String> {
        let gpu = self.gpu;
        let device = &gpu.device;
        let bytes = if self.format.bits > 8 { 2 } else { 1 };
        for (texture, data) in self.planes.iter().zip(planes) {
            gpu.upload(texture, data, bytes);
        }
        let flat_depth = WarpDepth { values: &[0, 0, 0, 0], size: (2, 2), value_scale: 0.0, value_bias: 0.0, uv: ([1.0; 2], [0.0; 2]) };
        let depth = depth.unwrap_or(&flat_depth);
        let strength = if std::ptr::eq(depth, &flat_depth) { 0.0 } else { strength };
        if self.depth.as_ref().is_none_or(|t| t.size().width as usize != depth.size.0 || t.size().height as usize != depth.size.1) {
            self.depth = Some(gpu.texture("depth", depth.size.0, depth.size.1, wgpu::TextureFormat::R8Unorm,
                                          wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 1));
        }
        let depth_texture = self.depth.as_ref().expect("made above");
        gpu.upload(depth_texture, depth.values, 1);
        for (eye, buffer) in self.mesh_uniforms.iter().enumerate() {
            let u = MeshUniforms {
                depth_strength: strength,
                convergence,
                eye_sign: if eye == 0 { 1.0 } else { -1.0 },
                value_scale: depth.value_scale,
                value_bias: depth.value_bias,
                knee_lo: crate::maths::Budget::STUDIO.knee_lo,
                knee_hi: crate::maths::Budget::STUDIO.knee_hi,
                pad: 0.0,
                depth_uv_scale: depth.uv.0,
                depth_uv_offset: depth.uv.1,
            };
            gpu.queue.write_buffer(buffer, 0, bytemuck::bytes_of(&u));
        }
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        let (ew, eh) = self.format.eye;
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            let group = bind(device, &gpu.yuv_to_rgb.get_bind_group_layout(0), &[
                wgpu::BindingResource::TextureView(&view(&self.planes[0])),
                wgpu::BindingResource::TextureView(&view(&self.planes[1])),
                wgpu::BindingResource::TextureView(&view(&self.planes[2])),
                wgpu::BindingResource::TextureView(&view(&self.source)),
                self.colour.as_entire_binding(),
            ]);
            pass.set_pipeline(&gpu.yuv_to_rgb);
            pass.set_bind_group(0, &group, &[]);
            dispatch(&mut pass, ew as u32, eh as u32);
        }
        let (msaa, z) = (view(&self.msaa), view(&self.z));
        for eye in 0..2 {
            let target = view(&self.eyes[eye]);
            let group = bind(device, &gpu.mesh.get_bind_group_layout(0), &[
                self.mesh_uniforms[eye].as_entire_binding(),
                wgpu::BindingResource::TextureView(&view(depth_texture)),
                wgpu::BindingResource::TextureView(&view(&self.source)),
                wgpu::BindingResource::Sampler(&gpu.linear),
            ]);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("eye"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa,
                    depth_slice: None,
                    resolve_target: Some(&target),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &z,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gpu.mesh);
            pass.set_bind_group(0, &group, &[]);
            pass.set_vertex_buffer(0, gpu.grid.0.slice(..));
            pass.set_index_buffer(gpu.grid.1.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.grid.2, 0, 0..1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            let group = bind(device, &gpu.pack.get_bind_group_layout(0), &[
                wgpu::BindingResource::TextureView(&view(&self.eyes[0])),
                wgpu::BindingResource::TextureView(&view(&self.eyes[1])),
                wgpu::BindingResource::Sampler(&gpu.linear),
                self.pack_params.as_entire_binding(),
                self.colour.as_entire_binding(),
                self.picture.as_entire_binding(),
            ]);
            pass.set_pipeline(&gpu.pack);
            pass.set_bind_group(0, &group, &[]);
            let (pw, ph) = self.format.picture;
            let per = if self.format.bits > 8 { 2 } else { 4 };
            dispatch(&mut pass, pw.div_ceil(per) as u32, ph.div_ceil(2) as u32);
        }
        let slot = self.next_slot;
        self.next_slot = 1 - slot;
        encoder.copy_buffer_to_buffer(&self.picture, 0, &self.readback[slot], 0, self.picture.size());
        let submission = gpu.queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::channel();
        self.readback[slot].map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = send.send(r);
        });
        let previous = self.pending.replace(Pending { slot, submission, mapped: receive });
        previous.map(|p| self.collect(p)).transpose()
    }

    /// The last frame `render` drew.
    pub fn finish(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.pending.take().map(|p| self.collect(p)).transpose()
    }

    fn collect(&self, pending: Pending) -> Result<Vec<u8>, String> {
        self.gpu
            .device
            .poll(wgpu::PollType::Wait { submission_index: Some(pending.submission), timeout: None })
            .map_err(|e| format!("the GPU stopped: {e}"))?;
        pending.mapped.recv().map_err(|e| e.to_string())?.map_err(|e| format!("reading back from the GPU: {e}"))?;
        let buffer = &self.readback[pending.slot];
        let mut out;
        {
            let raw = buffer.get_mapped_range(..).map_err(|e| format!("reading back from the GPU: {e}"))?;
            // Strip NV12's row padding.
            let (pw, ph) = self.format.picture;
            let row = if self.format.bits > 8 { pw * 2 } else { pw };
            let stride = self.luma_words * 4;
            out = Vec::with_capacity(row * ph * 3 / 2);
            for y in 0..ph {
                out.extend_from_slice(&raw[y * stride..y * stride + row]);
            }
            let chroma = self.chroma_offset_words * 4;
            for y in 0..ph / 2 {
                out.extend_from_slice(&raw[chroma + y * stride..chroma + y * stride + row]);
            }
        }
        buffer.unmap();
        Ok(out)
    }
}

struct Pending {
    slot: usize,
    submission: wgpu::SubmissionIndex,
    mapped: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}
