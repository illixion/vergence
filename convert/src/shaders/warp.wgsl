// Both eyes of a frame from its depth, as VergenceApple draws cached depth
// (StereoShaders.metal: videoStereoMeshVertex / videoStereoMeshFragment on the
// dense 769×433 grid, 4× multisampled, depth-tested), between a YCbCr frame in
// and a packed YCbCr picture out. The picture stays in its own code values
// (an HDR frame's PQ or HLG ones), as the Mac's warp leaves them.

struct Colour {
    // Kr, Kb, bits, full range (1) or limited (0)
    k: vec4<f32>,
};

fn code_max(bits: f32) -> f32 { return exp2(bits) - 1.0; }

fn to_rgb(y: f32, cb: f32, cr: f32, c: Colour) -> vec3<f32> {
    let s = exp2(c.k.z - 8.0);
    var yn: f32;
    var u: f32;
    var v: f32;
    if (c.k.w > 0.5) {
        let m = code_max(c.k.z);
        yn = y / m;
        u = (cb - 128.0 * s) / m;
        v = (cr - 128.0 * s) / m;
    } else {
        yn = (y - 16.0 * s) / (219.0 * s);
        u = (cb - 128.0 * s) / (224.0 * s);
        v = (cr - 128.0 * s) / (224.0 * s);
    }
    let kr = c.k.x;
    let kb = c.k.y;
    let kg = 1.0 - kr - kb;
    let r = yn + 2.0 * (1.0 - kr) * v;
    let b = yn + 2.0 * (1.0 - kb) * u;
    let g = (yn - kr * r - kb * b) / kg;
    return vec3<f32>(r, g, b);
}

// MARK: YCbCr 4:2:0 in → RGB

@group(0) @binding(0) var plane_y: texture_2d<u32>;
@group(0) @binding(1) var plane_u: texture_2d<u32>;
@group(0) @binding(2) var plane_v: texture_2d<u32>;
@group(0) @binding(3) var rgb_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> in_colour: Colour;

fn chroma(plane: texture_2d<u32>, p: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(plane));
    let base = floor(p);
    let f = p - base;
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), size - 1);
    let i1 = clamp(vec2<i32>(base) + 1, vec2<i32>(0), size - 1);
    let a = f32(textureLoad(plane, vec2<i32>(i0.x, i0.y), 0).r);
    let b = f32(textureLoad(plane, vec2<i32>(i1.x, i0.y), 0).r);
    let c = f32(textureLoad(plane, vec2<i32>(i0.x, i1.y), 0).r);
    let d = f32(textureLoad(plane, vec2<i32>(i1.x, i1.y), 0).r);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Chroma sited left of its two luma columns and between its two rows (MPEG-2,
// H.264 and HEVC's default), interpolated bilinearly.
@compute @workgroup_size(16, 16)
fn yuv_to_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(rgb_out);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let y = f32(textureLoad(plane_y, vec2<i32>(gid.xy), 0).r);
    let p = vec2<f32>(f32(gid.x) * 0.5, (f32(gid.y) - 0.5) * 0.5);
    let rgb = to_rgb(y, chroma(plane_u, p), chroma(plane_v, p), in_colour);
    textureStore(rgb_out, gid.xy, vec4<f32>(rgb, 1.0));
}

// MARK: The mesh

struct Uniforms {
    depth_strength: f32,
    convergence: f32,
    eye_sign: f32,
    value_scale: f32,
    value_bias: f32,
    knee_lo: f32,
    knee_hi: f32,
    _pad: f32,
    depth_uv_scale: vec2<f32>,
    depth_uv_offset: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var source_tex: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;

fn depth_budget(y: f32) -> f32 {
    if (y > u.knee_hi) {
        let w = 1.0 - u.knee_hi;
        if (w > 0.0) { return u.knee_hi + w * tanh((y - u.knee_hi) / w); }
        return min(y, 1.0);
    }
    if (y < u.knee_lo) {
        let w = u.knee_lo;
        if (w > 0.0) { return u.knee_lo - w * tanh((u.knee_lo - y) / w); }
        return max(y, 0.0);
    }
    return y;
}

fn edge_feather(x: f32) -> f32 {
    let feather = 0.05;
    return smoothstep(0.0, feather, x) * (1.0 - smoothstep(1.0 - feather, 1.0, x));
}

struct MeshOut {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn mesh_vertex(@location(0) grid: vec2<f32>) -> MeshOut {
    let duv = grid * u.depth_uv_scale + u.depth_uv_offset;
    let depth = depth_budget(textureSampleLevel(depth_tex, linear_sampler, duv, 0.0).r * u.value_scale + u.value_bias);
    let disparity = (depth - u.convergence) * u.depth_strength * edge_feather(grid.x);
    var out: MeshOut;
    out.position = vec4<f32>(grid.x * 2.0 - 1.0 + u.eye_sign * disparity * 2.0, 1.0 - grid.y * 2.0,
                             clamp(1.0 - depth, 0.0, 1.0), 1.0);
    out.tex_coord = grid;
    return out;
}

@fragment
fn mesh_fragment(in: MeshOut) -> @location(0) vec4<f32> {
    let colour = textureSample(source_tex, linear_sampler, in.tex_coord);
    return vec4<f32>(clamp(colour.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// MARK: Both eyes → a packed YCbCr picture

struct Pack {
    // picture width, height, layout (0 SBS, 1 top and bottom), half (1) or full (0)
    size: vec4<u32>,
    // bits, u32 words a luma row, u32 words a chroma row, offset of the chroma plane in words
    words: vec4<u32>,
};

@group(0) @binding(0) var left_eye: texture_2d<f32>;
@group(0) @binding(1) var right_eye: texture_2d<f32>;
@group(0) @binding(2) var pack_sampler: sampler;
@group(0) @binding(3) var<uniform> pack: Pack;
@group(0) @binding(4) var<uniform> out_colour: Colour;
@group(0) @binding(5) var<storage, read_write> picture: array<u32>;

// The picture's pixel x, y: which eye and where in it. A half layout's eye is
// squeezed by sampling between two texels (their average).
fn picture_rgb(px: i32, py: i32) -> vec3<f32> {
    let w = i32(pack.size.x);
    let h = i32(pack.size.y);
    let x = clamp(px, 0, w - 1);
    let y = clamp(py, 0, h - 1);
    var eye = 0;
    var local = vec2<f32>(f32(x), f32(y));
    var span = vec2<f32>(f32(w), f32(h));
    if (pack.size.z == 0u) {
        let half_w = f32(w) * 0.5;
        if (f32(x) >= half_w) { eye = 1; local.x = f32(x) - half_w; }
        span.x = half_w;
    } else {
        let half_h = f32(h) * 0.5;
        if (f32(y) >= half_h) { eye = 1; local.y = f32(y) - half_h; }
        span.y = half_h;
    }
    let uv = (local + 0.5) / span;
    if (eye == 0) { return textureSampleLevel(left_eye, pack_sampler, uv, 0.0).rgb; }
    return textureSampleLevel(right_eye, pack_sampler, uv, 0.0).rgb;
}

fn luma_code(rgb: vec3<f32>, c: Colour) -> u32 {
    let kr = c.k.x;
    let kb = c.k.y;
    let yn = kr * rgb.r + (1.0 - kr - kb) * rgb.g + kb * rgb.b;
    let s = exp2(c.k.z - 8.0);
    var code: f32;
    if (c.k.w > 0.5) { code = yn * code_max(c.k.z); } else { code = yn * 219.0 * s + 16.0 * s; }
    return u32(clamp(round(code), 0.0, code_max(c.k.z)));
}

fn chroma_codes(rgb: vec3<f32>, c: Colour) -> vec2<u32> {
    let kr = c.k.x;
    let kb = c.k.y;
    let yn = kr * rgb.r + (1.0 - kr - kb) * rgb.g + kb * rgb.b;
    let u = (rgb.b - yn) / (2.0 * (1.0 - kb));
    let v = (rgb.r - yn) / (2.0 * (1.0 - kr));
    let s = exp2(c.k.z - 8.0);
    var scale = 224.0 * s;
    if (c.k.w > 0.5) { scale = code_max(c.k.z); }
    let m = code_max(c.k.z);
    return vec2<u32>(u32(clamp(round(u * scale + 128.0 * s), 0.0, m)), u32(clamp(round(v * scale + 128.0 * s), 0.0, m)));
}

// One chroma site per invocation: its 2×2 luma and its Cb/Cr, sited like the
// input (left column, between the rows: [¼ ½ ¼] across, both rows down).
// 8-bit output is NV12 (four luma bytes a word: an invocation writes two
// luma words' halves, so it covers 4×2 pixels); 10-bit is P010.
@compute @workgroup_size(16, 16)
fn pack_picture(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bits = pack.words.x;
    let per = select(2u, 4u, bits == 8u); // luma pixels an invocation covers across
    let x0 = gid.x * per;
    let y0 = gid.y * 2u;
    if (x0 >= pack.size.x || y0 >= pack.size.y) { return; }
    let luma_words = pack.words.y;
    let chroma_words = pack.words.z;
    let chroma_base = pack.words.w;
    for (var row = 0u; row < 2u; row = row + 1u) {
        var word = 0u;
        for (var k = 0u; k < per; k = k + 1u) {
            let x = x0 + k;
            if (x < pack.size.x) {
                let code = luma_code(picture_rgb(i32(x), i32(y0 + row)), out_colour);
                if (bits == 8u) { word = word | (code << (8u * k)); } else { word = word | ((code << 6u) << (16u * k)); }
            }
        }
        if (y0 + row < pack.size.y) {
            picture[(y0 + row) * luma_words + (x0 * select(2u, 1u, bits == 8u)) / 4u] = word;
        }
    }
    var cword = 0u;
    for (var k = 0u; k < per / 2u; k = k + 1u) {
        let cx = i32(x0 + 2u * k);
        if (u32(cx) < pack.size.x) {
            let yy = i32(y0);
            let rgb = 0.25 * (picture_rgb(cx - 1, yy) + picture_rgb(cx - 1, yy + 1)) * 0.5
                    + 0.5 * (picture_rgb(cx, yy) + picture_rgb(cx, yy + 1)) * 0.5
                    + 0.25 * (picture_rgb(cx + 1, yy) + picture_rgb(cx + 1, yy + 1)) * 0.5;
            let c = chroma_codes(rgb, out_colour);
            if (bits == 8u) {
                cword = cword | (c.x << (16u * k)) | (c.y << (16u * k + 8u));
            } else {
                cword = (c.x << 6u) | ((c.y << 6u) << 16u);
            }
        }
    }
    picture[chroma_base + gid.y * chroma_words + (x0 * select(2u, 1u, bits == 8u)) / 4u] = cword;
}
