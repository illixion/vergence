// The offline depth refinement, ported from VergenceApple's StereoShaders.metal
// (depthGuideLuma, depthJointBilateralH/V, depthTemporalEncode) so the
// converter's depth matches the Mac's: the same taps, sigmas and sampling
// (linear, clamp to edge, at texel centres), every intermediate stored as
// half floats as the Metal path's r16Float textures store it.

struct GuideParams {
    uv_scale: vec2<f32>,
    uv_offset: vec2<f32>,
};

@group(0) @binding(0) var guide_video: texture_2d<f32>;
@group(0) @binding(1) var guide_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var guide_sampler: sampler;
@group(0) @binding(3) var<uniform> guide_params: GuideParams;

// The video frame's luma at depth resolution, in the depth map's letterboxed UV space.
@compute @workgroup_size(16, 16)
fn guide_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(guide_out);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let video_uv = clamp((uv - guide_params.uv_offset) / max(guide_params.uv_scale, vec2<f32>(1e-5)), vec2<f32>(0.0), vec2<f32>(1.0));
    let luma = dot(textureSampleLevel(guide_video, guide_sampler, video_uv, 0.0).rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    textureStore(guide_out, gid.xy, vec4<f32>(luma, 0.0, 0.0, 1.0));
}

struct BilateralParams {
    direction: vec2<f32>,
    radius: i32,
    sigma: f32,
    sigma_luma: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var bl_src: texture_2d<f32>;
@group(0) @binding(1) var bl_guide: texture_2d<f32>;
@group(0) @binding(2) var bl_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var bl_sampler: sampler;
@group(0) @binding(4) var<uniform> bl: BilateralParams;

// One direction of the joint bilateral: spatial gaussian times luma similarity.
@compute @workgroup_size(16, 16)
fn joint_bilateral(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(bl_out);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let fsize = vec2<f32>(size);
    let uv = (vec2<f32>(gid.xy) + 0.5) / fsize;
    let texel_step = bl.direction / fsize;
    var v = textureSampleLevel(bl_src, bl_sampler, uv, 0.0).r;
    if (bl.radius > 0) {
        let inv2ss = 1.0 / (2.0 * bl.sigma * bl.sigma);
        let inv2sl = 1.0 / (2.0 * bl.sigma_luma * bl.sigma_luma);
        let g0 = textureSampleLevel(bl_guide, bl_sampler, uv, 0.0).r;
        var sum = 0.0;
        var wsum = 0.0;
        for (var d = -bl.radius; d <= bl.radius; d = d + 1) {
            let tuv = uv + texel_step * f32(d);
            let dg = textureSampleLevel(bl_guide, bl_sampler, tuv, 0.0).r - g0;
            let w = exp(-f32(d * d) * inv2ss - dg * dg * inv2sl);
            sum = sum + w * textureSampleLevel(bl_src, bl_sampler, tuv, 0.0).r;
            wsum = wsum + w;
        }
        if (wsum > 1e-5) { v = sum / wsum; }
    }
    textureStore(bl_out, gid.xy, vec4<f32>(v, 0.0, 0.0, 1.0));
}

struct EncodeParams {
    // range_lo, range_inv_span, motion_sigma_inv, stride (u32 words a row)
    range: vec4<f32>,
    weights: vec4<f32>,
    scales: vec4<f32>,
    biases: vec4<f32>,
    // slot 4's weight, scale, bias; the output width
    last: vec4<f32>,
};

@group(0) @binding(0) var map0: texture_2d<f32>;
@group(0) @binding(1) var map1: texture_2d<f32>;
@group(0) @binding(2) var map2: texture_2d<f32>;
@group(0) @binding(3) var map3: texture_2d<f32>;
@group(0) @binding(4) var map4: texture_2d<f32>;
@group(0) @binding(5) var enc_sampler: sampler;
@group(0) @binding(6) var<uniform> enc: EncodeParams;
@group(0) @binding(7) var<storage, read_write> enc_out: array<u32>;

fn encode_one(uv: vec2<f32>) -> u32 {
    let center = textureSampleLevel(map2, enc_sampler, uv, 0.0).r;
    var samples = array<f32, 5>(
        textureSampleLevel(map0, enc_sampler, uv, 0.0).r,
        textureSampleLevel(map1, enc_sampler, uv, 0.0).r,
        center,
        textureSampleLevel(map3, enc_sampler, uv, 0.0).r,
        textureSampleLevel(map4, enc_sampler, uv, 0.0).r);
    let weights = array<f32, 5>(enc.weights.x, enc.weights.y, enc.weights.z, enc.weights.w, enc.last.x);
    let scales = array<f32, 5>(enc.scales.x, enc.scales.y, enc.scales.z, enc.scales.w, enc.last.y);
    let biases = array<f32, 5>(enc.biases.x, enc.biases.y, enc.biases.z, enc.biases.w, enc.last.z);
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let d = samples[i] * scales[i] + biases[i];
        let delta = d - center;
        let w = weights[i] * exp(-delta * delta * enc.range.z);
        acc = acc + w * d;
        wsum = wsum + w;
    }
    var v = center;
    if (wsum > 1e-6) { v = acc / wsum; }
    return u32(round(clamp((v - enc.range.x) * enc.range.y, 0.0, 1.0) * 255.0));
}

// The ±2 window, normalised into the centre frame's range, as 8-bit luma: four
// pixels of a row per invocation, one u32.
@compute @workgroup_size(16, 16)
fn temporal_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(map2);
    let width = u32(enc.last.w);
    let stride = u32(enc.range.w);
    if (gid.x >= stride || gid.y >= size.y) { return; }
    var word = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let x = gid.x * 4u + k;
        if (x < width) {
            let uv = (vec2<f32>(f32(x), f32(gid.y)) + 0.5) / vec2<f32>(size);
            word = word | (encode_one(uv) << (8u * k));
        }
    }
    enc_out[gid.y * stride + gid.x] = word;
}
