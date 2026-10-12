/*
 RAVEMedia - Stereo Warp and Depth Shaders

 The pseudo-3D half of what used to be Spatial Stash's single Shaders.metal:
 the per-eye warp (backward and depth-displaced-mesh), realtime depth
 stabilization, and the offline conversion kernels. The image-display shaders
 (RCAS, FXAA, tonal) stayed in the app — they serve the photo path and have
 nothing to do with video depth.

 Compiled into the package's own default library, reached via
 `device.makeDefaultLibrary(bundle: .module)`. The app's `makeDefaultLibrary()`
 sees only the app's own functions, so the two never collide.

 `stereoQuadVertex` is the app's `imageVertexShader` carried over verbatim (a
 procedural fullscreen quad, UVs flipped vertically so a Metal top-left origin
 reads a bottom-left-origin source). It is duplicated rather than shared
 because a Metal library cannot span module boundaries; keep the two in sync
 only in the sense that both must keep producing a top-left-origin quad.
 */

#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Fullscreen quad: 6 vertices (2 triangles), no vertex buffer needed.
// Uses vertex_id to generate positions and UVs procedurally.
vertex VertexOut stereoQuadVertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {
        float2(-1, -1), // bottom-left
        float2( 1, -1), // bottom-right
        float2(-1,  1), // top-left
        float2(-1,  1), // top-left
        float2( 1, -1), // bottom-right
        float2( 1,  1)  // top-right
    };

    const float2 texCoords[] = {
        float2(0, 1), // bottom-left (UV flipped vertically)
        float2(1, 1), // bottom-right
        float2(0, 0), // top-left
        float2(0, 0), // top-left
        float2(1, 1), // bottom-right
        float2(1, 0)  // top-right
    };

    VertexOut out;
    out.position = float4(positions[vertexID], 0, 1);
    out.texCoord = texCoords[vertexID];
    return out;
}

// MARK: - Pseudo-3D Eye Warp
//
// Real-time "fake 3D": synthesize one eye of a stereo pair from a single mono
// frame, with no pre-compute. A cheap heuristic depth estimate (lower-of-frame
// = nearer, brighter = nearer, edges = slightly nearer) drives a horizontal
// parallax shift. Rendered once per eye into a full-frame eye texture; the eye
// is selected by `eyeSign` (+1 left, -1 right). Brightness/contrast/saturation
// are folded in here so the windowed RealityKit VideoPlayerComponent path keeps
// the same look as the flat MetalKit player.
//
// The depth heuristic is intentionally isolated in `pseudo3DHeuristicDepth` so a
// future Core ML depth provider can replace it (sample a supplied depth texture
// instead) without touching the warp/tonal math.

struct VideoStereoUniforms {
    float brightness;
    float contrast;
    float saturation;
    float depthStrength;  // max horizontal disparity in UV (fraction of width)
    float convergence;    // depth put on the window plane. Inverse depth, so 1
                          // is NEAR: raising this pushes the scene BACK (at 1.0
                          // nothing crosses in front of the window frame).
    float eyeSign;        // +1 = left eye, -1 = right eye
    float mirror;         // 1 = mirror horizontally (flip), 0 = normal
    float useDepth;       // 1 = sample real depth from depthTex, 0 = heuristic
    // Maps frame UV into the content region of the depth map. Vision letterboxes
    // the frame into the model's square input (.scaleFit), so the depth for
    // frame UV lives at uv * scale + offset; identity when aspects match.
    float2 depthUVScale;
    float2 depthUVOffset;
    // Maps the stored depth sample to display-normalized depth (near≈1):
    // d = sample * valueScale + valueBias. Identity (1, 0) for realtime depth;
    // cached depth bakes each frame's encode range + the lookahead-smoothed
    // display range into this per-frame affine.
    float depthValueScale;
    float depthValueBias;
    // Depth budget knees (see DepthBudget): identity between them, tanh toe
    // and shoulder outside. (0, 1) is the old clamp.
    float2 depthBudget;
    // 1 = fill the strips a near subject uncovers with the background beside
    // them (mesh warp only), 0 = stretch the image across them as before.
    float edgeFill;
    float pad0;
};

// The depth budget: shot-space depth (unbounded once anchored) → 0…1, near
// content plateauing short of 1 instead of clipping. Mirrors DepthBudget.apply.
static inline float pseudo3DDepthBudget(float y, float2 knees) {
    if (y > knees.y) {
        const float w = 1.0 - knees.y;
        return w > 0.0 ? knees.y + w * precise::tanh((y - knees.y) / w) : min(y, 1.0);
    }
    if (y < knees.x) {
        const float w = knees.x;
        return w > 0.0 ? knees.x - w * precise::tanh((knees.x - y) / w) : max(y, 0.0);
    }
    return y;
}

// Stereo-window guard: feather disparity to zero over the outer few percent of
// frame width. Without it, near (popped-out) content cut by the frame edge is a
// classic window violation, and the warp's clamp_to_edge sampling paints
// per-eye-different strips along the borders — both read as binocular rivalry
// and get markedly worse at higher depth strengths. Applied in display space
// (pre-mirror), symmetric so the flip doesn't matter.
static inline float pseudo3DEdgeFeather(float x) {
    const float feather = 0.05;
    return smoothstep(0.0, feather, x) * (1.0 - smoothstep(1.0 - feather, 1.0, x));
}

static inline float pseudo3DHeuristicDepth(texture2d<float> tex, sampler s, float2 uv) {
    float2 px = float2(1.0) / float2(tex.get_width(0), tex.get_height(0));
    float3 c  = tex.sample(s, uv).rgb;
    float l   = dot(c, float3(0.2126, 0.7152, 0.0722));
    float lL  = dot(tex.sample(s, uv + float2(-px.x, 0.0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float lR  = dot(tex.sample(s, uv + float2( px.x, 0.0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float lU  = dot(tex.sample(s, uv + float2(0.0, -px.y)).rgb, float3(0.2126, 0.7152, 0.0722));
    float lD  = dot(tex.sample(s, uv + float2(0.0,  px.y)).rgb, float3(0.2126, 0.7152, 0.0722));
    float edge       = saturate((abs(lR - lL) + abs(lD - lU)) * 1.2);
    float lowerNear  = smoothstep(0.10, 0.95, uv.y); // uv.y=1 is bottom of frame
    float brightNear = smoothstep(0.25, 0.85, l);
    return saturate(lowerNear * 0.70 + brightNear * 0.25 + edge * 0.05);
}

fragment float4 videoPseudo3DEyeFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    texture2d<float> depthTex [[texture(1)]],
    constant VideoStereoUniforms &u [[buffer(0)]]
) {
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    float2 uv = in.texCoord;
    if (u.mirror > 0.5) { uv.x = 1.0 - uv.x; }

    // Real Core ML depth when available (normalized inverse depth: near≈1,
    // far≈0), otherwise the cheap heuristic. (The useDepth branch is currently
    // unreachable — a real depth map always takes the mesh-warp path — but is
    // kept consistent with videoStereoMeshVertex's letterbox remap.)
    float depth = u.useDepth > 0.5
        ? pseudo3DDepthBudget(depthTex.sample(texSampler, uv * u.depthUVScale + u.depthUVOffset).r
                              * u.depthValueScale + u.depthValueBias, u.depthBudget)
        : pseudo3DHeuristicDepth(tex, texSampler, uv);
    // Near objects (depth high) must get CROSSED disparity to read as "in front":
    // the left eye sees them shifted right, the right eye left. That means
    // sampling the source toward the OPPOSITE side of the shift — hence the
    // minus. (The previous `+` gave uncrossed disparity, pushing near objects
    // behind = inverted/weak depth that won't fuse.)
    float disparity = (depth - u.convergence) * u.depthStrength
                      * pseudo3DEdgeFeather(in.texCoord.x);
    float2 warpedUV = float2(uv.x - u.eyeSign * disparity, uv.y);

    float4 color = tex.sample(texSampler, warpedUV);

    color.rgb += u.brightness;
    color.rgb = (color.rgb - 0.5) * u.contrast + 0.5;
    float luminance = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    color.rgb = mix(float3(luminance), color.rgb, u.saturation);
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    color.a = 1.0;
    return color;
}

// MARK: - Packed Stereo Eye
//
// Real stereo, no synthesis: a side-by-side or over/under video already holds
// both eyes, so each eye is a crop of the decoded frame, stretched to the eye
// buffer (which also undoes a half-resolution squeeze). `region` places the
// eye in the frame, origin top left: sample = uv * scale + offset. Mirroring
// flips within the eye; the CPU side swaps which eye is sampled, since a
// mirrored pair is only a pair with the eyes exchanged. Tonal adjustments
// match the fake-3D warp so the two modes look the same.

struct PackedEyeUniforms {
    float brightness;
    float contrast;
    float saturation;
    float mirror;
    float2 regionScale;
    float2 regionOffset;
};

fragment float4 videoPackedEyeFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    constant PackedEyeUniforms &u [[buffer(0)]]
) {
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    float2 uv = in.texCoord;
    if (u.mirror > 0.5) { uv.x = 1.0 - uv.x; }
    // Keep the bilinear footprint inside this eye's half, or the seam column
    // blends in the other eye's edge pixels.
    float2 halfTexel = 0.5 / float2(tex.get_width(0), tex.get_height(0));
    float2 lo = u.regionOffset + halfTexel;
    float2 hi = u.regionOffset + u.regionScale - halfTexel;
    float2 sampleUV = clamp(uv * u.regionScale + u.regionOffset, lo, hi);

    float4 color = tex.sample(texSampler, sampleUV);
    color.rgb += u.brightness;
    color.rgb = (color.rgb - 0.5) * u.contrast + 0.5;
    float luminance = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    color.rgb = mix(float3(luminance), color.rgb, u.saturation);
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    color.a = 1.0;
    return color;
}

// MARK: - Pseudo-3D Mesh Warp (depth-displaced grid, occlusion-correct)
//
// When a real depth map is available, render the frame as a displaced grid
// instead of a per-pixel backward warp. Each vertex shifts horizontally by its
// disparity (parallax) and takes a clip-space Z from its depth; with depth
// testing, nearer geometry occludes farther geometry, so foreground/background
// boundaries stay clean instead of smearing into halos. Disocclusions become
// stretched triangles rather than edge smears. Reuses VideoStereoUniforms.

struct MeshVertexOut {
    float4 position [[position]];
    float2 texCoord;
    // Screen distance per unit of source distance across the cell: 1 on the
    // plain picture, above 1 where the mesh stretches over an uncovered strip,
    // below 0 where it folds back over a subject (see meshEdgeFill).
    float stretch;
};

// The disparity at a grid point, as the vertex shader shifts it.
static inline float meshDisparity(float2 grid, constant VideoStereoUniforms &u, texture2d<float> depthTex, sampler s) {
    float2 uv = grid;
    if (u.mirror > 0.5) { uv.x = 1.0 - uv.x; }
    const float2 duv = uv * u.depthUVScale + u.depthUVOffset;
    const float depth = pseudo3DDepthBudget(depthTex.sample(s, duv, level(0)).r
                                            * u.depthValueScale + u.depthValueBias, u.depthBudget);
    return (depth - u.convergence) * u.depthStrength * pseudo3DEdgeFeather(grid.x);
}

vertex MeshVertexOut videoStereoMeshVertex(
    uint vid [[vertex_id]],
    const device float2 *gridPositions [[buffer(0)]],
    constant VideoStereoUniforms &u [[buffer(1)]],
    texture2d<float> depthTex [[texture(0)]]
) {
    constexpr sampler depthSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 grid = gridPositions[vid];           // [0,1] across the frame
    float2 uv = grid;
    if (u.mirror > 0.5) { uv.x = 1.0 - uv.x; }

    // uv already indexes the source pixel being displayed (post-mirror), and
    // depth was inferred on the unmirrored frame, so the letterbox remap applies
    // after the flip.
    float2 duv = uv * u.depthUVScale + u.depthUVOffset;
    // near≈1 after the per-frame value mapping and the depth budget
    float depth = pseudo3DDepthBudget(depthTex.sample(depthSampler, duv, level(0)).r
                                      * u.depthValueScale + u.depthValueBias, u.depthBudget);
    float disparity = (depth - u.convergence) * u.depthStrength
                      * pseudo3DEdgeFeather(grid.x);

    float ndcX = grid.x * 2.0 - 1.0 + u.eyeSign * disparity * 2.0;
    float ndcY = 1.0 - grid.y * 2.0;
    float ndcZ = clamp(1.0 - depth, 0.0, 1.0);   // near (depth 1) → 0 → wins depth test

    MeshVertexOut out;
    out.position = float4(ndcX, ndcY, ndcZ, 1.0);
    out.texCoord = uv;
    // Measured per vertex from the depth either side and interpolated across
    // the cell, so it varies smoothly; the fragment's own derivative of its
    // source coordinate flickered per 2×2 pixel block along an outline, which
    // drew the fill as spikes (headset, 2026-10-11).
    const float h = 1.0 / 769.0;
    const float left = meshDisparity(float2(max(grid.x - h, 0.0), grid.y), u, depthTex, depthSampler);
    const float right = meshDisparity(float2(min(grid.x + h, 1.0), grid.y), u, depthTex, depthSampler);
    out.stretch = 1.0 + u.eyeSign * (right - left) / (2.0 * h);
    return out;
}

// Where a near subject's outline moves off the background, the mesh's cells
// across the outline stretch over the strip of background that comes into
// view, smearing the outline's own pixels across it. Such a fragment is
// recognised by its cell's stretch (MeshVertexOut.stretch); it is given the source pixel on the far side of the outline instead (whichever
// neighbour is farther), so the strip repeats the background beside it.
static inline float2 meshEdgeFill(
    float2 tc, float stretch, texture2d<float> depthTex, sampler s, constant VideoStereoUniforms &u
) {
    // Only stretched cells. Cells folded back over a subject (stretch below 0:
    // the background shifting onto it) are left alone: their fragments
    // z-fight the subject's own, and painting them background bit pieces out
    // of its outline (an ear, on the headset).
    // From 1.2×: with outlines snapped sharp, a cell across one is stretched
    // a little or a lot, and below 1.4× (the first threshold) the picture
    // still smeared across it, which read as every edge looking stretched
    // (Nope, headset 2026-10-12). A gentle depth slope stretches cells by a
    // few per cent at most.
    const float fill = smoothstep(1.2, 1.8, stretch);
    if (fill <= 0.0) { return tc; }
    // Two dense cells either side: past the outline whatever side it is on.
    const float reach = 2.0 / 769.0;
    const float2 leftUV = float2(max(tc.x - reach, 0.0), tc.y);
    const float2 rightUV = float2(min(tc.x + reach, 1.0), tc.y);
    // Depth was inferred on the unmirrored frame (as in the vertex shader).
    float2 dl = leftUV, dr = rightUV;
    if (u.mirror > 0.5) { dl.x = 1.0 - dl.x; dr.x = 1.0 - dr.x; }
    const float depthLeft = depthTex.sample(s, dl * u.depthUVScale + u.depthUVOffset, level(0)).r;
    const float depthRight = depthTex.sample(s, dr * u.depthUVScale + u.depthUVOffset, level(0)).r;
    const float2 far = depthLeft < depthRight ? leftUV : rightUV;
    return mix(tc, far, fill);
}

fragment float4 videoStereoMeshFragment(
    MeshVertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    texture2d<float> depthTex [[texture(1)]],
    constant VideoStereoUniforms &u [[buffer(0)]]
) {
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 tc = u.edgeFill > 0.5
        ? meshEdgeFill(in.texCoord, in.stretch, depthTex, texSampler, u)
        : in.texCoord;
    float4 color = tex.sample(texSampler, tc);
    color.rgb += u.brightness;
    color.rgb = (color.rgb - 0.5) * u.contrast + 0.5;
    float luminance = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    color.rgb = mix(float3(luminance), color.rgb, u.saturation);
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    color.a = 1.0;
    return color;
}

// MARK: - Depth Stabilization (spatial band-limit + motion-adaptive temporal EMA)
//
// Two depth artifacts feed the warp:
//  1. Spatial: the depth model (esp. Base) carries detail finer than the warp
//     grid can sample, so the grid *aliases* it into rippling standing waves
//     across the frame. A gaussian low-pass band-limits the depth to the grid's
//     sampling rate, removing the ripples (at the cost of slightly softer
//     silhouettes — the same trade the smoother low-res depth used to make).
//  2. Temporal: monocular depth jitters frame-to-frame, wobbling subjects. A
//     motion-adaptive EMA smooths hard where depth is stable (kills flicker) but
//     trusts the new map where it changed a lot (no ghosting on motion).
// Both run at the depth's native resolution.

struct DepthStabilizeParams {
    int   blurRadius;  // spatial low-pass radius in depth texels (0 = no blur)
    float blurSigma;   // gaussian sigma for the low-pass
    float baseAlpha;   // temporal blend toward new map where depth is stable
    float motionGain;  // how fast the blend trusts the new map as depth changes
    uint  hasPrev;     // 1 when emaPrev holds a valid previous frame
    float sigmaLuma;   // joint-bilateral luma sigma (bilateral kernels only)
};

// The gaussian is separable, so it runs as two 1D passes (H into a scratch
// texture, then V) — 2·(2r+1) taps per texel instead of (2r+1)², a 6.5× tap
// reduction at the default radius 6. The EMA rides the tail of the V pass so
// the whole stabilization is still one command buffer.

static inline float depthGaussian1D(
    texture2d<float, access::sample> src, sampler s,
    float2 uv, float2 texelStep, constant DepthStabilizeParams &p
) {
    if (p.blurRadius <= 0) { return src.sample(s, uv).r; }
    const float inv2s2 = 1.0 / (2.0 * p.blurSigma * p.blurSigma);
    float sum = 0.0, wsum = 0.0;
    for (int d = -p.blurRadius; d <= p.blurRadius; ++d) {
        const float w = exp(-float(d * d) * inv2s2);
        sum += w * src.sample(s, uv + texelStep * float(d)).r;
        wsum += w;
    }
    return wsum > 1e-5 ? (sum / wsum) : src.sample(s, uv).r;
}

kernel void depthBlurH(
    texture2d<float, access::sample> rawDepth [[texture(0)]],
    texture2d<float, access::write>  blurred  [[texture(1)]],
    constant DepthStabilizeParams &p          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint W = blurred.get_width();
    const uint H = blurred.get_height();
    if (gid.x >= W || gid.y >= H) { return; }

    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 size = float2(W, H);
    const float2 uv = (float2(gid) + 0.5) / size;
    const float raw = depthGaussian1D(rawDepth, s, uv, float2(1.0 / size.x, 0.0), p);
    blurred.write(float4(raw, 0.0, 0.0, 1.0), gid);
}

kernel void depthBlurVEMA(
    texture2d<float, access::sample> blurredH [[texture(0)]],
    texture2d<float, access::sample> emaPrev  [[texture(1)]],
    texture2d<float, access::write>  emaNext  [[texture(2)]],
    constant DepthStabilizeParams &p          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint W = emaNext.get_width();
    const uint H = emaNext.get_height();
    if (gid.x >= W || gid.y >= H) { return; }

    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 size = float2(W, H);
    const float2 uv = (float2(gid) + 0.5) / size;
    const float raw = depthGaussian1D(blurredH, s, uv, float2(0.0, 1.0 / size.y), p);

    float out = raw;
    if (p.hasPrev != 0) {
        const float prev = emaPrev.sample(s, uv).r;
        const float alpha = clamp(p.baseAlpha + p.motionGain * abs(raw - prev), p.baseAlpha, 1.0);
        out = mix(prev, raw, alpha);
    }
    emaNext.write(float4(out, 0.0, 0.0, 1.0), gid);
}

// MARK: - Edge-Aware Depth Refinement (joint bilateral, offline conversion)
//
// Monocular depth edges rarely align exactly with image edges, and a plain
// gaussian band-limit smears them further — visible as halos around crisp
// silhouettes (worst on synthetic/CG content). The joint bilateral filter
// weights each spatial tap by luma similarity to the center pixel, so depth
// smooths within regions but snaps to image edges. Run as two 1D passes — an
// approximation of the true 2D bilateral, standard for this use.

struct DepthGuideParams {
    // Letterbox transform: guide texel UV → source-video UV (the inverse of the
    // warp's depthUVScale/Offset), so the guide aligns with the depth map.
    float2 uvScale;
    float2 uvOffset;
};

/// Downsample the video frame into a luma guide at depth resolution, laid out
/// in the depth map's (letterboxed) UV space.
kernel void depthGuideLuma(
    texture2d<float, access::sample> video [[texture(0)]],
    texture2d<float, access::write>  guide [[texture(1)]],
    constant DepthGuideParams &p           [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint W = guide.get_width();
    const uint H = guide.get_height();
    if (gid.x >= W || gid.y >= H) { return; }
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 uv = (float2(gid) + 0.5) / float2(W, H);
    const float2 videoUV = clamp((uv - p.uvOffset) / max(p.uvScale, float2(1e-5)), 0.0, 1.0);
    const float luma = dot(video.sample(s, videoUV).rgb, float3(0.2126, 0.7152, 0.0722));
    guide.write(float4(luma, 0.0, 0.0, 1.0), gid);
}

static inline float depthJointBilateral1D(
    texture2d<float, access::sample> src,
    texture2d<float, access::sample> guide,
    sampler s, float2 uv, float2 texelStep, constant DepthStabilizeParams &p
) {
    if (p.blurRadius <= 0) { return src.sample(s, uv).r; }
    const float inv2ss = 1.0 / (2.0 * p.blurSigma * p.blurSigma);
    const float inv2sl = 1.0 / (2.0 * p.sigmaLuma * p.sigmaLuma);
    const float g0 = guide.sample(s, uv).r;
    float sum = 0.0, wsum = 0.0;
    for (int d = -p.blurRadius; d <= p.blurRadius; ++d) {
        const float2 tuv = uv + texelStep * float(d);
        const float dg = guide.sample(s, tuv).r - g0;
        const float w = exp(-float(d * d) * inv2ss - dg * dg * inv2sl);
        sum += w * src.sample(s, tuv).r;
        wsum += w;
    }
    return wsum > 1e-5 ? (sum / wsum) : src.sample(s, uv).r;
}

kernel void depthJointBilateralH(
    texture2d<float, access::sample> rawDepth [[texture(0)]],
    texture2d<float, access::sample> guide    [[texture(1)]],
    texture2d<float, access::write>  dst      [[texture(2)]],
    constant DepthStabilizeParams &p          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint W = dst.get_width();
    const uint H = dst.get_height();
    if (gid.x >= W || gid.y >= H) { return; }
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 size = float2(W, H);
    const float2 uv = (float2(gid) + 0.5) / size;
    const float v = depthJointBilateral1D(rawDepth, guide, s, uv, float2(1.0 / size.x, 0.0), p);
    dst.write(float4(v, 0.0, 0.0, 1.0), gid);
}

kernel void depthJointBilateralV(
    texture2d<float, access::sample> blurredH [[texture(0)]],
    texture2d<float, access::sample> guide    [[texture(1)]],
    texture2d<float, access::write>  dst      [[texture(2)]],
    constant DepthStabilizeParams &p          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint W = dst.get_width();
    const uint H = dst.get_height();
    if (gid.x >= W || gid.y >= H) { return; }
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 size = float2(W, H);
    const float2 uv = (float2(gid) + 0.5) / size;
    const float v = depthJointBilateral1D(blurredH, guide, s, uv, float2(0.0, 1.0 / size.y), p);
    dst.write(float4(v, 0.0, 0.0, 1.0), gid);
}

// MARK: - Offline Depth Conversion (per-frame stats + temporal encode)
//
// The offline "Convert to 3D" pre-processing pass computes robust per-frame
// depth statistics on the GPU (min/max reduction feeding a 256-bin histogram
// for percentiles/median), then encodes a lookahead-smoothed depth frame into
// the cache video's luma plane. Raw model output has an arbitrary per-frame
// scale, so all stats are in raw units; DepthConverter turns them into a
// temporally stable display mapping in its post-pass.

/// One (min, max) pair per 16×16 threadgroup; the CPU folds the partials.
/// Must be dispatched with threadsPerThreadgroup = 16×16 (256 threads).
kernel void depthMinMaxReduce(
    texture2d<float, access::read> src [[texture(0)]],
    device float2 *partials            [[buffer(0)]],
    uint2 gid  [[thread_position_in_grid]],
    uint  tix  [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgPerGrid [[threadgroups_per_grid]]
) {
    threadgroup float sMin[256];
    threadgroup float sMax[256];
    float lo = INFINITY;
    float hi = -INFINITY;
    if (gid.x < src.get_width() && gid.y < src.get_height()) {
        const float d = src.read(gid).r;
        if (isfinite(d)) { lo = d; hi = d; }
    }
    sMin[tix] = lo;
    sMax[tix] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tix < stride) {
            sMin[tix] = min(sMin[tix], sMin[tix + stride]);
            sMax[tix] = max(sMax[tix], sMax[tix + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tix == 0) {
        partials[tgid.y * tgPerGrid.x + tgid.x] = float2(sMin[0], sMax[0]);
    }
}

struct DepthHistogramParams {
    float lo;      // histogram range start (raw units)
    float invSpan; // 1 / (hi - lo)
};

kernel void depthHistogram256(
    texture2d<float, access::read> src [[texture(0)]],
    device atomic_uint *bins           [[buffer(0)]],
    constant DepthHistogramParams &p   [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) { return; }
    const float d = src.read(gid).r;
    if (!isfinite(d)) { return; }
    const uint bin = uint(clamp((d - p.lo) * p.invSpan, 0.0f, 0.999999f) * 256.0f);
    atomic_fetch_add_explicit(&bins[bin], 1u, memory_order_relaxed);
}

struct DepthEncodeParams {
    float rangeLo;        // center frame's robust lo (raw units)
    float rangeInvSpan;   // 1 / (hi - lo)
    float weights[5];     // temporal window weights, normalized; 0 for unused slots
    float motionSigmaInv; // depth-similarity gate: 1/(2σ²), raw units; 0 = ungated
    float slotScale[5];   // each slot's raw units → the center frame's (anchoring)
    float slotBias[5];
};

/// Weighted temporal average of up to 5 refined raw-depth maps (the center
/// frame ±2, truncated at scene cuts), normalized into the center frame's
/// robust range and written to the cache video's luma plane. Unused slots must
/// still be bound (weight 0).
///
/// Each neighbor's weight is additionally gated by its depth similarity to the
/// center frame at this texel. The window has no motion compensation, so at a
/// moving silhouette the neighbors disagree with the center by ~the edge's
/// depth jump; averaging them in paints ghost bands that scroll with the
/// motion — invisible when the encode was soft, but a visible edge shimmer at
/// the sharp 2x guided-upsample encode. The gate collapses those taps toward
/// zero while depths within σ of the center (stable regions and genuine slow
/// drift — the jitter this window exists to smooth) keep ~full weight.
kernel void depthTemporalEncode(
    array<texture2d<float, access::sample>, 5> maps [[texture(0)]],
    texture2d<float, access::write> outLuma [[texture(5)]],
    constant DepthEncodeParams &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint W = outLuma.get_width();
    const uint H = outLuma.get_height();
    if (gid.x >= W || gid.y >= H) { return; }

    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 uv = (float2(gid) + 0.5) / float2(W, H);
    const float center = maps[2].sample(s, uv).r; // window is centered on slot 2
    float acc = 0.0;
    float wsum = 0.0;
    for (uint i = 0; i < 5; ++i) {
        // Neighbours are brought into the center frame's units first: the
        // model renormalizes per frame, so raw values of adjacent frames are
        // not comparable when a near subject enters or leaves.
        const float d = maps[i].sample(s, uv).r * p.slotScale[i] + p.slotBias[i];
        const float delta = d - center;
        const float w = p.weights[i] * exp(-delta * delta * p.motionSigmaInv);
        acc += w * d;
        wsum += w;
    }
    const float v = wsum > 1e-6 ? (acc / wsum) : center;
    outLuma.write(float4(saturate((v - p.rangeLo) * p.rangeInvSpan), 0.0, 0.0, 1.0), gid);
}

// MARK: - Depth Anchoring (scale/shift locking, see DepthRefinement.swift)
//
// Monocular depth has an arbitrary scale and shift per frame, so a near
// subject entering renormalizes the map and pushes the unchanged room back.
// Pixels whose image did not change since the last frame anchor a fit of this
// frame's depth units onto the previous frame's; the CPU chains those fits into
// a temporally locked mapping and the apply kernel writes the result. Both
// kernels run at depth resolution.

struct DepthAnchorReduceParams {
    float staticLuma;  // luma change below which a pixel anchors the fit
    uint  hasPrev;     // 1 when depthPrev / guidePrev hold the previous frame
    // Background layer: only pixels farther than this in both frames (the
    // previous frame's units, soft over `ceilingSoftness`) anchor the fit, so
    // a near subject — or a pixel it just covered, whose luma happened to
    // match — can't pull it.
    float ceiling;
    float ceilingSoftness;
};

/// Per 16×16 threadgroup, `depthAnchorPartialStride` floats: the anchor's
/// weighted sums w, wx, wy, wxx, wyy, wxy (x = this frame, y = previous), the
/// summed absolute luma change, and this frame's min and max depth. Must be
/// dispatched with threadsPerThreadgroup = 16×16.
constant uint depthAnchorPartialStride = 9;

kernel void depthAnchorReduce(
    texture2d<float, access::read> depth     [[texture(0)]],
    texture2d<float, access::read> depthPrev [[texture(1)]],
    texture2d<float, access::read> guide     [[texture(2)]],
    texture2d<float, access::read> guidePrev [[texture(3)]],
    device float *partials                   [[buffer(0)]],
    constant DepthAnchorReduceParams &p      [[buffer(1)]],
    uint2 gid  [[thread_position_in_grid]],
    uint  tix  [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgPerGrid [[threadgroups_per_grid]]
) {
    threadgroup float s[7 * 256];
    threadgroup float sMin[256];
    threadgroup float sMax[256];
    float v[7] = { 0, 0, 0, 0, 0, 0, 0 };
    float lo = INFINITY;
    float hi = -INFINITY;
    if (gid.x < depth.get_width() && gid.y < depth.get_height()) {
        const float x = depth.read(gid).r;
        if (isfinite(x)) {
            lo = x;
            hi = x;
            if (p.hasPrev != 0) {
                const float y = depthPrev.read(gid).r;
                const float dl = abs(guide.read(gid).r - guidePrev.read(gid).r);
                // Soft gate: full weight when unchanged, none at the threshold.
                const float w = isfinite(y)
                    ? saturate(1.0 - dl / p.staticLuma)
                      * saturate((p.ceiling - max(x, y)) / p.ceilingSoftness)
                    : 0.0;
                v[0] = w;
                v[1] = w * x;
                v[2] = w * y;
                v[3] = w * x * x;
                v[4] = w * y * y;
                v[5] = w * x * y;
                v[6] = dl;
            }
        }
    }
    for (uint k = 0; k < 7; ++k) { s[k * 256 + tix] = v[k]; }
    sMin[tix] = lo;
    sMax[tix] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tix < stride) {
            for (uint k = 0; k < 7; ++k) { s[k * 256 + tix] += s[k * 256 + tix + stride]; }
            sMin[tix] = min(sMin[tix], sMin[tix + stride]);
            sMax[tix] = max(sMax[tix], sMax[tix + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tix == 0) {
        device float *out = partials + (tgid.y * tgPerGrid.x + tgid.x) * depthAnchorPartialStride;
        for (uint k = 0; k < 7; ++k) { out[k] = s[k * 256]; }
        out[7] = sMin[0];
        out[8] = sMax[0];
    }
}

/// Folds the reduce's per-group min/max into `range` (one float2) so the
/// histogram can follow in the same command buffer. One threadgroup of 256.
kernel void depthAnchorFoldRange(
    device const float *partials [[buffer(0)]],
    constant uint &count         [[buffer(1)]],
    device float2 *range         [[buffer(2)]],
    uint tix [[thread_index_in_threadgroup]]
) {
    threadgroup float sMin[256];
    threadgroup float sMax[256];
    float lo = INFINITY;
    float hi = -INFINITY;
    for (uint i = tix; i < count; i += 256) {
        lo = min(lo, partials[i * depthAnchorPartialStride + 7]);
        hi = max(hi, partials[i * depthAnchorPartialStride + 8]);
    }
    sMin[tix] = lo;
    sMax[tix] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tix < stride) {
            sMin[tix] = min(sMin[tix], sMin[tix + stride]);
            sMax[tix] = max(sMax[tix], sMax[tix + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tix == 0) {
        range[0] = isfinite(sMin[0]) && isfinite(sMax[0]) ? float2(sMin[0], sMax[0]) : float2(0.0, 1.0);
    }
}

/// `depthHistogram256` with its range read from the fold's buffer.
kernel void depthHistogram256Ranged(
    texture2d<float, access::read> src [[texture(0)]],
    device atomic_uint *bins           [[buffer(0)]],
    device const float2 *range         [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) { return; }
    const float d = src.read(gid).r;
    if (!isfinite(d)) { return; }
    const float lo = range[0].x;
    const float invSpan = 1.0 / max(range[0].y - lo, 1e-6);
    const uint bin = uint(clamp((d - lo) * invSpan, 0.0f, 0.999999f) * 256.0f);
    atomic_fetch_add_explicit(&bins[bin], 1u, memory_order_relaxed);
}

// Measurement only: how much of a shot-space depth map sits on a large depth
// step once shown through the budget. Each texel is compared with the texels
// `offset` to its right and below; steps[i] counts texels whose larger step
// exceeds thresholds[i]. A near subject against a far background is a step of
// most of the budget along its whole outline, which an interior rarely has.
kernel void depthStepCensus(
    texture2d<float, access::read> src [[texture(0)]],
    device atomic_uint *steps          [[buffer(0)]],
    constant float4 &thresholds        [[buffer(1)]],
    constant float2 &knees             [[buffer(2)]],
    constant uint &offset              [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = src.get_width(), h = src.get_height();
    if (gid.x >= w || gid.y >= h) { return; }
    const float d = pseudo3DDepthBudget(src.read(gid).r, knees);
    const float dx = pseudo3DDepthBudget(src.read(uint2(min(gid.x + offset, w - 1), gid.y)).r, knees);
    const float dy = pseudo3DDepthBudget(src.read(uint2(gid.x, min(gid.y + offset, h - 1))).r, knees);
    const float step = max(abs(d - dx), abs(d - dy));
    if (!isfinite(step)) { return; }
    for (uint i = 0; i < 3; i++) {
        if (step > thresholds[i]) { atomic_fetch_add_explicit(&steps[i], 1u, memory_order_relaxed); }
    }
}

// Snap a soft depth outline to the picture. The model draws outlines soft,
// so around a near subject against a far background there is a band of
// in-between depth, and the warp stretches both the subject's edge and the
// background across it. Within `radius` texels along the row, a texel
// whose window's depth range exceeds `minStep` takes the near or the
// far depth, whichever end's colour it matches. Ambiguous colour leaves the
// texel as it was: a wrong snap tears an outline, which is worse than a soft
// one. Gentle slopes (floors, walls) never span `minStep` within the radius.
struct DepthEdgeSnapParams {
    float2 uvScale;   // as DepthGuideParams: depth texel UV → video UV
    float2 uvOffset;
    int   radius;
    float minStep;    // shot-space depth range that counts as an outline
    float minContrast; // colour difference between the two ends to trust them
};

kernel void depthEdgeSnap(
    texture2d<float, access::read>   src   [[texture(0)]],
    texture2d<float, access::sample> video [[texture(1)]],
    texture2d<float, access::write>  dst   [[texture(2)]],
    constant DepthEdgeSnapParams &p        [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int W = int(src.get_width()), H = int(src.get_height());
    if (int(gid.x) >= W || int(gid.y) >= H) { return; }
    const int x = int(gid.x), y = int(gid.y);
    const float d = src.read(gid).r;
    float dMin = d, dMax = d;
    for (int i = -p.radius; i <= p.radius; i++) {
        const float v = src.read(uint2(clamp(x + i, 0, W - 1), y)).r;
        dMin = min(dMin, v);
        dMax = max(dMax, v);
    }
    const float range = dMax - dMin;
    if (!(range > p.minStep)) { dst.write(float4(d, 0, 0, 1), gid); return; }
    const float tDepth = (d - dMin) / range;
    // Confident depth (inside the subject, the background) stands.
    if (tDepth < 0.08 || tDepth > 0.92) { dst.write(float4(d, 0, 0, 1), gid); return; }
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 size = float2(W, H);
    const float2 sc = max(p.uvScale, float2(1e-5));
    #define VIDEO_AT(px) video.sample(s, clamp(((float2(px, y) + 0.5) / size - p.uvOffset) / sc, 0.0, 1.0)).rgb
    // The outline is the strongest colour edge inside the soft band along the
    // row: the model says roughly where the outline is, the picture says
    // exactly. Earlier versions sorted each texel by colour, near or far, and
    // a dark subject before a dark background (a cap and neck against trees,
    // Nope, headset 2026-10-12) had texels inside it sorted far, as holes the
    // far eye saw the background through. With one edge per outline, every
    // texel of the band agrees on it, and nothing deeper inside can flip.
    int edge = -1000;
    float strongest = 0.0, strongestG = 0.0;
    float3 prevC = VIDEO_AT(x - p.radius);
    float prevT = (src.read(uint2(clamp(x - p.radius, 0, W - 1), y)).r - dMin) / range;
    for (int i = -p.radius + 1; i <= p.radius; i++) {
        const int q = clamp(x + i, 0, W - 1);
        const float3 cq = VIDEO_AT(q);
        const float tq = (src.read(uint2(q, y)).r - dMin) / range;
        // Only between two texels inside the band's core, and weighted
        // towards where the model's depth is halfway, a little outwards of it.
        // The strongest edge alone was at times one inside the subject (a
        // cap's brim against the neck, beside a dark head against dark trees),
        // which put the depth outline inside the head and the far eye's fill
        // painted background over it (Nope, headset 2026-10-12). Erring
        // outwards costs a sliver of background moving with the subject,
        // erring inwards a missing piece of it.
        const float mid = 0.5 * (tq + prevT);
        if (mid > 0.15 && mid < 0.85) {
            const float off = (mid - 0.42) / 0.45;
            const float centred = 1.0 - off * off;
            const float g = length(cq - prevC);
            const float score = g * max(centred, 0.0);
            if (score > strongest) { strongest = score; edge = i; strongestG = g; }   // between i-1 and i
        }
        prevC = cq;
        prevT = tq;
    }
    if (edge == -1000 || strongestG < p.minContrast) { dst.write(float4(d, 0, 0, 1), gid); return; }
    // Which side of the edge is near: the depth just past it on each side.
    const float tLeft = (src.read(uint2(clamp(x + edge - 3, 0, W - 1), y)).r - dMin) / range;
    const float tRight = (src.read(uint2(clamp(x + edge + 2, 0, W - 1), y)).r - dMin) / range;
    const bool nearOnRight = tRight > tLeft;
    // The two sides' depth: the window's ends.
    const float dNear = dMax, dFar = dMin;
    // A texel at the edge keeps its share of each side's colour (the outline's
    // sub-texel position); the rest take their side's depth.
    float t;
    if (edge == 0 || edge == 1) {
        const float3 cA = VIDEO_AT(x + edge - 2), cB = VIDEO_AT(x + edge + 1);
        const float3 span = cB - cA;
        const float tc = clamp(dot(VIDEO_AT(x) - cA, span) / max(dot(span, span), 1e-6), 0.0, 1.0);
        t = nearOnRight ? tc : 1.0 - tc;
    } else {
        const bool onRight = edge <= 0;
        t = (onRight == nearOnRight) ? 1.0 : 0.0;
    }
    #undef VIDEO_AT
    // Snap fully inside the band's core, easing out towards its ends.
    const float w = smoothstep(0.08, 0.2, tDepth) * (1.0 - smoothstep(0.8, 0.92, tDepth));
    dst.write(float4(mix(d, mix(dFar, dNear, t), w), 0, 0, 1), gid);
}

// The vertical half of the snap, after `depthEdgeSnap`. The model draws a
// thin near object's top and bottom soft too, and that band spilled onto
// whatever sits just below: the dark steps under Nope's corral rail took
// part of the rail's depth on their top rows, which shifted less than their
// bottom rows and sheared them like a smudge (headset 2026-10-11). Colour
// alone can't settle it (the first, symmetric column arm matched the dark
// steps to the rail's shaded underside and pulled them up to it), so this
// side is earned: inside the soft band a texel keeps the near depth only if
// its colour is the near side's and every texel between it and the near
// side's is too. Anything else in the band goes to the far side. Leaning far
// is cheap here: depth shifts the picture sideways, and a horizontal edge's
// rows sliding sideways along it doesn't show.
kernel void depthEdgeSnapVertical(
    texture2d<float, access::read>   src   [[texture(0)]],
    texture2d<float, access::sample> video [[texture(1)]],
    texture2d<float, access::write>  dst   [[texture(2)]],
    constant DepthEdgeSnapParams &p        [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int W = int(src.get_width()), H = int(src.get_height());
    if (int(gid.x) >= W || int(gid.y) >= H) { return; }
    const int x = int(gid.x), y = int(gid.y);
    const float d = src.read(gid).r;
    // Twice the row pass's reach: a thin object's soft band spans more than
    // the object itself, and the window must reach past it to see the far side.
    const int R = 2 * p.radius;
    float dMin = d, dMax = d;
    // The steepest change over a few texels: an outline's soft band is steep,
    // the ground running off to the horizon is a gentle slope over the whole
    // window. Without this the ground counted as an outline, its darker tufts
    // were pushed to the far side as speckles, and each column deciding alone
    // drew streaks (Nope's corral, headset 2026-10-12).
    constexpr int kStep = 3;
    float steepest = 0.0;
    float back[kStep];
    for (int k = 0; k < kStep; k++) { back[k] = src.read(uint2(x, clamp(y - R + k, 0, H - 1))).r; }
    for (int i = -R; i <= R; i++) {
        const float v = src.read(uint2(x, clamp(y + i, 0, H - 1))).r;
        dMin = min(dMin, v);
        dMax = max(dMax, v);
        if (i >= -R + kStep) {
            const int slot = (i + R) % kStep;
            steepest = max(steepest, abs(v - back[slot]));
            back[slot] = v;
        }
    }
    const float range = dMax - dMin;
    if (!(range > p.minStep) || steepest < max(0.25 * p.minStep, 0.12 * range)) {
        dst.write(float4(d, 0, 0, 1), gid);
        return;
    }
    // Only where the outline runs across the column: a mostly sideways depth
    // change is the row pass's, and there this pass would only add streaks.
    {
        const float gy = abs(src.read(uint2(x, clamp(y + kStep, 0, H - 1))).r - src.read(uint2(x, clamp(y - kStep, 0, H - 1))).r);
        const float gx = abs(src.read(uint2(clamp(x + kStep, 0, W - 1), y)).r - src.read(uint2(clamp(x - kStep, 0, W - 1), y)).r);
        if (gy < 2.0 * gx) { dst.write(float4(d, 0, 0, 1), gid); return; }
    }
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 size = float2(W, H);
    const float2 sc = max(p.uvScale, float2(1e-5));
    #define VIDEO_AT(px, py) video.sample(s, clamp(((float2(px, py) + 0.5) / size - p.uvOffset) / sc, 0.0, 1.0)).rgb
    const float mid = 0.5 * (dMin + dMax);
    float3 cNearSum = 0, cFarSum = 0;
    float dNearSum = 0, dFarSum = 0, nNear = 0, nFar = 0;
    for (int i = -R; i <= R; i++) {
        if (i == 0) { continue; }
        const int qy = clamp(y + i, 0, H - 1);
        const float v = src.read(uint2(x, qy)).r;
        const float e = 2.0 * (v - mid) / range;
        const float wq = smoothstep(0.5, 1.0, abs(e));
        if (wq <= 1e-4) { continue; }
        const float3 cq = VIDEO_AT(x, qy);
        if (v > mid) { cNearSum += wq * cq; dNearSum += wq * v; nNear += wq; }
        else { cFarSum += wq * cq; dFarSum += wq * v; nFar += wq; }
    }
    if (nNear < 0.25 || nFar < 0.25) { dst.write(float4(d, 0, 0, 1), gid); return; }
    const float3 cNear = cNearSum / nNear, cFar = cFarSum / nFar;
    const float dNear = dNearSum / nNear, dFar = dFarSum / nFar;
    const float3 span = cNear - cFar;
    const float spanLen = length(span);
    if (spanLen < p.minContrast) { dst.write(float4(d, 0, 0, 1), gid); return; }
    const float tDepth = clamp((d - dFar) / max(dNear - dFar, 1e-6), 0.0, 1.0);
    // Confident depth (the near object's core, the background) stands.
    const float inBand = smoothstep(0.03, 0.12, tDepth) * (1.0 - smoothstep(0.93, 0.99, tDepth));
    if (inBand <= 0.0) { dst.write(float4(d, 0, 0, 1), gid); return; }
    // Which sure region the texel belongs to: walk up and down the column
    // until the colour changes sharply, and see whether the walk reaches sure
    // near depth, sure far depth, or neither. The first version leaned far
    // (near only for near-coloured texels joined to the near side), and a
    // tan glove before tan hills, coloured like neither the jacket and horse
    // below it nor anything near, lost its top to the background (Nope,
    // headset 2026-10-12). Joined to both, it stays near: a near object keeps
    // all of itself, and at worst a sliver of background moves with it.
    // Joined to neither, the model's depth stands.
    const float edgeStep = max(p.minContrast, 0.35 * spanLen);
    const float3 cOwn = VIDEO_AT(x, y);
    bool reachesNear = false, reachesFar = false;
    for (int dir = -1; dir <= 1; dir += 2) {
        float3 prev = cOwn;
        for (int i = 1; i <= R; i++) {
            const int qy = y + dir * i;
            if (qy < 0 || qy >= H) { break; }
            const float3 cq = VIDEO_AT(x, qy);
            if (length(cq - prev) > edgeStep) { break; }
            const float tq = (src.read(uint2(x, qy)).r - dFar) / max(dNear - dFar, 1e-6);
            if (tq > 0.9) { reachesNear = true; break; }
            if (tq < 0.1) { reachesFar = true; break; }
            prev = cq;
        }
    }
    #undef VIDEO_AT
    float target = d;
    if (reachesNear) { target = max(d, dNear); }
    else if (reachesFar) { target = dFar; }
    dst.write(float4(mix(d, target, inBand), 0, 0, 1), gid);
}

// A 3×3 median after the snap. Where a subject's colour is close to the
// background's (a pale shirt against pale ground) the snap decides texel by
// texel, and a few on the subject's side of the outline went far: single
// specks, which the mesh warp's fill then painted with background as pale
// blotches along the outline (Nope's corral, headset 2026-10-12). A median
// drops a speck up to 2×2 texels and leaves an outline where it is.
//
// Then held over time where the picture didn't move. The snap runs after the
// refiner's temporal blend and decided every texel afresh each frame, so on
// fine texture (a jacket's weave, film grain) its choice of edge changed from
// frame to frame and the outline shimmered (headset 2026-10-12). Where the
// luma is unchanged, the previous result is blended toward slowly; where it
// moved, the new one is taken at once, so a moving outline doesn't lag.
struct DepthSnapHoldParams {
    uint  hasPrev;
    float staticLuma;  // luma change below which a texel counts as static
    float staticAlpha; // blend toward the new result for a static texel
    float pad;
};

kernel void depthMedian3x3(
    texture2d<float, access::read>   src       [[texture(0)]],
    texture2d<float, access::write>  dst       [[texture(1)]],
    texture2d<float, access::read>   prev      [[texture(2)]],
    texture2d<float, access::sample> guide     [[texture(3)]],
    texture2d<float, access::sample> guidePrev [[texture(4)]],
    constant DepthSnapHoldParams &p            [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int W = int(src.get_width()), H = int(src.get_height());
    if (int(gid.x) >= W || int(gid.y) >= H) { return; }
    float v[9];
    int n = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            v[n++] = src.read(uint2(clamp(int(gid.x) + i, 0, W - 1), clamp(int(gid.y) + j, 0, H - 1))).r;
        }
    }
    // Partial selection: the 5th smallest of nine.
    for (int a = 0; a < 5; a++) {
        int m = a;
        for (int b = a + 1; b < 9; b++) { if (v[b] < v[m]) { m = b; } }
        const float t = v[a]; v[a] = v[m]; v[m] = t;
    }
    float out = v[4];
    if (p.hasPrev != 0) {
        constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
        const float2 uv = (float2(gid) + 0.5) / float2(W, H);
        const float change = abs(guide.sample(s, uv).r - guidePrev.sample(s, uv).r);
        const float still = saturate(1.0 - change / p.staticLuma);
        out = mix(prev.read(gid).r, out, mix(1.0, p.staticAlpha, still));
    }
    dst.write(float4(out, 0, 0, 1), gid);
}

struct DepthAnchorApplyParams {
    float a;           // this frame's units → shot space
    float b;
    float baseAlpha;   // temporal blend, as DepthStabilizeParams
    float motionGain;
    uint  hasPrev;
    float staticLuma;  // luma change below which a pixel counts as static
    float staticAlpha; // temporal blend for a static pixel
};

/// Shot-space depth with a temporal EMA. Where the image moved, the
/// motion-adaptive EMA as before (now on a scale that no longer jumps when the
/// model renormalizes, so a rescale no longer reads as "everything moved").
/// Where it didn't, the surface is the same surface at the same depth, so the
/// pixel's depth is held: only a slow blend toward the new estimate, whatever
/// the model now says. Monocular depth reshuffles the background non-affinely
/// when a subject enters; the global anchor can't undo that, this holds it.
/// The shot's gap curve (DepthGapCurve.apply): piecewise linear through
/// `depthGapKnots + 1` knots over 0…1, slope 1 outside.
constant uint depthGapKnots = 32;
static inline float depthGapCurve(float y, constant float *knots) {
    if (y <= 0.0) { return knots[0] + y; }
    if (y >= 1.0) { return knots[depthGapKnots] + (y - 1.0); }
    const float x = y * float(depthGapKnots);
    const uint k = min(uint(x), depthGapKnots - 1);
    return mix(knots[k], knots[k + 1], x - float(k));
}

kernel void depthAnchorApply(
    texture2d<float, access::read>  depth     [[texture(0)]],
    texture2d<float, access::read>  prev      [[texture(1)]],
    texture2d<float, access::write> next      [[texture(2)]],
    texture2d<float, access::read>  guide     [[texture(3)]],
    texture2d<float, access::read>  guidePrev [[texture(4)]],
    constant DepthAnchorApplyParams &p        [[buffer(0)]],
    constant float *gapKnots                  [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= next.get_width() || gid.y >= next.get_height()) { return; }
    float y = depthGapCurve(depth.read(gid).r * p.a + p.b, gapKnots);
    if (p.hasPrev != 0) {
        const float old = prev.read(gid).r;
        const float moving = clamp(p.baseAlpha + p.motionGain * abs(y - old), p.baseAlpha, 1.0);
        const float still = saturate(1.0 - abs(guide.read(gid).r - guidePrev.read(gid).r) / p.staticLuma);
        y = mix(old, y, mix(moving, p.staticAlpha, still));
    }
    next.write(float4(y, 0.0, 0.0, 1.0), gid);
}

// MARK: - Subtitles in the Stereo Picture
//
// See StereoSubtitles.swift. A subtitle image is drawn over each finished eye
// image, shifted sideways by that eye's share of the text's disparity.

struct SubtitleQuadUniforms {
    float2 rectOrigin;   // eye UV, origin top left, before the shift
    float2 rectSize;
    float shift;         // this eye's horizontal shift in eye UV
    float transfer;      // 0 = SDR, 1 = PQ, 2 = HLG
    float whiteNits;     // the text's white on an HDR picture
    float padding;
};

vertex VertexOut stereoSubtitleVertex(
    uint vertexID [[vertex_id]],
    constant SubtitleQuadUniforms &u [[buffer(0)]]
) {
    const float2 corners[] = {
        float2(0, 1), float2(1, 1), float2(0, 0),
        float2(0, 0), float2(1, 1), float2(1, 0)
    };
    float2 corner = corners[vertexID];
    float2 uv = u.rectOrigin + corner * u.rectSize + float2(u.shift, 0);
    VertexOut out;
    out.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0, 1);
    out.texCoord = corner;   // the image's rows run top down
    return out;
}

static inline float3 subtitleSRGBToLinear(float3 c) {
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

// SMPTE ST 2084 inverse EOTF: absolute luminance in nits to the PQ signal.
static inline float3 subtitlePQEncode(float3 nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

// BT.2100 HLG: display luminance on the nominal 1000-nit display (system
// gamma 1.2) back to scene light, then the OETF. 203 nits lands on 75 %,
// BT.2408's reference white.
static inline float3 subtitleHLGEncode(float3 nits) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    float3 e = pow(clamp(nits / 1000.0, 0.0, 1.0), 1.0 / 1.2);
    return select(a * log(max(12.0 * e - b, 1e-6)) + c, sqrt(3.0 * e), e <= 1.0 / 12.0);
}

fragment float4 stereoSubtitleFragment(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    constant SubtitleQuadUniforms &u [[buffer(0)]]
) {
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_zero);
    float4 c = tex.sample(s, in.texCoord);   // premultiplied sRGB
    if (c.a <= 0.0) { discard_fragment(); }
    // SDR eyes are BT.709-coded, close enough to sRGB to write as is.
    if (u.transfer < 0.5) { return c; }
    float3 nits = subtitleSRGBToLinear(c.rgb / c.a) * u.whiteNits;
    float3 coded = u.transfer < 1.5 ? subtitlePQEncode(nits) : subtitleHLGEncode(nits);
    return float4(coded * c.a, c.a);
}

// Disparity behind the subtitle text in a real stereo pair. One threadgroup
// per patch of the region, one thread per candidate disparity d: the left
// eye's 8×8 patch against the right eye's patch d pixels to the left
// (xL - xR = d, so positive d is crossed: in front of the screen), compared
// mean-removed so the eyes' exposure needn't match. A patch answers only if
// it has texture and one candidate clearly wins; flat sky or a repeating
// pattern answers nothing rather than something wrong.
struct SubtitleMatchParams {
    float2 regionOrigin;   // eye pixels
    float2 regionSize;
    uint2 patchGrid;
    int candidateStep;     // pixels between candidates
    int sampleStep;        // pixels between patch samples
};

static inline float subtitleLuma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

kernel void stereoSubtitleMatch(
    texture2d<float, access::read> left [[texture(0)]],
    texture2d<float, access::read> right [[texture(1)]],
    constant SubtitleMatchParams &p [[buffer(0)]],
    device float2 *results [[buffer(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int kCandidates = 128;   // = threads per group, StereoSubtitleCompositor.candidateCount
    constexpr int kSamples = 64;       // 8×8
    threadgroup float leftLuma[kSamples];
    threadgroup float cost[kCandidates];

    int2 limit = int2(left.get_width(), left.get_height()) - 1;
    float2 cell = p.regionSize / float2(p.patchGrid);
    int2 origin = int2(p.regionOrigin + (float2(group) + 0.5) * cell) - 4 * p.sampleStep;

    if (tid < uint(kSamples)) {
        int2 at = clamp(origin + int2(tid % 8, tid / 8) * p.sampleStep, int2(0), limit);
        leftLuma[tid] = subtitleLuma(left.read(uint2(at)).rgb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float leftMean = 0;
    for (int i = 0; i < kSamples; i++) { leftMean += leftLuma[i]; }
    leftMean /= kSamples;

    int d = (int(tid) - kCandidates / 2) * p.candidateStep;
    float rightMean = 0;
    for (int i = 0; i < kSamples; i++) {
        int2 at = clamp(origin + int2(i % 8 * p.sampleStep - d, i / 8 * p.sampleStep), int2(0), limit);
        rightMean += subtitleLuma(right.read(uint2(at)).rgb);
    }
    rightMean /= kSamples;
    float sad = 0;
    for (int i = 0; i < kSamples; i++) {
        int2 at = clamp(origin + int2(i % 8 * p.sampleStep - d, i / 8 * p.sampleStep), int2(0), limit);
        sad += abs((leftLuma[i] - leftMean) - (subtitleLuma(right.read(uint2(at)).rgb) - rightMean));
    }
    cost[tid] = sad;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) { return; }
    int best = 0;
    for (int k = 1; k < kCandidates; k++) { if (cost[k] < cost[best]) { best = k; } }
    float runnerUp = 1e30;   // not INFINITY: the library builds with fast math
    for (int k = 0; k < kCandidates; k++) {
        if (abs(k - best) > 2) { runnerUp = min(runnerUp, cost[k]); }
    }
    float texture = 0;
    for (int i = 0; i < kSamples; i++) { texture += abs(leftLuma[i] - leftMean); }
    texture /= kSamples;
    bool valid = texture > 0.02 && best > 0 && best < kCandidates - 1 && cost[best] < 0.8 * runnerUp;
    results[group.y * p.patchGrid.x + group.x] =
        float2(float((best - kCandidates / 2) * p.candidateStep), valid ? 1.0 : 0.0);
}
