#if os(macOS)
  import Accelerate
  import CoreImage
  import CoreVideo
  import Foundation
  import ImageIO
  import Metal
  import Testing
  @testable import VergenceApple
  import VergenceCore

  /// Opt-in local-file investigation. Never runs model inference in normal tests.
  /// `VERGENCE_VIDEO_FRAME` accepts one PNG or a directory of consecutive PNGs.
  /// A directory also requires `VERGENCE_VIDEO_FPS`; `VERGENCE_VIDEO_START`
  /// supplies its nominal starting time (default 742). These are caller-supplied
  /// times, not timestamps decoded from the original video. Optional comma-
  /// separated `VERGENCE_VIDEO_VARIANTS` limits the shader experiments rendered.
  @Suite(.serialized)
  struct VideoStereoFrameLabTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VERGENCE_VIDEO_FRAME"] != nil))
    func exportFrame() throws {
      let env = ProcessInfo.processInfo.environment
      let inputURL = URL(fileURLWithPath: try #require(env["VERGENCE_VIDEO_FRAME"]))
      let modelURL = URL(fileURLWithPath: try #require(env["VERGENCE_VIDEO_MODEL"]))
      let outputRoot = URL(fileURLWithPath: try #require(env["VERGENCE_VIDEO_OUTPUT"]))
      // A PNG directory is a consecutive sequence, with one provider retained across
      // frames so stabilization and shot-space anchoring are exercised too.
      let isSequence = try inputURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
      let inputs =
        isSequence
        ? try FileManager.default.contentsOfDirectory(at: inputURL, includingPropertiesForKeys: nil)
          .filter { $0.pathExtension.lowercased() == "png" }.sorted { $0.path < $1.path }
        : [inputURL]
      try #require(!inputs.isEmpty)
      let fps = isSequence ? try #require(Double(env["VERGENCE_VIDEO_FPS"] ?? "")) : 24
      try #require(fps.isFinite && fps > 0)
      let start = try #require(Double(env["VERGENCE_VIDEO_START"] ?? "742"))
      try #require(start.isFinite)
      let device = try #require(MTLCreateSystemDefaultDevice())
      let queue = try #require(device.makeCommandQueue())
      let provider = try #require(CoreMLDepthProvider(device: device, modelURL: modelURL))
      try #require(provider.supportsRefinement)
      for (frameIndex, imageURL) in inputs.enumerated() {
        let output =
          isSequence
          ? outputRoot.appendingPathComponent(imageURL.deletingPathExtension().lastPathComponent)
          : outputRoot
        try #require(
          !FileManager.default.fileExists(atPath: output.path), "Choose a new output directory")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let image = try #require(CIImage(contentsOf: imageURL))
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(mtlDevice: device, options: [.workingColorSpace: NSNull()])
        var buffer: CVPixelBuffer?
        #expect(
          CVPixelBufferCreate(
            nil, width, height, kCVPixelFormatType_32BGRA,
            [
              kCVPixelBufferMetalCompatibilityKey: true,
              kCVPixelBufferIOSurfacePropertiesKey: [:],
            ] as CFDictionary, &buffer) == kCVReturnSuccess)
        let frame = try #require(buffer)
        context.render(image, to: frame, bounds: image.extent, colorSpace: srgb)
        CVBufferSetAttachment(
          frame, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2,
          .shouldPropagate)
        CVBufferSetAttachment(
          frame, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2,
          .shouldPropagate)
        CVBufferSetAttachment(
          frame, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2,
          .shouldPropagate
        )
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        var cvTexture: CVMetalTexture?
        #expect(
          CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, frame, nil, .bgra8Unorm,
            width, height, 0, &cvTexture) == kCVReturnSuccess)
        let sourceView = try #require(cvTexture)
        let source = try #require(CVMetalTextureGetTexture(sourceView))
        let snap = DepthRefinement.snapsEdges
        let fill = DepthRefinement.fillsUncoveredEdges
        DepthRefinement.snapsEdges = false
        DepthRefinement.fillsUncoveredEdges = false
        defer {
          DepthRefinement.snapsEdges = snap
          DepthRefinement.fillsUncoveredEdges = fill
        }
        let seconds = start + Double(frameIndex) / fps
        let depth = try #require(provider.refinedDepth(for: frame, frameSeconds: seconds))
        let uv = CoreMLDepthProvider.letterboxUVTransform(
          videoWidth: width, videoHeight: height,
          depthWidth: depth.width, depthHeight: depth.height)
        let knees = DepthBudget.studio.knees
        func texture(_ format: MTLPixelFormat, _ w: Int, _ h: Int) throws -> MTLTexture {
          let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: w, height: h, mipmapped: false)
          d.storageMode = .shared
          d.usage = [.shaderRead, .shaderWrite, .renderTarget]
          return try #require(device.makeTexture(descriptor: d))
        }
        // Snapshot full precision as stored, including the shot-space range.
        let staging = try texture(depth.pixelFormat, depth.width, depth.height)
        let copy = try #require(queue.makeCommandBuffer())
        let blit = try #require(copy.makeBlitCommandEncoder())
        blit.copy(from: depth, to: staging)
        blit.endEncoding()
        copy.commit()
        copy.waitUntilCompleted()
        #expect(copy.status == .completed)
        var half = [UInt16](repeating: 0, count: depth.width * depth.height)
        try #require(depth.pixelFormat == .r16Float)
        staging.getBytes(
          &half, bytesPerRow: depth.width * 2,
          from: MTLRegionMake2D(0, 0, depth.width, depth.height), mipmapLevel: 0)
        var floats = [Float](repeating: 0, count: half.count)
        let pixelCount = half.count
        half.withUnsafeMutableBytes { input in
          floats.withUnsafeMutableBytes { output in
            var src = vImage_Buffer(
              data: input.baseAddress, height: 1, width: vImagePixelCount(pixelCount),
              rowBytes: pixelCount * 2)
            var dst = vImage_Buffer(
              data: output.baseAddress, height: 1, width: vImagePixelCount(pixelCount),
              rowBytes: pixelCount * 4)
            #expect(vImageConvert_Planar16FtoPlanarF(&src, &dst, 0) == kvImageNoError)
          }
        }
        try floats.withUnsafeBytes {
          try Data($0).write(to: output.appendingPathComponent("depth.f32"))
        }
        let metadata: [String: Any] = [
          "model": provider.modelName, "sourceWidth": width, "sourceHeight": height,
          "width": depth.width, "height": depth.height, "format": "float32-little-endian",
          "origin": "top-left",
          "uvScale": [uv.scale.x, uv.scale.y], "uvOffset": [uv.offset.x, uv.offset.y],
          "valueScale": 1, "valueBias": 0, "depthBudget": [knees.x, knees.y], "strength": 0.008,
          "convergence": 1, "temporalState": frameIndex == 0 ? "first-frame" : "consecutive-frame",
          "frameSeconds": seconds, "sequenceIndex": frameIndex,
          "timeSource": "caller-supplied sequence start and rate",
          "compressesGaps": DepthRefinement.compressesGaps,
        ]
        try JSONSerialization.data(
          withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]
        )
        .write(to: output.appendingPathComponent("depth.json"))
        func write(_ texture: MTLTexture, _ name: String) throws {
          let ci = try #require(CIImage(mtlTexture: texture, options: [.colorSpace: srgb]))
          let upright = ci.transformed(
            by: CGAffineTransform(translationX: 0, y: CGFloat(texture.height)).scaledBy(x: 1, y: -1)
          )
          let data = try #require(
            context.pngRepresentation(of: upright, format: .RGBA8, colorSpace: srgb))
          try data.write(to: output.appendingPathComponent(name + ".png"))
        }
        try write(source, "source")
        let shaderURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
          .deletingLastPathComponent()
          .deletingLastPathComponent().appendingPathComponent(
            "Sources/VergenceApple/Stereo/StereoShaders.metal")
        let original = try String(contentsOf: shaderURL, encoding: .utf8)
        let linearDepth = original.replacingOccurrences(
          of: "depth.sample(point, depthUV).r",
          with: "depth.sample(s, in.uv * u.uv.xy + u.uv.zw).r")
        let linearColor = linearDepth.replacingOccurrences(
          of:
            "(in.split.x >= 0 ? source.sample(point, pointUV).rgb : source.sample(s, centerUV).rgb)",
          with: "source.sample(s, in.uv).rgb")
        let nearLinear = linearDepth.replacingOccurrences(
          of:
            "(in.split.x >= 0 ? source.sample(point, pointUV).rgb : source.sample(s, centerUV).rgb)",
          with:
            "(in.split.x >= 0 && in.split.y >= 0.5 ? source.sample(point, pointUV).rgb : source.sample(s, in.uv).rgb)"
        )
        // Research variant: all triangles share one depth-defined layer mask,
        // instead of choosing different midpoint thresholds for each triangle.
        let begin = try #require(original.range(of: "vertex BoundaryVertexOut boundaryMeshVertex("))
        let end = try #require(original.range(of: "struct BoundaryFragmentOut {"))
        let layerVertex = """
          static inline float2 layerRange(float2 uv, texture2d<float> depth, constant BoundaryUniforms &u) {
              constexpr sampler s(filter::linear, address::clamp_to_edge);
              float lo = 1, hi = 0;
              for (int i = -6; i <= 6; ++i) {
                  float2 duv = uv * u.uv.xy + u.uv.zw + float2(float(i) / float(depth.get_width()), 0);
                  float d = pseudo3DDepthBudget(depth.sample(s, duv).r * u.mapping.x + u.mapping.y, u.mapping.zw);
                  lo = min(lo,d); hi = max(hi,d);
              }
              return float2(lo,hi);
          }
          vertex BoundaryVertexOut boundaryMeshVertex(
              uint vid [[vertex_id]], const device float2 *positions [[buffer(0)]],
              const device uint *indices [[buffer(1)]], constant BoundaryUniforms &u [[buffer(2)]],
              texture2d<float> depth [[texture(0)]]) {
              uint count = uint(u.adjustments.w), layer = vid / count;
              float2 grid = positions[indices[vid % count]];
              float2 uv = float2(u.stereo.w > .5 ? 1-grid.x : grid.x, grid.y);
              float2 range = layerRange(uv,depth,u);
              bool split = range.y-range.x > .08;
              float d = split ? (layer == 0 ? range.y : range.x) : boundaryDepth(grid,depth,u);
              float shift = (d-u.stereo.y)*u.stereo.x*pseudo3DEdgeFeather(grid.x);
              BoundaryVertexOut out;
              out.position = float4(grid.x*2-1+u.stereo.z*shift*2,1-grid.y*2,1-d,1);
              out.uv=uv; out.depth=d; out.reject=0;
              out.split=float2(-1,float(layer));
              return out;
          }
          """
        var consistent = original.replacingCharacters(
          in: begin.lowerBound..<end.lowerBound, with: layerVertex)
        let clipStart = try #require(consistent.range(of: "    if (in.split.x >= 0) {"))
        let clipEnd = try #require(consistent.range(of: "    // Across a split, bilinear colour"))
        consistent.replaceSubrange(
          clipStart.lowerBound..<clipEnd.lowerBound,
          with: """
                float2 range = layerRange(in.uv,depth,u);
                float local = pseudo3DDepthBudget(depth.sample(s,in.uv*u.uv.xy+u.uv.zw).r*u.mapping.x+u.mapping.y,u.mapping.zw);
                bool near = range.y-range.x <= .08 || local >= (range.x+range.y)*.5;
                if ((in.split.y < .5) != near) { discard_fragment(); }

            """)
        consistent = consistent.replacingOccurrences(
          of:
            "(in.split.x >= 0 ? source.sample(point, pointUV).rgb : source.sample(s, centerUV).rgb)",
          with: "source.sample(s,in.uv).rgb")
        let confidentBackground = consistent.replacingOccurrences(
          of:
            "if ((in.split.y < .5) != near) { discard_fragment(); }",
          with:
            "if ((in.split.y < .5) != near || (in.split.y >= .5 && local > mix(range.x,range.y,.03))) { discard_fragment(); }"
        )
        let farBeforeFill = consistent.replacingOccurrences(
          of:
            "if (in.reject > 0.5) { discard_fragment(); }",
          with: "if (in.reject > 0.5 || in.split.y < .5) { discard_fragment(); }"
        )
        .replacingOccurrences(
          of: "const float4 packed = surfaces.read(p);",
          with:
            "output.write(float4(surfaces.read(p).rgb,1),p); return; const float4 packed = surfaces.read(p);"
        )
        #expect(linearDepth != original && linearColor != linearDepth && nearLinear != linearDepth)
        for (name, code, separated) in [
          ("connected", original, false), ("separated", original, true),
          ("linear-depth", linearDepth, true), ("linear-color", linearColor, true),
          ("near-linear", nearLinear, true), ("consistent-layers", consistent, true),
          ("confident-background", confidentBackground, true),
          ("far-before-fill", farBeforeFill, true),
          (
            "layers-quarter",
            consistent.replacingOccurrences(
              of: "(range.x+range.y)*.5", with: "mix(range.x,range.y,.25)"), true
          ),
          (
            "layers-tenth",
            consistent.replacingOccurrences(
              of: "(range.x+range.y)*.5", with: "mix(range.x,range.y,.1)"), true
          ),
        ] {
          if let selected = env["VERGENCE_VIDEO_VARIANTS"],
            !selected.split(separator: ",").contains(Substring(name))
          {
            continue
          }
          let library = try device.makeLibrary(source: code, options: nil)
          let resources = try #require(StereoWarpResources(device: device, library: library))
          let mesh = StereoMeshWarp(resources: resources)
          let boundary = try #require(StereoBoundaryWarp(device: device, library: library))
          for (eyeIndex, sign) in [Float(1), -1].enumerated() {
            let target = try texture(.bgra8Unorm, width, height)
            let command = try #require(queue.makeCommandBuffer())
            if separated {
              let u = StereoBoundaryWarp.Uniforms(
                stereo: SIMD4(0.008, 1, sign, 0),
                uv: SIMD4(uv.scale.x, uv.scale.y, uv.offset.x, uv.offset.y),
                mapping: SIMD4(1, 0, knees.x, knees.y), adjustments: SIMD4(0, 1, 1, 0),
                boundary: SIMD4(
                  0.08, Float(width), 1 / Float(StereoWarpResources.denseGridColumns - 1), 0))
              #expect(
                boundary.encode(
                  source: source, depth: depth, destination: target,
                  positions: resources.denseGridPositions, indices: resources.denseGridIndices,
                  indexCount: resources.denseGridIndexCount, uniforms: u, commandBuffer: command))
            } else {
              let u = VideoStereoUniforms(
                brightness: 0, contrast: 1, saturation: 1,
                depthStrength: 0.008, convergence: 1, eyeSign: sign, mirror: 0, useDepth: 1,
                depthUVScale: uv.scale, depthUVOffset: uv.offset, depthValueScale: 1,
                depthValueBias: 0,
                depthBudget: knees)
              #expect(
                mesh.encode(
                  source: source, depth: depth, destination: target, uniforms: u,
                  dense: true, eye: eyeIndex, commandBuffer: command))
            }
            command.commit()
            command.waitUntilCompleted()
            #expect(command.status == .completed)
            try write(target, name + (sign > 0 ? "-left" : "-right"))
          }
        }
        withExtendedLifetime([sourceView, frame]) {}
      }
    }
  }
#endif
