/*
 Vergence - AI 3D depth from an analysis track ("warp only")

 A worker (`vergence worker`, workload `ai-3d-depth`) makes a 2D title's
 depth with the offline pipeline and a larger model than a headset runs live:
 one `depth` record per frame, the picture's shot-space inverse depth before
 the studio budget's soft knee, as `OfflineDepthConversion.depthRecord`
 writes it. This source hands the pump those maps by the frame's time, so
 the headset only warps; a frame the track doesn't hold yet (the first
 seconds after opening or a seek, before the worker's pieces arrive) goes to
 the `fallback`, the player's own real-time source, with its ≤1-frame hold.

 The track holds a map for every frame, so nothing is held across frames
 here: a frame either has its own map or is the fallback's.
 */

#if canImport(Metal)
import CoreMedia
import CoreVideo
import Foundation
import Metal
import VergenceCore

/// A title's depth track as a player is handed it: where its records come
/// from (a file's index, or a coordinator's chunks as they fill) and the
/// title's frame rate.
public struct AnalysisDepthTrack: Sendable {
    public let lookup: any AnalysisRecordLookup
    public let frameRate: Double

    public init(lookup: any AnalysisRecordLookup, frameRate: Double) {
        self.lookup = lookup
        self.frameRate = frameRate
    }

    /// The pump's depth source: this track first, `fallback` for the rest.
    public func source(device: MTLDevice, fallback: PumpDepthSource?) -> AnalysisDepthSource {
        AnalysisDepthSource(lookup: lookup, frameRate: frameRate, device: device, fallback: fallback,
                            inflate: AnalysisDeflate.inflate)
    }
}

public final class AnalysisDepthSource: PumpDepthSource, @unchecked Sendable {
    private let lookup: any AnalysisRecordLookup
    private let fallback: PumpDepthSource?
    private let inflate: (Data, Int) throws -> Data
    private let device: MTLDevice
    /// A frame's map is matched within this many seconds of its time (half a frame).
    private let tolerance: Double
    private let lock = NSLock()
    private var free: [MTLTexture] = []
    private var lastSource: String?
    public private(set) var trackFrames = 0
    public private(set) var fallbackFrames = 0

    /// - Parameters:
    ///   - frameRate: the title's, which sets how near a map's time must be.
    ///   - fallback: depth for the frames the track doesn't hold (nil: those render flat).
    public init(lookup: any AnalysisRecordLookup, frameRate: Double, device: MTLDevice,
                fallback: PumpDepthSource?, inflate: @escaping (Data, Int) throws -> Data) {
        self.lookup = lookup
        self.fallback = fallback
        self.inflate = inflate
        self.device = device
        tolerance = 0.5 / max(frameRate, 1)
    }

    public let flattensWhenUnavailable = true
    /// Track depth is edge-aware (joint-bilateral) like cached depth, so it takes the dense grid.
    public var prefersDenseWarpGrid: Bool { true }
    public var limitsToSourceFrameRate: Bool { fallback?.limitsToSourceFrameRate ?? false }
    public var computesDepthAhead: Bool { fallback?.computesDepthAhead ?? false }

    public func requestDepth(for frame: PumpFrame, shownAt: CFTimeInterval,
                             whenReady: @escaping @Sendable () -> Void) -> PumpDepthRequest {
        if let depth = trackDepth(at: frame.itemTime.seconds) {
            return PumpDepthRequest(depth: depth)
        }
        note(fallback: true)
        guard let fallback else { return PumpDepthRequest(depth: nil) }
        return fallback.requestDepth(for: frame, shownAt: shownAt, whenReady: whenReady)
    }

    public func frameDepth(itemTime: CMTime, frame: CVPixelBuffer) -> PumpFrameDepth? {
        if let depth = trackDepth(at: itemTime.seconds) { return depth }
        note(fallback: true)
        return fallback?.frameDepth(itemTime: itemTime, frame: frame)
    }

    public func invalidate() { fallback?.invalidate() }
    public func recomputeNextDepth() { fallback?.recomputeNextDepth() }

    /// The track's map for the frame at `seconds`, uploaded, or nil.
    func trackDepth(at seconds: Double) -> PumpFrameDepth? {
        lookup.noteDemand(at: seconds)
        guard let (index, entry) = lookup.nearestRecord(.depth, eye: .mono, to: AnalysisTime(seconds: seconds),
                                                        within: tolerance),
              let record = try? index.record(entry).inflated(using: inflate),
              record.channels == 1, record.encoding == .uint8,
              let texture = upload(record) else { return nil }
        note(fallback: false)
        // value = byte × scale + offset, and the texture reads byte / 255.
        return PumpFrameDepth(texture: texture.texture, uvScale: SIMD2(1, 1), uvOffset: SIMD2(0, 0),
                              valueScale: record.scale * 255, valueBias: record.offset, budget: .studio,
                              keepAlive: [texture])
    }

    private func note(fallback: Bool) {
        lock.lock()
        if fallback { fallbackFrames += 1 } else { trackFrames += 1 }
        let source = fallback ? "the device's model" : "the analysis track"
        let changed = source != lastSource
        lastSource = source
        lock.unlock()
        if changed { VergenceLog.pipeline.notice("AI 3D depth: from \(source, privacy: .public)") }
    }

    /// The record's bytes in an r8 texture, reused once the pump lets go of it.
    private func upload(_ record: AnalysisRecord) -> TrackDepthTexture? {
        lock.lock()
        let reused = free.firstIndex { $0.width == record.width && $0.height == record.height }.map { free.remove(at: $0) }
        lock.unlock()
        let texture: MTLTexture
        if let reused {
            texture = reused
        } else {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: record.width,
                                                                      height: record.height, mipmapped: false)
            descriptor.usage = .shaderRead
            descriptor.storageMode = .shared
            guard let made = device.makeTexture(descriptor: descriptor) else { return nil }
            texture = made
        }
        record.payload.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, record.width, record.height), mipmapLevel: 0,
                            withBytes: raw.baseAddress!, bytesPerRow: record.width)
        }
        return TrackDepthTexture(texture: texture) { [weak self] texture in
            guard let self else { return }
            self.lock.lock()
            if self.free.count < 8 { self.free.append(texture) }
            self.lock.unlock()
        }
    }
}

/// Keeps a track map's texture out of the free list while a drawn frame holds it.
final class TrackDepthTexture: @unchecked Sendable {
    let texture: MTLTexture
    private let giveBack: (MTLTexture) -> Void

    init(texture: MTLTexture, giveBack: @escaping (MTLTexture) -> Void) {
        self.texture = texture
        self.giveBack = giveBack
    }

    deinit { giveBack(texture) }
}
#endif
