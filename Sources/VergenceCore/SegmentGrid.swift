/*
 Vergence - a fixed grid of stream segments

 A converted stream is cut at fixed times from the video's start (0, D, 2D,
 ...), never at the source's own keyframes, so a segment's number says where
 it is without an index, any segment can be made on its own, and segments
 made by different runs line up. Segment n holds the frames presented in
 [nD, (n+1)D); each starts with a keyframe the encoder is told to make.
 */

import Foundation

public struct SegmentGrid: Sendable, Equatable, Codable {
    public let segmentSeconds: Double
    public let durationSeconds: Double
    public let frameRate: Double
    /// When the last frame presents, if known; otherwise one frame before the end.
    public let lastFrameSeconds: Double?

    public init(segmentSeconds: Double, durationSeconds: Double, frameRate: Double, lastFrameSeconds: Double? = nil) {
        self.segmentSeconds = segmentSeconds
        self.durationSeconds = durationSeconds
        self.frameRate = frameRate
        self.lastFrameSeconds = lastFrameSeconds
    }

    /// Segments up to the one holding the last frame, so none is empty: a
    /// video that ends a frame after a boundary has no frame past it.
    public var count: Int {
        guard segmentSeconds > 0, durationSeconds > 0 else { return 0 }
        let frame = frameRate > 0 ? 1 / frameRate : 0
        let lastFrame = max(0, min(durationSeconds, lastFrameSeconds ?? durationSeconds - frame))
        return Int((lastFrame / segmentSeconds).rounded(.down)) + 1
    }

    public func start(_ segment: Int) -> Double { Double(segment) * segmentSeconds }

    /// How long segment `segment` plays; the last one is what is left.
    public func duration(_ segment: Int) -> Double {
        guard segment >= 0, segment < count else { return 0 }
        return segment == count - 1 ? max(0, durationSeconds - start(segment)) : segmentSeconds
    }

    /// The segment a time into the video falls in.
    public func segment(at seconds: Double) -> Int {
        guard count > 0 else { return 0 }
        return min(count - 1, max(0, Int((seconds / segmentSeconds).rounded(.down))))
    }

    /// An HLS media playlist of the whole grid, `uri(n)` naming segment n.
    public func mediaPlaylist(initURI: String, uri: (Int) -> String) -> String {
        var lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-INDEPENDENT-SEGMENTS",
                     "#EXT-X-TARGETDURATION:\(Int(segmentSeconds.rounded(.up)))", "#EXT-X-MEDIA-SEQUENCE:0",
                     "#EXT-X-MAP:URI=\"\(initURI)\""]
        for n in 0 ..< count {
            lines.append(String(format: "#EXTINF:%.5f,", duration(n)))
            lines.append(uri(n))
        }
        lines.append("#EXT-X-ENDLIST")
        return lines.joined(separator: "\n") + "\n"
    }
}
