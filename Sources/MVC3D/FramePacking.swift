/*
 mvc3d - packing a picture pair into one frame

 Full-resolution frame packing: top-and-bottom doubles the height, side by
 side doubles the width, left eye first either way. The output is planar
 I420, the layout ffmpeg calls yuv420p, so `ffmpeg -f rawvideo` and the
 hardware encoders take it directly.
 */

import Foundation

public enum StereoPacking: String, Sendable, CaseIterable {
    case topAndBottom = "tab"
    case sideBySide = "sbs"

    public func packedSize(width: Int, height: Int) -> (width: Int, height: Int) {
        switch self {
        case .topAndBottom: (width, height * 2)
        case .sideBySide: (width * 2, height)
        }
    }
}

extension StereoFrame {
    /// Bytes of the packed I420 frame.
    public func packedByteCount(_ packing: StereoPacking) -> Int {
        let (w, h) = packing.packedSize(width: width, height: height)
        return w * h + 2 * (w / 2) * (h / 2)
    }

    /// Writes the packed frame into `destination`, which must hold
    /// `packedByteCount` bytes. A mono frame is packed with itself.
    public func pack(_ packing: StereoPacking, into destination: UnsafeMutableRawBufferPointer) {
        let right = self.right ?? left
        var out = destination.baseAddress!.assumingMemoryBound(to: UInt8.self)
        for plane in 0 ..< 3 {
            let l = left[plane], r = right[plane]
            switch packing {
            case .topAndBottom:
                for view in [l, r] {
                    for y in 0 ..< view.height {
                        out.update(from: view.base + y * view.stride, count: view.width)
                        out += view.width
                    }
                }
            case .sideBySide:
                for y in 0 ..< l.height {
                    out.update(from: l.base + y * l.stride, count: l.width)
                    out += l.width
                    out.update(from: r.base + y * r.stride, count: r.width)
                    out += r.width
                }
            }
        }
    }
}
