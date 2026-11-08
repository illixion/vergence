#if canImport(Metal)
import CoreVideo
import Foundation
import Metal

/// The motion an in-between frame was made along: at each of its pixels x,
/// the shift d with `mid(x) ≈ previous(x − d) ≈ current(x + d)`, in the
/// frame's pixels (half the motion from the previous real frame to the next).
/// `FlowMidFrameInterpolator` writes it when asked; VideoToolbox's
/// interpolator shows none.
///
/// AI 3D reads it to move the previous real frame's depth map to the
/// in-between frame (`MidFrameDepth`), so a moving subject's depth edge sits
/// on its picture edge instead of half a frame behind.
public final class MidFrameMotion: @unchecked Sendable {
    /// rg16Float at half the frame's size.
    public let texture: MTLTexture
    public let frameWidth: Int
    public let frameHeight: Int
    /// Presentation time of the real frame before, in seconds; set by the
    /// source that asked for it (`InterpolatingFrameSource`).
    public internal(set) var previousSeconds = Double.nan
    private let pool: MotionTexturePool?

    init(texture: MTLTexture, frameWidth: Int, frameHeight: Int, pool: MotionTexturePool?) {
        self.texture = texture
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.pool = pool
    }

    deinit { pool?.giveBack(texture) }
}

/// Half-size motion textures, reused: a mid frame's motion lives until the
/// pump draws it, up to the lookahead's half second.
final class MotionTexturePool: @unchecked Sendable {
    private let device: MTLDevice
    private let width: Int
    private let height: Int
    private let lock = NSLock()
    private var free: [MTLTexture] = []
    /// About the frames a half-second lookahead holds at 48 fps.
    private static let maxFree = 16

    init(device: MTLDevice, width: Int, height: Int) {
        self.device = device
        self.width = width
        self.height = height
    }

    func take(frameWidth: Int, frameHeight: Int) -> MidFrameMotion? {
        let reused: MTLTexture? = lock.withLock { free.popLast() }
        let texture = reused ?? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rg16Float, width: max(1, width), height: max(1, height), mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            return device.makeTexture(descriptor: descriptor)
        }()
        guard let texture else { return nil }
        return MidFrameMotion(texture: texture, frameWidth: frameWidth, frameHeight: frameHeight, pool: self)
    }

    func giveBack(_ texture: MTLTexture) {
        lock.withLock { if free.count < Self.maxFree { free.append(texture) } }
    }
}
#endif
