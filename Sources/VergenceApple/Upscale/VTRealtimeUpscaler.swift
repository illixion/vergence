#if canImport(VideoToolbox) && !targetEnvironment(simulator)
import CoreMedia
import CoreVideo
import Foundation
import QuartzCore
import VideoToolbox

/// VideoToolbox's low-latency super resolution
/// (`VTLowLatencySuperResolutionScalerConfiguration`, a learned upscaler the
/// OS ships), for playing a small picture sharper as each frame is shown.
/// The other upscaler here, `FrameUpscaler`, is Real-ESRGAN for `convert`:
/// far more work a frame, made ahead of time.
///
/// What the processor takes, measured on an M2 Max (macOS 27.2, 2026-11-08):
/// - factors 1.5, 2 and 4, each with its own input limit: 1280 a side at 1.5
///   (1,638,400 pixels), 960 at 2 (921,600), 480 at 4 (230,400). So what it
///   makes tops out at 1080p, and a 1080p picture gets nothing.
/// - `420v` in and out, nothing else. BGRA frames are converted in and the
///   result converted back, with the frame's own colour tags both ways.
/// - 0.7 ms a frame from 640×360 to 1280×720, 1.3 ms from 854×480 to
///   1708×960, 1.5 ms from 960×540 to 1080p, 2.5 ms from 1280×720 to 1080p.
/// - Starting a session loads its model: 0.5-2.3 s, so `prepare` is for a
///   work queue, never the main thread.
///
/// On real frames (a 4K demo's frames scaled to 1080p, then to 540p and back)
/// it brings back 2-4× the fine detail Lanczos does (high-pass energy 0.56 of
/// the original's against 0.21) and reads visibly sharper, at a PSNR up to a
/// few dB under Lanczos where it sharpens what the original had soft. On a
/// heavily compressed 540p frame it is no worse than Lanczos and a little
/// crisper.
///
/// One request at a time: `upscale` is called again only after the last
/// completion. Its completion runs on the processor's queue.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
public final class VTRealtimeUpscaler: RealtimeFrameUpscaler, @unchecked Sendable {
    public static var isSupported: Bool { VTLowLatencySuperResolutionScalerConfiguration.isSupported }

    /// The largest picture it makes: what Smooth Motion's VideoToolbox
    /// interpolator takes after it (1920 a side, 1920×1080 pixels), and the
    /// most the processor makes at 2× anyway.
    public static let maxOutputDimension = 1920
    public static let maxOutputPixels = 1920 * 1080

    /// The factor for a `width`×`height` picture: the largest the processor
    /// takes at that size whose result fits `maxOutputDimension` and
    /// `maxOutputPixels`. Nil where none does (720p's 1.5× is the last one
    /// that fits) or the processor isn't here.
    public static func factor(width: Int, height: Int) -> Float? {
        guard width > 0, height > 0, isSupported else { return nil }
        let fitting = VTLowLatencySuperResolutionScalerConfiguration
            .supportedScaleFactors(frameWidth: width, frameHeight: height)
            .filter { factor in
                let w = (Double(width) * Double(factor)).rounded(), h = (Double(height) * Double(factor)).rounded()
                return factor > 1 && w <= Double(maxOutputDimension) && h <= Double(maxOutputDimension)
                    && w * h <= Double(maxOutputPixels)
            }
        return fitting.max()
    }

    private var processor: VTFrameProcessor?
    private var inputSize = (width: 0, height: 0)
    private var output = (width: 0, height: 0)
    private var toYUV: VTPixelTransferSession?
    private var toBGRA: VTPixelTransferSession?
    private var yuvPool: CVPixelBufferPool?
    private var resultPool: CVPixelBufferPool?
    private var bgraPool: CVPixelBufferPool?
    private var timestamp: Int64 = 0

    public init() {}

    deinit { processor?.endSession() }

    public func outputSize(width: Int, height: Int) -> (width: Int, height: Int)? {
        guard let factor = Self.factor(width: width, height: height) else { return nil }
        return (Int((Double(width) * Double(factor)).rounded()), Int((Double(height) * Double(factor)).rounded()))
    }

    public func prepare(width: Int, height: Int) -> Bool {
        processor?.endSession()
        processor = nil
        guard let factor = Self.factor(width: width, height: height) else { return false }
        let configuration = VTLowLatencySuperResolutionScalerConfiguration(
            frameWidth: width, frameHeight: height, scaleFactor: factor)
        let format = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        guard configuration.supportedPixelFormats.contains(format) else {
            VergenceLog.pipeline.error("upscaling: the processor takes no 420v")
            return false
        }
        let processor = VTFrameProcessor()
        do {
            try processor.startSession(configuration: configuration)
        } catch {
            VergenceLog.pipeline.error("upscaling: session failed: \(error, privacy: .public)")
            return false
        }
        // The processor's own attributes: they pad the planes (ExtendedPixelsBottom),
        // and a buffer without that padding fails every request (-19730).
        let source = configuration.sourcePixelBufferAttributes as [String: Any]
        let destination = configuration.destinationPixelBufferAttributes as [String: Any]
        let outWidth = (destination[kCVPixelBufferWidthKey as String] as? Int)
            ?? Int((Double(width) * Double(factor)).rounded())
        let outHeight = (destination[kCVPixelBufferHeightKey as String] as? Int)
            ?? Int((Double(height) * Double(factor)).rounded())
        func pool(_ attributes: [String: Any], width: Int, height: Int, format: OSType) -> CVPixelBufferPool? {
            var attributes = attributes
            attributes[kCVPixelBufferPixelFormatTypeKey as String] = format
            attributes[kCVPixelBufferWidthKey as String] = width
            attributes[kCVPixelBufferHeightKey as String] = height
            if attributes[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
                attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]()
            }
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
            return pool
        }
        var toYUV: VTPixelTransferSession?
        var toBGRA: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &toYUV)
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &toBGRA)
        let yuvPool = pool(source, width: width, height: height, format: format)
        let resultPool = pool(destination, width: outWidth, height: outHeight, format: format)
        let bgraPool = pool([kCVPixelBufferMetalCompatibilityKey as String: true],
                            width: outWidth, height: outHeight, format: kCVPixelFormatType_32BGRA)
        guard let toYUV, let toBGRA, yuvPool != nil, resultPool != nil, bgraPool != nil else {
            processor.endSession()
            return false
        }
        self.processor = processor
        self.toYUV = toYUV
        self.toBGRA = toBGRA
        self.yuvPool = yuvPool
        self.resultPool = resultPool
        self.bgraPool = bgraPool
        inputSize = (width, height)
        output = (outWidth, outHeight)
        VergenceLog.pipeline.notice(
            "upscaling: \(width)×\(height) → \(outWidth)×\(outHeight) (\(factor, format: .fixed(precision: 1), privacy: .public)×)"
        )
        return true
    }

    public func upscale(_ frame: CVPixelBuffer, completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
        let started = CACurrentMediaTime()
        guard let processor, let toYUV, let toBGRA, let bgraPool,
              CVPixelBufferGetPixelFormatType(frame) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(frame) == inputSize.width, CVPixelBufferGetHeight(frame) == inputSize.height,
              let yuv = makeBuffer(yuvPool), let result = makeBuffer(resultPool) else {
            completion(nil, .nan)
            return
        }
        copyColorTags(from: frame, to: yuv)
        copyColorTags(from: frame, to: result)
        guard VTPixelTransferSessionTransferImage(toYUV, from: frame, to: yuv) == noErr else {
            completion(nil, .nan)
            return
        }
        // The times only label the request: each frame stands alone.
        timestamp += 1
        let time = CMTime(value: timestamp, timescale: 600)
        guard let sourceFrame = VTFrameProcessorFrame(buffer: yuv, presentationTimeStamp: time),
              let destinationFrame = VTFrameProcessorFrame(buffer: result, presentationTimeStamp: time) else {
            completion(nil, .nan)
            return
        }
        let parameters = VTLowLatencySuperResolutionScalerParameters(sourceFrame: sourceFrame,
                                                                     destinationFrame: destinationFrame)
        let job = Job(frame: frame, result: result, toBGRA: toBGRA, bgraPool: bgraPool)
        processor.process(parameters: parameters) { _, error in
            if let error {
                VergenceLog.pipeline.error("upscaling failed: \(error, privacy: .public)")
                completion(nil, (CACurrentMediaTime() - started) * 1000)
                return
            }
            var out: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, job.bgraPool, &out)
            guard let out, VTPixelTransferSessionTransferImage(job.toBGRA, from: job.result, to: out) == noErr else {
                completion(nil, (CACurrentMediaTime() - started) * 1000)
                return
            }
            CVBufferPropagateAttachments(job.frame, out)
            completion(out, (CACurrentMediaTime() - started) * 1000)
        }
    }

    private struct Job: @unchecked Sendable {
        let frame: CVPixelBuffer
        let result: CVPixelBuffer
        let toBGRA: VTPixelTransferSession
        let bgraPool: CVPixelBufferPool
    }

    private func makeBuffer(_ pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        return buffer
    }

    /// The frame's primaries and transfer function (so neither conversion
    /// changes its colour), and one fixed matrix both ways.
    private func copyColorTags(from source: CVPixelBuffer, to dest: CVPixelBuffer) {
        for key in [kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey] {
            if let value = CVBufferCopyAttachment(source, key, nil) {
                CVBufferSetAttachment(dest, key, value, .shouldPropagate)
            }
        }
        let matrix = CVBufferCopyAttachment(source, kCVImageBufferYCbCrMatrixKey, nil)
            ?? kCVImageBufferYCbCrMatrix_ITU_R_709_2
        CVBufferSetAttachment(dest, kCVImageBufferYCbCrMatrixKey, matrix, .shouldPropagate)
    }
}
#endif
