#if canImport(os)
import os
import VergenceCore

extension VergenceLog {
    /// Per-stage intervals for the 2D-to-3D pipeline (inference, stabilize,
    /// warp, transfer), for Instruments' os_signpost track.
    public static let signposter = OSSignposter(subsystem: subsystem, category: "Pseudo3D")
}
#endif
