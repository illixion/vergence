/// A bounded reserve of source pictures and their inferred depth. Stereo eyes
/// are rendered only close to presentation, so the reserve does not retain two
/// full eye buffers for every future picture.
public enum StereoLookaheadPolicy {
    public static let minimumLead = 0.75
    public static let maximumFrames = 60
    public static func lead(after current: Double, shown: Int, late: Int) -> Double {
        let fraction = Double(late) / Double(max(1, shown))
        if fraction > 0.05 { return min(1.5, current + 0.1) }
        if fraction < 0.01 { return max(minimumLead, current - 0.01) }
        return current
    }
    public static func capacity(width: Int, height: Int) -> Int {
        guard width > 0, height > 0, width <= 32768, height <= 32768 else { return 1 }
        // 256 MiB total: one BGRA source and ~1 MiB depth per queued frame;
        // reserve three stereo pairs for the short warp/presentation window.
        let pixels = width * height
        let available = max(0, 256 * 1024 * 1024 - pixels * 8 * 3)
        return max(1, min(maximumFrames, available / (pixels * 4 + 1024 * 1024)))
    }
    public static func reserve(capacity: Int, spacing: Double, rate: Double, lead: Double) -> Double {
        // Leave room for the current frame. Never
        // ask a small, high-resolution queue for a reserve it cannot hold.
        min(lead * rate, Double(max(0, capacity - 1)) * spacing * max(1, rate))
    }
}
