/// Bounded processing lead; presentation always stays on the source clock.
/// Large pictures receive fewer retained frames rather than a larger memory
/// allowance. One frame must fit even when it exceeds the nominal budget.
public enum StereoLookaheadPolicy {
    public static let minimumLead = 0.12
    public static func lead(after current: Double, shown: Int, late: Int) -> Double {
        let fraction = Double(late) / Double(max(1, shown))
        if fraction > 0.05 { return min(0.25, current + 0.025) }
        if fraction < 0.01 { return max(minimumLead, current - 0.005) }
        return current
    }
    public static func capacity(width: Int, height: Int) -> Int {
        guard width > 0, height > 0, width <= 32768, height <= 32768 else { return 1 }
        return max(1, min(16, (192 * 1024 * 1024) / (width * height * 8)))
    }
}
