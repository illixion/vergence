/// Briefly holds a clocked player when real-time depth falls behind the
/// playhead. The timeout and cooldown prevent a sustained overload from
/// repeatedly stopping playback.
public struct StereoProcessingHoldPolicy: Sendable {
    public let triggerDelay: Double
    public let maximumHold: Double
    public let cooldown: Double
    private(set) public var isHolding = false
    private var holdingSince: Double?
    private var cooldownUntil = 0.0

    public init(triggerDelay: Double = 0.12, maximumHold: Double = 2, cooldown: Double = 5) {
        self.triggerDelay = triggerDelay
        self.maximumHold = maximumHold
        self.cooldown = cooldown
    }

    /// `waitingSince` is the take time of the oldest unresolved frame that
    /// should already be visible. Resolving inference releases the hold.
    public mutating func update(now: Double, clockRunning: Bool, waitingSince: Double?) -> Bool {
        if isHolding {
            if waitingSince == nil {
                isHolding = false
                holdingSince = nil
            } else if let holdingSince, now - holdingSince >= maximumHold {
                isHolding = false
                self.holdingSince = nil
                cooldownUntil = now + cooldown
            }
            return isHolding
        }
        guard clockRunning, now >= cooldownUntil, let waitingSince,
              now - waitingSince >= triggerDelay else { return false }
        isHolding = true
        holdingSince = now
        return true
    }

    public mutating func reset() {
        isHolding = false
        holdingSince = nil
        cooldownUntil = 0
    }
}
