/// Hysteresis over contiguous prepared depth on the media timeline. A pause
/// keeps inference running until a reserve has refilled. Timeout releases the
/// transport and reports sustained overload so a host can lower quality.
public struct StereoProcessingHoldPolicy: Sendable {
    public let maximumHold: Double
    public let cooldown: Double
    public private(set) var isHolding = false
    public private(set) var recoveryExhausted = false
    private var holdingSince: Double?
    private var cooldownUntil = 0.0
    private var priming = true
    private var recoveryWindowStart: Double?
    private var recoverySpent = 0.0
    private var recoveryCount = 0

    public init(maximumHold: Double = 8, cooldown: Double = 10) {
        self.maximumHold = maximumHold
        self.cooldown = cooldown
    }

    public mutating func update(now: Double, clockRunning: Bool, readySeconds: Double,
                                resumeSeconds: Double, lowWaterSeconds: Double) -> Bool {
        if isHolding {
            if readySeconds + 0.001 >= resumeSeconds {
                isHolding = false
                recordRecovery(now: now)
                holdingSince = nil
                priming = false
                if recoveryCount >= 2, recoverySpent >= maximumHold {
                    recoveryExhausted = true
                    cooldownUntil = now + cooldown
                }
            } else if let holdingSince, now - holdingSince >= maximumHold {
                isHolding = false
                self.holdingSince = nil
                recoveryExhausted = true
                priming = false
                cooldownUntil = now + cooldown
            }
            return isHolding
        }
        guard now >= cooldownUntil, (clockRunning || priming),
              readySeconds + 0.001 < (priming ? resumeSeconds : lowWaterSeconds) else { return false }
        if recoveryWindowStart.map({ now - $0 >= 30 }) ?? true {
            recoveryWindowStart = now
            recoverySpent = 0
            recoveryCount = 0
        }
        recoveryExhausted = false
        isHolding = true
        holdingSince = now
        return true
    }

    private mutating func recordRecovery(now: Double) {
        recoverySpent += max(0, now - (holdingSince ?? now))
        recoveryCount += 1
    }

    public mutating func reset() { self = Self(maximumHold: maximumHold, cooldown: cooldown) }
}
