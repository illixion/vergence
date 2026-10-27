/// Hysteresis over contiguous prepared depth on the media timeline. A pause
/// keeps inference running until a reserve has refilled. Timeout releases the
/// transport and reports sustained overload so a host can lower quality.
///
/// Not every source can refill while paused: a paused `AVPlayer` hands out
/// the frame at the playhead and what it had already decoded (about 0.4 s
/// mid-play, nothing more before it first plays), never the whole reserve.
/// Such a source says `sourceExhausted` once everything it can give has its
/// depth, and the hold ends there; waiting for the reserve instead ran every
/// hold into the timeout (8 s of a still picture at every AI 3D start, and an
/// overload report that lowered the quality; device, 2026-10-27).
public struct StereoProcessingHoldPolicy: Sendable {
    public let maximumHold: Double
    public let cooldown: Double
    /// Holds within the 30 s window that also count as sustained overload.
    /// A source that can't refill while paused makes short holds, which would
    /// take dozens to spend `maximumHold`.
    public let maximumRecoveries: Int
    public private(set) var isHolding = false
    public private(set) var recoveryExhausted = false
    private var holdingSince: Double?
    private var cooldownUntil = 0.0
    private var priming = true
    private var recoveryWindowStart: Double?
    private var recoverySpent = 0.0
    private var recoveryCount = 0

    public init(maximumHold: Double = 8, cooldown: Double = 10, maximumRecoveries: Int = 6) {
        self.maximumHold = maximumHold
        self.cooldown = cooldown
        self.maximumRecoveries = maximumRecoveries
    }

    /// - Parameter sourceExhausted: the clock is stopped and every frame the
    ///   source can hand out has its depth, so no more can be prepared until
    ///   it runs (see the type's note).
    public mutating func update(now: Double, clockRunning: Bool, readySeconds: Double,
                                resumeSeconds: Double, lowWaterSeconds: Double,
                                sourceExhausted: Bool = false) -> Bool {
        if isHolding {
            if readySeconds + 0.001 >= resumeSeconds || (sourceExhausted && !clockRunning) {
                isHolding = false
                recordRecovery(now: now)
                holdingSince = nil
                priming = false
                if recoveryCount >= 2, recoverySpent >= maximumHold || recoveryCount >= maximumRecoveries {
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
        // Stopped with nothing left to prepare: holding would only wait.
        if sourceExhausted, !clockRunning { return false }
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

    public mutating func reset() {
        self = Self(maximumHold: maximumHold, cooldown: cooldown, maximumRecoveries: maximumRecoveries)
    }
}
