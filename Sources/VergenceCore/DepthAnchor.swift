/*
 Vergence - the CPU side of depth anchoring

 What the anchor reduction (a GPU pass in the Apple backend) reports about a
 frame, and the background ceiling the next reduction is told about. See
 DepthRefinement.swift for how anchoring works.
 */

/// The previous frame's background layer, in its own units: pixels it put
/// nearer than `ceiling` don't anchor the next fit.
public struct DepthAnchorCeiling: Sendable {
    public var ceiling: Float
    public var softness: Float

    public static let none = DepthAnchorCeiling(ceiling: .greatestFiniteMagnitude, softness: 1)

    /// Everything the frame's shot-space `mapping` put farther than
    /// `DepthAlignmentSums.anchorCeiling`, with an edge 0.05 of shot space wide.
    public init(mapping: DepthAffine) {
        guard let inverse = mapping.inverse else {
            self = .none
            return
        }
        ceiling = inverse(DepthAlignmentSums.anchorCeiling)
        softness = 0.05 / mapping.a
    }

    public init(ceiling: Float, softness: Float) {
        self.ceiling = ceiling
        self.softness = softness
    }
}

/// What one anchor reduction says about a frame.
public struct DepthAnchorStats: Sendable {
    public var sums: DepthAlignmentSums
    /// Mean absolute luma change against the previous frame (nil without one).
    public var meanLumaDelta: Float?
    public var min: Float
    public var max: Float

    public init(sums: DepthAlignmentSums, meanLumaDelta: Float?, min: Float, max: Float) {
        self.sums = sums
        self.meanLumaDelta = meanLumaDelta
        self.min = min
        self.max = max
    }
}
