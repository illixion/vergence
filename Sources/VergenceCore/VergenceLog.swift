/*
 Vergence - logging

 Messages are written like `os.Logger`'s, privacy labels included:

     VergenceLog.pipeline.info("loaded \(name, privacy: .public) in \(ms) ms")

 and keep each interpolated value apart with its privacy, so a host that
 routes them elsewhere (an in-app console, a debug trace) can apply its own
 redaction rules value by value instead of receiving finished text.

 With no `sink`, lines go to the unified log (`os.Logger`) where it exists:
 the message with hidden values replaced (`<private>`, or a salted hash for
 `mask: .hash`) as public text, followed by the private values as a private
 argument, which the OS shows only where private data is enabled. Sensitive
 values never leave the process. A host that sets `sink` takes over all
 output, so nothing is logged twice.

 The rules, as os_log has them: `.public` only for values the code itself
 defines (states, counts, codes); `.auto` (the default) makes numbers and
 booleans public and everything else private; `.sensitive` for credentials.
 */

import Foundation
#if canImport(os)
import os
#endif
#if canImport(CryptoKit)
import CryptoKit
#endif

/// How an interpolated value may be shown, spelled like `OSLogPrivacy`.
public struct VergenceLogPrivacy: Sendable, Equatable {
    public enum Level: Sendable, Equatable {
        case auto, `public`, `private`, sensitive
    }

    public enum Mask: Sendable, Equatable {
        case none, hash
    }

    public let level: Level
    public let mask: Mask

    public static let auto = VergenceLogPrivacy(level: .auto, mask: .none)
    public static let `public` = VergenceLogPrivacy(level: .public, mask: .none)
    public static let `private` = VergenceLogPrivacy(level: .private, mask: .none)
    public static let sensitive = VergenceLogPrivacy(level: .sensitive, mask: .none)

    public static func `private`(mask: Mask) -> VergenceLogPrivacy { VergenceLogPrivacy(level: .private, mask: mask) }
    public static func sensitive(mask: Mask) -> VergenceLogPrivacy { VergenceLogPrivacy(level: .sensitive, mask: mask) }

    fileprivate func resolved(scalar: Bool) -> VergenceLogPrivacy {
        guard level == .auto else { return self }
        return VergenceLogPrivacy(level: scalar ? .public : .private, mask: mask)
    }
}

/// A log message whose interpolated values carry their privacy.
public struct VergenceLogMessage: Sendable, ExpressibleByStringInterpolation {
    public enum Segment: Sendable, Equatable {
        /// Literal text and public values.
        case text(String)
        /// A value that isn't public, with its (resolved) privacy.
        case value(String, VergenceLogPrivacy)
    }

    public private(set) var segments: [Segment]

    public init(stringLiteral value: String) {
        segments = [.text(value)]
    }

    public init(stringInterpolation: StringInterpolation) {
        segments = stringInterpolation.segments
    }

    public struct StringInterpolation: StringInterpolationProtocol {
        var segments: [Segment] = []

        public init(literalCapacity: Int, interpolationCount: Int) {
            segments.reserveCapacity(interpolationCount * 2 + 1)
        }

        public mutating func appendLiteral(_ literal: String) {
            guard !literal.isEmpty else { return }
            appendText(literal)
        }

        public mutating func appendInterpolation<T>(_ value: T, privacy: VergenceLogPrivacy = .auto) {
            let scalar = value is any BinaryInteger || value is any BinaryFloatingPoint || value is Bool
            append(String(describing: value), privacy.resolved(scalar: scalar))
        }

        public mutating func appendInterpolation<T: BinaryFloatingPoint>(
            _ value: T, format: VergenceLogFloatFormat, privacy: VergenceLogPrivacy = .auto
        ) {
            append(format.render(Double(value)), privacy.resolved(scalar: true))
        }

        /// An error's domain and code are public; its description follows `privacy`.
        public mutating func appendInterpolation(_ error: some Error, privacy: VergenceLogPrivacy = .auto) {
            let bridged = error as NSError
            appendText("\(bridged.domain) \(bridged.code): ")
            append(bridged.localizedDescription, privacy.resolved(scalar: false))
        }

        /// An optional prints its value, or `nil`.
        public mutating func appendInterpolation<T>(_ value: T?, privacy: VergenceLogPrivacy = .auto) {
            guard let value else {
                appendText("nil")
                return
            }
            appendInterpolation(value, privacy: privacy)
        }

        private mutating func append(_ text: String, _ privacy: VergenceLogPrivacy) {
            if privacy.level == .public {
                appendText(text)
            } else {
                segments.append(.value(text, privacy))
            }
        }

        private mutating func appendText(_ text: String) {
            if case .text(let previous)? = segments.last {
                segments[segments.count - 1] = .text(previous + text)
            } else {
                segments.append(.text(text))
            }
        }
    }

    /// True when every value is public.
    public var isPublic: Bool {
        !segments.contains { if case .value = $0 { true } else { false } }
    }

    /// The message with every hidden value replaced by `<private>` or its
    /// salted hash.
    public var redacted: String {
        segments.reduce(into: "") { out, segment in
            switch segment {
            case .text(let text): out += text
            case .value(let value, let privacy): out += Self.placeholder(for: value, privacy)
            }
        }
    }

    /// The private (not sensitive) values, in order, for the unified log's
    /// private argument.
    fileprivate var privateValues: [String] {
        segments.compactMap {
            if case .value(let value, let privacy) = $0, privacy.level == .private { value } else { nil }
        }
    }

    static func placeholder(for value: String, _ privacy: VergenceLogPrivacy) -> String {
        switch privacy.mask {
        case .none: "<private>"
        case .hash: "<hash:\(VergenceLogSalt.hash(value))>"
        }
    }
}

/// Float formatting, spelled like `OSLogFloatFormatting`.
public enum VergenceLogFloatFormat: Sendable {
    case fixed(precision: Int)
    case exponential(precision: Int)

    public static var fixed: VergenceLogFloatFormat { .fixed(precision: 6) }

    func render(_ value: Double) -> String {
        switch self {
        case .fixed(let precision): String(format: "%.\(max(0, precision))f", value)
        case .exponential(let precision): String(format: "%.\(max(0, precision))e", value)
        }
    }
}

/// One line on its way out.
public struct VergenceLogEvent: Sendable {
    public enum Level: Sendable {
        case debug, info, notice, error, fault
    }

    public let category: String
    public let level: Level
    public let message: VergenceLogMessage
}

/// A category's logger. Same method names as `os.Logger`.
public struct VergenceLogger: Sendable {
    public let category: String

    public init(category: String) {
        self.category = category
    }

    public func debug(_ message: @autoclosure () -> VergenceLogMessage) { write(.debug, message) }
    public func info(_ message: @autoclosure () -> VergenceLogMessage) { write(.info, message) }
    public func notice(_ message: @autoclosure () -> VergenceLogMessage) { write(.notice, message) }
    public func log(_ message: @autoclosure () -> VergenceLogMessage) { write(.notice, message) }
    public func warning(_ message: @autoclosure () -> VergenceLogMessage) { write(.error, message) }
    public func error(_ message: @autoclosure () -> VergenceLogMessage) { write(.error, message) }
    public func fault(_ message: @autoclosure () -> VergenceLogMessage) { write(.fault, message) }

    private func write(_ level: VergenceLogEvent.Level, _ message: () -> VergenceLogMessage) {
        VergenceLog.emit(VergenceLogEvent(category: category, level: level, message: message()))
    }
}

public enum VergenceLog {
    /// Depth cache: conversions, eviction, entry lifecycle.
    public static let cache = VergenceLogger(category: "DepthCache")
    /// The live pipeline: model load, engage decisions, playback fit.
    public static let pipeline = VergenceLogger(category: "Pseudo3D")

    /// The unified log's subsystem: the host app's, so lines sort with its own.
    public static let subsystem = Bundle.main.bundleIdentifier ?? "vergence"

    /// Where lines go instead of the unified log. Set once, at launch.
    public static var sink: (@Sendable (VergenceLogEvent) -> Void)? {
        get { state.withLock { $0 } }
        set { state.withLock { $0 = newValue } }
    }

    private static let state = VergenceLock<(@Sendable (VergenceLogEvent) -> Void)?>(nil)

    static func emit(_ event: VergenceLogEvent) {
        if let sink { return sink(event) }
        #if canImport(os)
        let logger = Logger(subsystem: subsystem, category: event.category)
        let type: OSLogType = switch event.level {
        case .debug: .debug
        case .info: .info
        case .notice: .default
        case .error: .error
        case .fault: .fault
        }
        let text = event.message.redacted
        let hidden = event.message.privateValues
        if hidden.isEmpty {
            logger.log(level: type, "\(text, privacy: .public)")
        } else {
            logger.log(level: type, "\(text, privacy: .public) \(hidden.joined(separator: " | "), privacy: .private)")
        }
        #endif
    }
}

/// The per-launch salt behind `mask: .hash`, so one value is recognisable
/// across a session's lines but two sessions can't be joined on it.
enum VergenceLogSalt {
    private static let salt = UUID().uuidString

    static func hash(_ value: String) -> String {
        #if canImport(CryptoKit)
        let digest = SHA256.hash(data: Data((salt + value).utf8))
        return digest.prefix(4).map { String(format: "%02x", $0) }.joined()
        #else
        var h: UInt64 = 0xcbf2_9ce4_8422_2325 // FNV-1a
        for byte in (salt + value).utf8 { h = (h ^ UInt64(byte)) &* 0x100_0000_01b3 }
        return String(format: "%08x", UInt32(truncatingIfNeeded: h))
        #endif
    }
}

/// A minimal lock around a value, for state read from any thread.
public final class VergenceLock<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = NSLock()

    public init(_ value: Value) {
        self.value = value
    }

    public func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R {
        lock.lock()
        defer { lock.unlock() }
        return try body(&value)
    }
}
