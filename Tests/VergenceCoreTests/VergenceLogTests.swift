import Foundation
import Testing
@testable import VergenceCore

@Test func interpolatedValuesKeepTheirPrivacy() {
    let name = "film.mkv", count = 3, token = "abc"
    let message: VergenceLogMessage = "opened \(name) with \(count) tracks \(token, privacy: .sensitive) \(1.5, format: .fixed(precision: 1))"
    #expect(message.segments == [
        .text("opened "), .value("film.mkv", .private), .text(" with 3 tracks "), .value("abc", .sensitive), .text(" 1.5"),
    ])
    #expect(!message.isPublic)
    #expect(message.redacted == "opened <private> with 3 tracks <private> 1.5")
}

@Test func publicValuesAndOptionalsRenderInline() {
    let state = "ready", missing: Int? = nil
    let message: VergenceLogMessage = "state \(state, privacy: .public), frames \(missing)"
    #expect(message.isPublic)
    #expect(message.redacted == "state ready, frames nil")
}

@Test func hashedValuesMatchWithinASession() {
    let id = "item-42"
    let a: VergenceLogMessage = "\(id, privacy: .private(mask: .hash))"
    let b: VergenceLogMessage = "\(id, privacy: .private(mask: .hash))"
    #expect(a.redacted == b.redacted)
    #expect(a.redacted.hasPrefix("<hash:") && !a.redacted.contains(id))
}

@Test func aSinkReceivesStructuredEvents() {
    let received = VergenceLock<[VergenceLogEvent]>([])
    VergenceLog.sink = { event in received.withLock { $0.append(event) } }
    defer { VergenceLog.sink = nil }
    let path = "/private/path"
    VergenceLog.cache.error("evicted \(path)")
    let events = received.withLock { $0 }
    #expect(events.count == 1)
    #expect(events.first?.category == "DepthCache" && events.first?.level == .error)
    #expect(events.first?.message.segments.last == .value(path, .private))
}
