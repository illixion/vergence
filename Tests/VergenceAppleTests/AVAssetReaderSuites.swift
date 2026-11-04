#if os(macOS)
import Testing

/// The suites that read media through `AVAssetReader`, run one at a time.
///
/// `copyNextSampleBuffer()` blocks its thread, and these tests call it from
/// Swift concurrency's cooperative pool (one thread per core). Run in parallel,
/// a dozen of them took every thread at once, each waiting on CoreMedia work
/// that then had nowhere to run, and `swift test` hung for good (2026-11-04, a
/// 12-core Mac). Serialised they hold at most one thread and take ~1.5 s.
@Suite(.serialized) enum AVAssetReaderSuites {}
#endif
