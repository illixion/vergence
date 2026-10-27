import Foundation
import Testing
@testable import VergenceApple

struct ModelFileDownloaderTests {
    @Test func alreadyCancelledDownloaderResumesItsContinuationOnce() async {
        let task = Task {
            // Catching cancellation keeps the task's cancelled flag set while
            // exercising the downloader itself, rather than Task.sleep.
            do { try await Task.sleep(for: .milliseconds(10)) } catch { }
            return try await ModelFileDownloader(expectedBytes: 0, onFraction: { _ in })
                .download(from: URL(string: "https://invalid.example/model")!)
        }
        task.cancel()
        do { _ = try await task.value; Issue.record("Cancelled download succeeded") }
        catch { #expect(error is CancellationError) }
    }
}
