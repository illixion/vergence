import Foundation

/// Downloads large model assets to disk without holding them in memory.
/// Both depth and audio catalogs use this transport. Cancellation and delegate
/// completion race through one locked continuation, resumed exactly once.
public final class ModelFileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let expectedBytes: Int64
    private let onFraction: @Sendable (Double) -> Void
    private var lastReported = -1.0 // delegate queue only
    private let lock = NSLock()
    private var continuation: CheckedContinuation<URL, Error>?
    private var networkTask: URLSessionDownloadTask?
    private var cancelled = false
    private var session: URLSession!

    public init(expectedBytes: Int64, onFraction: @escaping @Sendable (Double) -> Void) {
        self.expectedBytes = expectedBytes; self.onFraction = onFraction
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }
    public func download(from url: URL) async throws -> URL {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let task = lock.withLock { () -> URLSessionDownloadTask? in
                    guard !cancelled else { return nil }
                    let task = session.downloadTask(with: url)
                    self.continuation = continuation; networkTask = task
                    return task
                }
                if let task { task.resume() }
                else { continuation.resume(throwing: CancellationError()); session.invalidateAndCancel() }
            }
        } onCancel: {
            let task = self.lock.withLock { self.cancelled = true; return self.networkTask }
            task?.cancel()
            _ = self.finish(.failure(CancellationError()))
            if task != nil { self.session.invalidateAndCancel() }
        }
    }
    @discardableResult private func finish(_ result: Result<URL, Error>) -> Bool {
        let continuation = lock.withLock {
            let value = self.continuation
            self.continuation = nil; networkTask = nil
            return value
        }
        continuation?.resume(with: result)
        return continuation != nil
    }
    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : expectedBytes
        guard total > 0 else { return }
        let fraction = min(1, Double(totalBytesWritten) / Double(total))
        if fraction - lastReported >= 0.01 || fraction >= 1 { lastReported = fraction; onFraction(fraction) }
    }
    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        if let response = downloadTask.response as? HTTPURLResponse, !(200...299).contains(response.statusCode) {
            finish(.failure(URLError(.badServerResponse))); return
        }
        let stable = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: stable)
            if !finish(.success(stable)) { try? FileManager.default.removeItem(at: stable) }
        } catch { finish(.failure(error)) }
    }
    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)) }
        session.finishTasksAndInvalidate()
    }
}
