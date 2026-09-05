import Foundation
import Synchronization
@testable import MferenceAppCore

/// Wraps any client and counts the calls that reach it, so a test can prove
/// that a UI action (cancel, generate) arrived at the client instead of
/// being swallowed by a layer in between.
final class RecordingInferenceClient: AppInferenceClient,
    AppInferenceTransportControlling, @unchecked Sendable {
    private let wrapped: any AppInferenceClient
    private let counts = Mutex((generate: 0, cancel: 0, disconnect: 0, shutdown: 0))

    init(wrapping wrapped: any AppInferenceClient) {
        self.wrapped = wrapped
    }

    var generateCount: Int { counts.withLock { $0.generate } }
    var cancelCount: Int { counts.withLock { $0.cancel } }
    var disconnectCount: Int { counts.withLock { $0.disconnect } }
    var shutdownCount: Int { counts.withLock { $0.shutdown } }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        counts.withLock { $0.generate += 1 }
        return wrapped.generate(request)
    }

    func cancel() {
        counts.withLock { $0.cancel += 1 }
        wrapped.cancel()
    }

    func disconnect() {
        counts.withLock { $0.disconnect += 1 }
        (wrapped as? any AppInferenceTransportControlling)?.disconnect()
    }

    func shutdown() {
        counts.withLock { $0.shutdown += 1 }
        (wrapped as? any AppInferenceTransportControlling)?.shutdown()
    }
}
