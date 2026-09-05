import Foundation
@testable import MferenceAppCore

/// A client whose streams the test drives by hand: nothing is emitted until
/// the test says so, and every call that reaches the client is counted.
final class ScriptedInferenceClient: AppInferenceClient,
    AppInferenceTransportControlling, @unchecked Sendable {
    private let lock = NSLock()
    private var live: [AsyncThrowingStream<AppInferenceEvent, Error>.Continuation] = []
    private var maxOverlap = 0
    private var _generateCount = 0
    private var _cancelCount = 0
    private var _shutdownCount = 0

    /// When true, `cancel()` ends the live stream with a `.cancelled` event,
    /// the way the decode service answers a wire cancel.
    var cancelEmitsTerminal = true

    var generateCount: Int { lock.withLock { _generateCount } }
    var cancelCount: Int { lock.withLock { _cancelCount } }
    var shutdownCount: Int { lock.withLock { _shutdownCount } }
    var liveStreamCount: Int { lock.withLock { live.count } }
    /// The most streams ever open at once; the arbiter must keep this at 1.
    var maxConcurrentStreams: Int { lock.withLock { maxOverlap } }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            lock.withLock {
                _generateCount += 1
                live.append(continuation)
                maxOverlap = max(maxOverlap, live.count)
            }
        }
    }

    func cancel() {
        let continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation? = lock.withLock {
            _cancelCount += 1
            guard cancelEmitsTerminal, !live.isEmpty else { return nil }
            return live.removeFirst()
        }
        guard let continuation else { return }
        continuation.yield(.cancelled(Self.diagnostics(stopReason: .cancelled)))
        continuation.finish()
    }

    func shutdown() {
        lock.withLock { _shutdownCount += 1 }
    }

    func emitToken(_ text: String, index: Int = 0) {
        current?.yield(.token(AppTokenEvent(
            index: index, textDelta: text, elapsedDecodeSeconds: 0.01)))
    }

    func finishCurrent(_ stopReason: AppStopReason = .eos) {
        let continuation = takeCurrent()
        switch stopReason {
        case .cancelled:
            continuation?.yield(.cancelled(Self.diagnostics(stopReason: .cancelled)))
        default:
            continuation?.yield(.finished(Self.diagnostics(stopReason: stopReason)))
        }
        continuation?.finish()
    }

    func failCurrent(_ error: AppInferenceError) {
        let continuation = takeCurrent()
        continuation?.yield(.failed(error, partial: nil))
        continuation?.finish(throwing: error)
    }

    private var current: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation? {
        lock.withLock { live.first }
    }

    private func takeCurrent() -> AsyncThrowingStream<AppInferenceEvent, Error>.Continuation? {
        lock.withLock { live.isEmpty ? nil : live.removeFirst() }
    }

    static func diagnostics(stopReason: AppStopReason) -> AppDiagnostics {
        AppDiagnostics(generatedTokens: 1, stopReason: stopReason,
                       promptTokenCount: 1, prefillSeconds: 0,
                       timeToFirstTokenSeconds: 0, decodeSeconds: 0.01,
                       tokensPerSecond: 100, peakMemoryBytes: nil,
                       runtimeOptions: AppRuntimeOptions())
    }
}
