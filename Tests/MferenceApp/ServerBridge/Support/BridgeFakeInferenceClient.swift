import Foundation
import Synchronization
// `@testable` for `AppTokenEvent`'s memberwise initializer, which is internal.
@testable import MferenceAppCore

/// Model-free `AppInferenceClient` for the bridge tests: streams a scripted
/// response word by word, hands every piece to `onTextDelta` the way the
/// decode-service client does, answers `cancel()` with a `.cancelled`
/// terminal event followed by a normal finish (the service's shape), and
/// counts every call that reaches it. Lives here rather than in
/// `Tests/MferenceApp/Core/Support` because test targets cannot share files.
final class BridgeFakeInferenceClient: AppInferenceClient, AppInferenceDeltaStreaming, Sendable {
    private struct State {
        var response: String
        var pieceDelay: Duration
        var prefillSteps: Int
        var promptTokenCount: Int?
        var active: (id: UUID, task: Task<Void, Never>)?
        var generateCount = 0
        var cancelCount = 0
        var requests: [AppGenerationRequest] = []
    }

    private let state: Mutex<State>

    init(response: String = "Simulated response from the fake decode session.",
         pieceDelay: Duration = .milliseconds(15),
         prefillSteps: Int = 2,
         promptTokenCount: Int? = nil) {
        state = Mutex(State(response: response,
                            pieceDelay: pieceDelay,
                            prefillSteps: prefillSteps,
                            promptTokenCount: promptTokenCount))
    }

    var generateCount: Int { state.withLock { $0.generateCount } }
    var cancelCount: Int { state.withLock { $0.cancelCount } }
    var requests: [AppGenerationRequest] { state.withLock { $0.requests } }
    var lastRequest: AppGenerationRequest? { state.withLock { $0.requests.last } }

    func setResponse(_ text: String) {
        state.withLock { $0.response = text }
    }

    /// The pieces `generate` streams for `text`: words with their leading space.
    static func pieces(of text: String) -> [String] {
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        return words.enumerated().map { index, word in
            index == 0 ? String(word) : " " + word
        }
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        generate(request, onTextDelta: { _ in })
    }

    func generate(_ request: AppGenerationRequest,
                  onTextDelta: @escaping @Sendable (String) -> Void)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let script = state.withLock { state -> (String, Duration, Int, Int?) in
                state.generateCount += 1
                state.requests.append(request)
                return (state.response, state.pieceDelay, state.prefillSteps, state.promptTokenCount)
            }
            do {
                try request.validate(requireModelDirectory: false)
            } catch {
                let appError = error as? AppInferenceError ?? .unknown("\(error)")
                continuation.yield(.failed(appError, partial: nil))
                continuation.finish(throwing: appError)
                return
            }
            let id = UUID()
            let task = Task { [self] in
                await stream(request, response: script.0, delay: script.1,
                             prefillSteps: script.2, promptTokenCount: script.3,
                             onTextDelta: onTextDelta, continuation: continuation)
                state.withLock { state in
                    if state.active?.id == id { state.active = nil }
                }
            }
            state.withLock { $0.active = (id, task) }
        }
    }

    func cancel() {
        let task = state.withLock { state -> Task<Void, Never>? in
            state.cancelCount += 1
            return state.active?.task
        }
        task?.cancel()
    }

    private func stream(_ request: AppGenerationRequest,
                        response: String,
                        delay: Duration,
                        prefillSteps: Int,
                        promptTokenCount: Int?,
                        onTextDelta: @escaping @Sendable (String) -> Void,
                        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        var generated = 0
        func diagnostics(_ reason: AppStopReason) -> AppDiagnostics {
            AppDiagnostics(generatedTokens: generated,
                           stopReason: reason,
                           promptTokenCount: promptTokenCount,
                           prefillSeconds: 0.01,
                           timeToFirstTokenSeconds: 0.01,
                           decodeSeconds: 0.05,
                           tokensPerSecond: 20,
                           peakMemoryBytes: nil,
                           runtimeOptions: request.runtimeOptions)
        }
        do {
            if prefillSteps > 0 {
                for step in 1...prefillSteps {
                    try await Task.sleep(for: delay)
                    try Task.checkCancellation()
                    continuation.yield(.prefillProgress(done: step, total: prefillSteps))
                }
            }
            let pieces = Self.pieces(of: response).prefix(request.maxNewTokens)
            for (index, piece) in pieces.enumerated() {
                try await Task.sleep(for: delay)
                try Task.checkCancellation()
                generated += 1
                onTextDelta(piece)
                // The chat-facing event carries text only for the first piece,
                // like the decode-service client's throttled `.token`.
                continuation.yield(.token(AppTokenEvent(
                    index: index,
                    textDelta: index == 0 ? piece : "",
                    elapsedDecodeSeconds: 0.01 * Double(index + 1))))
            }
            let reason: AppStopReason = generated >= request.maxNewTokens ? .maxTokens : .eos
            continuation.yield(.finished(diagnostics(reason)))
            continuation.finish()
        } catch {
            continuation.yield(.cancelled(diagnostics(.cancelled)))
            continuation.finish()
        }
    }
}
