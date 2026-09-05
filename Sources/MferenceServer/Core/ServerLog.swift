import Foundation

/// Request lifecycle logging. Responses stay deliberately generic so runtime
/// details never reach a client; the operator needs the opposite, so the log
/// carries the underlying error verbatim.
///
/// Start and completion are both logged: a long prefill emits no output for
/// minutes, and without a start line that is indistinguishable from a wedged
/// server.
///
/// The lines themselves come from the `ServerLogSink` this wraps: the
/// command-line server keeps `StandardErrorServerLogSink`, an embedding host
/// supplies its own.
struct ServerLog: Sendable {
    private let sink: any ServerLogSink

    init(sink: any ServerLogSink = StandardErrorServerLogSink()) {
        self.sink = sink
    }

    func requestStarted(id: String,
                        method: String,
                        path: String,
                        requestedModel: String?,
                        streaming: Bool) {
        sink.record(ServerLogEvent(
            id: id, method: method, path: path, requestedModel: requestedModel,
            kind: .requestStarted(method: method, path: path, streaming: streaming)))
    }

    func requestCompleted(id: String,
                          method: String,
                          path: String,
                          requestedModel: String?,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        sink.record(ServerLogEvent(
            id: id, method: method, path: path, requestedModel: requestedModel,
            kind: .requestCompleted(duration: duration,
                                    promptTokens: usage.promptTokens,
                                    cachedTokens: usage.promptTokensDetails.cachedTokens,
                                    completionTokens: usage.completionTokens,
                                    finishReason: completion.finishReason)))
    }

    func requestFailed(id: String,
                       method: String,
                       path: String,
                       requestedModel: String?,
                       status: UInt,
                       streaming: Bool,
                       error: any Error) {
        let detail = switch error {
        case let error as ServerRequestError: Self.describe(error)
        default: String(describing: error)
        }
        sink.record(ServerLogEvent(
            id: id, method: method, path: path, requestedModel: requestedModel,
            kind: .requestFailed(status: status, streaming: streaming, detail: detail)))
    }

    func streamAborted(id: String,
                       method: String,
                       path: String,
                       requestedModel: String?,
                       reason: String) {
        sink.record(ServerLogEvent(
            id: id, method: method, path: path, requestedModel: requestedModel,
            kind: .streamAborted(reason: reason)))
    }

    /// A rejection decided on the event loop: the request never reached the
    /// backend and has no id.
    func requestRejected(method: String,
                         path: String,
                         requestedModel: String? = nil,
                         status: UInt,
                         envelope: OpenAIErrorEnvelope) {
        sink.record(ServerLogEvent(
            method: method, path: path, requestedModel: requestedModel,
            kind: .requestRejected(status: status,
                                   code: envelope.error.code,
                                   message: envelope.error.message)))
    }

    /// A served route with no request lifecycle of its own.
    func routed(method: String, path: String, status: UInt) {
        sink.record(ServerLogEvent(method: method, path: path, kind: .routed(status: status)))
    }

    private static func describe(_ error: ServerRequestError) -> String {
        let detail = error.envelope.error
        return "\(detail.code): \(detail.message)"
    }
}
