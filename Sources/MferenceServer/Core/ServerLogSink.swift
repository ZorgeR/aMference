import Foundation

/// One request-lifecycle observation from the HTTP server. Every route emits
/// at least one event, including the rejections that never reach the
/// backend, so a sink sees the whole traffic profile rather than only the
/// requests that generated.
public struct ServerLogEvent: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case requestStarted(method: String, path: String, streaming: Bool)
        case requestCompleted(duration: Duration,
                              promptTokens: Int,
                              cachedTokens: Int,
                              completionTokens: Int,
                              finishReason: String)
        /// Synchronous rejections that never reach the backend: oversized
        /// bodies, wrong media type or method, unknown routes, malformed
        /// JSON, and validation failures.
        case requestRejected(status: UInt, code: String, message: String)
        /// A served route that produced no request lifecycle
        /// (`GET /health`, `GET /v1/models`).
        case routed(status: UInt)
        case requestFailed(status: UInt, streaming: Bool, detail: String)
        case streamAborted(reason: String)
    }

    /// The `chatcmpl-…` identifier when one exists. Rejections and served
    /// routes carry none.
    public let id: String?
    public let date: Date
    public let method: String
    public let path: String
    /// The `model` field as the client sent it, captured before validation
    /// so an `unknownModel` rejection can name what was actually requested.
    public let requestedModel: String?
    public let kind: Kind

    public init(id: String? = nil,
                date: Date = Date(),
                method: String,
                path: String,
                requestedModel: String? = nil,
                kind: Kind) {
        self.id = id
        self.date = date
        self.method = method
        self.path = path
        self.requestedModel = requestedModel
        self.kind = kind
    }
}

public protocol ServerLogSink: Sendable {
    /// Must not block: the server calls this on the NIO event loop at every
    /// synchronous rejection site, and from the generation task otherwise.
    func record(_ event: ServerLogEvent)
}

/// The command-line server's log: one line per lifecycle event on stderr.
///
/// Only the four lifecycle kinds the server has always logged are written;
/// rejections and served routes are dropped so the command-line server's
/// stderr keeps its established line set (a supervisor polling `/health`
/// would otherwise fill the log). An in-process sink that wants those
/// events records them itself.
public struct StandardErrorServerLogSink: ServerLogSink, Sendable {
    private let write: @Sendable (Data) -> Void

    public init() {
        self.init(write: { FileHandle.standardError.write($0) })
    }

    /// The same lines, delivered to `write` instead of standard error.
    init(write: @escaping @Sendable (Data) -> Void) {
        self.write = write
    }

    public func record(_ event: ServerLogEvent) {
        guard let line = Self.line(for: event) else { return }
        write(Data(line.utf8))
    }

    /// The exact bytes `record` writes for `event`, or `nil` when the event
    /// is one the standard-error log does not carry.
    public static func line(for event: ServerLogEvent) -> String? {
        let id = event.id ?? "-"
        let message: String
        switch event.kind {
        case .requestStarted(_, _, let streaming):
            message = "request \(id) started streaming=\(streaming)"
        case .requestCompleted(let duration, let promptTokens, let cachedTokens,
                               let completionTokens, let finishReason):
            message = """
            request \(id) completed in \(format(duration)) \
            prompt=\(promptTokens) \
            cached=\(cachedTokens) \
            completion=\(completionTokens) \
            finish=\(finishReason)
            """
        case .requestFailed(let status, let streaming, let detail):
            message = "request \(id) failed status=\(status) streaming=\(streaming) error=\(detail)"
        case .streamAborted(let reason):
            message = "request \(id) stream aborted: \(reason)"
        case .requestRejected, .routed:
            return nil
        }
        return "[\(event.date.formatted(.iso8601))] \(message)\n"
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.1fs", seconds)
    }
}
