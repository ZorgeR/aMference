import Foundation
import Synchronization
import MferenceAppCore
import MferenceServerCore

/// A log row only the bridge knows about: lifecycle of the listener, queue
/// waits behind the chat UI, sampling substitutions, stop-string matches,
/// and session changes. Server rows and bridge rows go through the same sink,
/// so their relative order is preserved.
public struct APIServerBridgeNote: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case listening(host: String, port: Int, modelID: String)
        case stopped(reason: String)
        case startFailed(String)
        /// A request is waiting for the decode session held by `behind`.
        case waitingForSession(behind: AppInferenceOwner)
        case substitution(APIServerSubstitution)
        case stopStringMatched
        /// A completion was refused because no model is loaded.
        case modelNotLoaded
        case activeRequestCancelled
        case modelUnloaded
        case modelChanged(String)
    }

    public let date: Date
    public let kind: Kind

    public init(date: Date = Date(), kind: Kind) {
        self.date = date
        self.kind = kind
    }
}

public struct APIServerLogEntry: Identifiable, Equatable, Sendable {
    public enum Payload: Equatable, Sendable {
        case server(ServerLogEvent)
        case bridge(APIServerBridgeNote)
    }

    /// Monotonic per sink; survives ring drops, so a row keeps its identity
    /// in the list while older rows fall off the front.
    public let id: UInt64
    public let payload: Payload

    public init(id: UInt64, payload: Payload) {
        self.id = id
        self.payload = payload
    }

    public var date: Date {
        switch payload {
        case .server(let event): event.date
        case .bridge(let note): note.date
        }
    }
}

public struct APIServerLogSnapshot: Equatable, Sendable {
    public let entries: [APIServerLogEntry]
    /// Rows dropped from the front of the ring since the last `clear()`.
    public let droppedBefore: Int

    public init(entries: [APIServerLogEntry], droppedBefore: Int) {
        self.entries = entries
        self.droppedBefore = droppedBefore
    }

    public static let empty = APIServerLogSnapshot(entries: [], droppedBefore: 0)
}

/// Bounded ring of log rows shared by the HTTP server (as its
/// `ServerLogSink`) and the bridge. `record` is called on the NIO event loop
/// at several rejection sites, so it only appends under a mutex and, when no
/// flush is already pending, schedules exactly one MainActor hop after
/// `flushDelay`; a client hammering 404s produces a handful of observable
/// mutations per second rather than hundreds. Nothing is formatted here.
public final class APIServerLogSink: ServerLogSink, Sendable {
    public static let defaultCapacity = 500

    private struct State {
        var entries: [APIServerLogEntry] = []
        var droppedBefore = 0
        var nextID: UInt64 = 1
        var flushScheduled = false
        var onFlush: (@MainActor @Sendable () -> Void)?
    }

    private let state = Mutex(State())
    public let capacity: Int
    public let flushDelay: Duration

    public init(capacity: Int = APIServerLogSink.defaultCapacity,
                flushDelay: Duration = .milliseconds(100)) {
        self.capacity = max(1, capacity)
        self.flushDelay = flushDelay
    }

    /// Runs on the MainActor after each coalesced batch of appends. Set once
    /// by the owner; `nil` disables flushing (rows still accumulate).
    public func setFlushHandler(_ handler: (@MainActor @Sendable () -> Void)?) {
        state.withLock { $0.onFlush = handler }
    }

    public func record(_ event: ServerLogEvent) {
        append(.server(event))
    }

    public func record(_ note: APIServerBridgeNote) {
        append(.bridge(note))
    }

    public var snapshot: APIServerLogSnapshot {
        state.withLock {
            APIServerLogSnapshot(entries: $0.entries, droppedBefore: $0.droppedBefore)
        }
    }

    public func clear() {
        state.withLock {
            $0.entries.removeAll()
            $0.droppedBefore = 0
        }
    }

    private func append(_ payload: APIServerLogEntry.Payload) {
        let schedule = state.withLock { state -> Bool in
            state.entries.append(APIServerLogEntry(id: state.nextID, payload: payload))
            state.nextID &+= 1
            let overflow = state.entries.count - capacity
            if overflow > 0 {
                state.entries.removeFirst(overflow)
                state.droppedBefore += overflow
            }
            guard !state.flushScheduled else { return false }
            state.flushScheduled = true
            return true
        }
        guard schedule else { return }
        let delay = flushDelay
        Task.detached { [self] in
            try? await Task.sleep(for: delay)
            // Cleared before the handler runs, so a row recorded during the
            // drain schedules the next flush instead of being stranded.
            let handler = state.withLock { state -> (@MainActor @Sendable () -> Void)? in
                state.flushScheduled = false
                return state.onFlush
            }
            guard let handler else { return }
            await MainActor.run { handler() }
        }
    }
}

/// Pure formatting of one row, resolved by the list at render time.
public struct APIServerLogEntryPresentation: Equatable, Sendable {
    public let summary: String
    public let detail: String?
    public let severity: AppPresentationSeverity
    /// The `chatcmpl-…` id when the row belongs to a completion.
    public let requestID: String?
    /// `GET /health` and `GET /v1/models`: rows the noise filter hides.
    public let isPollingNoise: Bool

    public init(summary: String,
                detail: String? = nil,
                severity: AppPresentationSeverity = .neutral,
                requestID: String? = nil,
                isPollingNoise: Bool = false) {
        self.summary = summary
        self.detail = detail
        self.severity = severity
        self.requestID = requestID
        self.isPollingNoise = isPollingNoise
    }

    public static func resolve(_ entry: APIServerLogEntry) -> Self {
        switch entry.payload {
        case .server(let event): resolve(event)
        case .bridge(let note): resolve(note)
        }
    }

    private static func resolve(_ event: ServerLogEvent) -> Self {
        let route = "\(event.method) \(event.path)"
        let model = event.requestedModel.map { "model: \($0)" }
        switch event.kind {
        case .requestStarted(_, _, let streaming):
            return Self(summary: "\(route) started\(streaming ? " (stream)" : "")",
                        detail: model, severity: .active, requestID: event.id)
        case .requestCompleted(let duration, let prompt, let cached, let completion, let finish):
            let summary = "\(route) 200 · \(format(duration)) · prompt \(prompt)"
                + (cached > 0 ? " (\(cached) cached)" : "")
                + " · completion \(completion) · \(finish)"
            return Self(summary: summary, detail: model, severity: .success, requestID: event.id)
        case .requestRejected(let status, let code, let message):
            return Self(summary: "\(route) \(status) \(code)",
                        detail: [message, model].compactMap { $0 }.joined(separator: " · "),
                        severity: status >= 500 ? .error : .warning)
        case .routed(let status):
            return Self(summary: "\(route) \(status)",
                        severity: .neutral,
                        isPollingNoise: event.path == "/health" || event.path == "/v1/models")
        case .requestFailed(let status, let streaming, let detail):
            return Self(summary: "\(route) \(status)\(streaming ? " (in-band)" : "") failed",
                        detail: [detail, model].compactMap { $0 }.joined(separator: " · "),
                        severity: status >= 500 ? .error : .warning,
                        requestID: event.id)
        case .streamAborted(let reason):
            return Self(summary: "\(route) stream aborted", detail: reason,
                        severity: .error, requestID: event.id)
        }
    }

    private static func resolve(_ note: APIServerBridgeNote) -> Self {
        switch note.kind {
        case .listening(let host, let port, let modelID):
            return Self(summary: "Listening on http://\(host):\(port)/v1",
                        detail: "model id \(modelID)", severity: .success)
        case .stopped(let reason):
            return Self(summary: "Stopped", detail: reason, severity: .neutral)
        case .startFailed(let message):
            return Self(summary: "Could not start", detail: message, severity: .error)
        case .waitingForSession(let behind):
            let holder = switch behind {
            case .chat: "the chat"
            case .http: "another request"
            }
            return Self(summary: "Request waiting for \(holder) to finish", severity: .active)
        case .substitution(let substitution):
            return Self(summary: "\(substitution.parameter) pinned: \(substitution.requested) → \(substitution.applied)",
                        detail: substitution.reason, severity: .warning)
        case .stopStringMatched:
            return Self(summary: "Stop string matched; generation cancelled", severity: .neutral)
        case .modelNotLoaded:
            return Self(summary: "Completion refused: no model is loaded", severity: .warning)
        case .activeRequestCancelled:
            return Self(summary: "Active request cancelled from the app", severity: .warning)
        case .modelUnloaded:
            return Self(summary: "Model unloaded; completions answer 503 until a model is loaded",
                        severity: .warning)
        case .modelChanged(let name):
            return Self(summary: "Model changed to \(name); the server stops",
                        severity: .warning)
        }
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.1fs", seconds)
    }
}
