import Foundation
import MferenceAppCore
import MferenceServerCore

public enum APIServerRunState: Equatable, Sendable {
    case stopped
    case starting
    case listening(host: String, port: Int)
    case stopping
    /// The last start attempt failed; the message is already plain English.
    case failed(String)

    public var isListening: Bool {
        if case .listening = self { return true }
        return false
    }

    /// True while a start or stop is in flight; the button is disabled then.
    public var isTransitioning: Bool {
        switch self {
        case .starting, .stopping: true
        case .stopped, .listening, .failed: false
        }
    }
}

public enum APIServerAction: Equatable, Sendable {
    case start
    case stop
    case cancelActiveRequest
}

/// Everything the pane's status strip needs, captured as a value so the
/// label logic is testable without SwiftUI.
public struct APIServerSnapshot: Equatable, Sendable {
    public var runState: APIServerRunState
    public var bindMode: ServerBindMode
    public var hasLoadedSession: Bool
    public var activeOwner: AppInferenceOwner?
    public var queuedHTTPCount: Int
    public var queuedChat: Bool

    public init(runState: APIServerRunState,
                bindMode: ServerBindMode = .loopback,
                hasLoadedSession: Bool,
                activeOwner: AppInferenceOwner? = nil,
                queuedHTTPCount: Int = 0,
                queuedChat: Bool = false) {
        self.runState = runState
        self.bindMode = bindMode
        self.hasLoadedSession = hasLoadedSession
        self.activeOwner = activeOwner
        self.queuedHTTPCount = queuedHTTPCount
        self.queuedChat = queuedChat
    }
}

public struct APIServerPresentationState: Equatable, Sendable {
    public var label: String
    public var detail: String?
    public var severity: AppPresentationSeverity
    public var showsActivity: Bool
    public var primaryAction: APIServerAction?
    public var secondaryAction: APIServerAction?

    public init(label: String,
                detail: String? = nil,
                severity: AppPresentationSeverity = .neutral,
                showsActivity: Bool = false,
                primaryAction: APIServerAction? = nil,
                secondaryAction: APIServerAction? = nil) {
        self.label = label
        self.detail = detail
        self.severity = severity
        self.showsActivity = showsActivity
        self.primaryAction = primaryAction
        self.secondaryAction = secondaryAction
    }

    public static func resolve(_ snapshot: APIServerSnapshot) -> Self {
        switch snapshot.runState {
        case .starting:
            return Self(label: "Starting", severity: .active, showsActivity: true)
        case .stopping:
            return Self(label: "Stopping", severity: .active, showsActivity: true)
        case .failed(let message):
            return Self(label: "Could not start", detail: message, severity: .error,
                        primaryAction: snapshot.hasLoadedSession ? .start : nil)
        case .stopped:
            if snapshot.hasLoadedSession {
                return Self(label: "Stopped", primaryAction: .start)
            }
            return Self(label: "Stopped", detail: "Load a model to serve it.")
        case .listening(let host, let port):
            let endpoint = "http://\(host):\(port)/v1"
            guard snapshot.hasLoadedSession else {
                return Self(label: "Listening · no model loaded",
                            detail: "\(endpoint) answers 503 until a model is loaded.",
                            severity: .warning, primaryAction: .stop)
            }
            let queued = snapshot.queuedHTTPCount > 0
                ? " · \(snapshot.queuedHTTPCount) waiting" : ""
            switch snapshot.activeOwner {
            case .http:
                return Self(label: "Serving a request\(queued)", detail: endpoint,
                            severity: .active, showsActivity: true,
                            primaryAction: .stop, secondaryAction: .cancelActiveRequest)
            case .chat:
                return Self(label: "Listening · chat is generating\(queued)",
                            detail: endpoint, severity: .active, showsActivity: true,
                            primaryAction: .stop)
            case nil:
                return Self(label: "Listening on \(host):\(port)", detail: endpoint,
                            severity: .success, primaryAction: .stop)
            }
        }
    }
}
