import Foundation
import Observation
import NIOCore
import MferenceAppCore
import MferenceServerCore

/// The three things `APIServerModel` needs from outside the app process:
/// the loaded install's API identity, a prompt measurer over its tokenizer,
/// and the bind address. Tests supply model-free versions.
public struct APIServerEnvironment: Sendable {
    public var resolveIdentity: @Sendable (URL) async throws -> APIServerModelIdentity
    public var measurePrompt: APIServerPromptMeasurer
    public var resolveHost: @Sendable (ServerBindMode) async throws -> String

    public init(resolveIdentity: @escaping @Sendable (URL) async throws -> APIServerModelIdentity,
                measurePrompt: @escaping APIServerPromptMeasurer,
                resolveHost: @escaping @Sendable (ServerBindMode) async throws -> String) {
        self.resolveIdentity = resolveIdentity
        self.measurePrompt = measurePrompt
        self.resolveHost = resolveHost
    }

    /// Manifest and tokenizer reads, and the `tailscale ip -4` spawn for
    /// `.tailnet`, all run off the MainActor.
    public static let live = APIServerEnvironment(
        resolveIdentity: { directory in
            try await Task.detached {
                try await APIServerModelIdentity.resolve(modelDirectory: directory)
            }.value
        },
        measurePrompt: { request in
            try await ArbiterInferenceBackend.measureWithModelTokenizer(request)
        },
        resolveHost: { mode in
            try await Task.detached { try mode.host() }.value
        })
}

/// State of the in-app OpenAI-compatible server. Owns the listener's
/// lifecycle, the log ring, and the pane's settings. Every start builds a
/// fresh `MferenceHTTPServer`: the server is single-use (its shutdown
/// retires the event-loop group it created), so start and stop are
/// serialised behind one task chain and never reuse an instance.
@MainActor
@Observable
public final class APIServerModel {
    public static let defaultPort = 8080
    public static let defaultQueueLimit = 4
    public static let defaultBusyTimeout: Duration = .seconds(120)
    public static let portRange = 1...65_535

    public private(set) var runState: APIServerRunState = .stopped
    /// `0` asks the system for a free port; the bound port is read back from
    /// the channel and published in `runState`.
    public var port: Int = APIServerModel.defaultPort
    public var bindMode: ServerBindMode = .loopback
    public var queueLimit: Int = APIServerModel.defaultQueueLimit
    public var samplingPolicy: APIServerSamplingPolicy = .rejectOnMismatch
    /// How long a request waits for the decode session before `429`.
    public var busyTimeout: Duration = APIServerModel.defaultBusyTimeout
    public private(set) var logEntries: [APIServerLogEntry] = []
    public private(set) var droppedBefore = 0
    public private(set) var inferenceActivity: AppInferenceActivity = .idle
    /// The identity frozen into the running server; nil while stopped.
    public private(set) var servedIdentity: APIServerModelIdentity?
    public private(set) var servedModelDirectory: URL?
    /// Successful starts so far; each one was a fresh server instance.
    public private(set) var startCount = 0

    @ObservationIgnored public let logSink: APIServerLogSink
    @ObservationIgnored private let arbiter: AppInferenceArbiter
    @ObservationIgnored private let environment: APIServerEnvironment
    @ObservationIgnored private var server: MferenceHTTPServer?
    @ObservationIgnored private var backend: ArbiterInferenceBackend?
    @ObservationIgnored private var lifecycle: Task<Void, Never>?
    @ObservationIgnored private var activityTask: Task<Void, Never>?

    /// `arbiter` must be the app's one arbiter (`AppModel.arbiter`): exactly
    /// one `AppInferenceArbiter` may exist per `DecodeServiceInferenceClient`,
    /// because the decode pipe tolerates a single reader. The composition
    /// root passes the same instance here and to `AppModel`; building a
    /// second arbiter, or a second client, for the server puts two readers
    /// on the pipe.
    public init(arbiter: AppInferenceArbiter,
                environment: APIServerEnvironment = .live,
                logCapacity: Int = APIServerLogSink.defaultCapacity,
                logFlushDelay: Duration = .milliseconds(100)) {
        self.arbiter = arbiter
        self.environment = environment
        self.logSink = APIServerLogSink(capacity: logCapacity, flushDelay: logFlushDelay)
        logSink.setFlushHandler { [weak self] in self?.drainLog() }
        let activities = arbiter.activityStream
        activityTask = Task { [weak self] in
            for await activity in activities {
                guard let self else { return }
                self.apply(activity)
            }
        }
    }

    deinit {
        activityTask?.cancel()
        lifecycle?.cancel()
    }

    // MARK: Derived state

    public var isListening: Bool { runState.isListening }

    public var listeningEndpoint: (host: String, port: Int)? {
        if case .listening(let host, let port) = runState { return (host, port) }
        return nil
    }

    /// `http://host:port/v1`, the value clients paste as their base URL.
    public var baseURL: String? {
        listeningEndpoint.map { "http://\($0.host):\($0.port)/v1" }
    }

    public var snapshot: APIServerSnapshot {
        APIServerSnapshot(runState: runState,
                          bindMode: bindMode,
                          hasLoadedSession: inferenceActivity.loadedSession != nil,
                          activeOwner: inferenceActivity.activeOwner,
                          queuedHTTPCount: inferenceActivity.queuedHTTPCount,
                          queuedChat: inferenceActivity.queuedChat)
    }

    public var presentation: APIServerPresentationState {
        .resolve(snapshot)
    }

    public var canStart: Bool {
        switch runState {
        case .stopped, .failed: inferenceActivity.loadedSession != nil
        case .starting, .listening, .stopping: false
        }
    }

    public var canStop: Bool { runState.isListening }

    public var canCancelActiveRequest: Bool {
        guard runState.isListening, case .http = inferenceActivity.activeOwner else { return false }
        return true
    }

    // MARK: Commands

    public func start() {
        enqueue { await self.performStart() }
    }

    public func stop() {
        enqueue { await self.performStop(reason: "stopped from the app") }
    }

    /// Cancels the HTTP request holding the decode session. Lease scoped: a
    /// chat generation is never touched. Logged once the backend confirms a
    /// request was actually cancelled, so the remote client's truncated
    /// answer has a visible cause and a click that lands after the request
    /// finished leaves no false row.
    public func cancelActiveRequest() {
        guard let backend else { return }
        let logSink = logSink
        Task {
            if await backend.cancelActiveRequest() {
                logSink.record(APIServerBridgeNote(kind: .activeRequestCancelled))
            }
        }
    }

    public func clearLog() {
        logSink.clear()
        drainLog()
    }

    /// Waits for every queued start and stop to finish. A test seam; the UI
    /// observes `runState` instead.
    public func awaitIdle() async {
        while let task = lifecycle {
            await task.value
            if lifecycle == task { lifecycle = nil }
        }
    }

    // MARK: Lifecycle

    private func enqueue(_ operation: @escaping @MainActor () async -> Void) {
        let previous = lifecycle
        lifecycle = Task { @MainActor in
            await previous?.value
            await operation()
        }
    }

    private func performStart() async {
        switch runState {
        case .stopped, .failed: break
        case .starting, .listening, .stopping: return
        }
        guard let session = await arbiter.loadedSession else {
            fail("Load a model before starting the server.")
            return
        }
        runState = .starting
        let identity: APIServerModelIdentity
        let host: String
        do {
            identity = try await environment.resolveIdentity(session.modelDirectory)
            host = try await environment.resolveHost(bindMode)
        } catch {
            fail(Self.describe(error))
            return
        }
        let backend = ArbiterInferenceBackend(arbiter: arbiter,
                                              log: logSink,
                                              policy: samplingPolicy,
                                              busyTimeout: busyTimeout,
                                              measurePrompt: environment.measurePrompt)
        let server = MferenceHTTPServer(modelID: identity.modelID,
                                        queueLimit: max(1, queueLimit),
                                        backend: backend,
                                        chatDialect: identity.chatDialect,
                                        log: logSink)
        let requestedPort = port
        let boundPort: Int
        do {
            let channel = try await server.start(host: host, port: requestedPort)
            boundPort = channel.localAddress?.port ?? requestedPort
        } catch {
            // Retires the event-loop group the failed instance created.
            try? await server.shutdown()
            fail(Self.bindFailureMessage(error, host: host, port: requestedPort))
            return
        }
        self.server = server
        self.backend = backend
        servedIdentity = identity
        servedModelDirectory = session.modelDirectory
        startCount += 1
        runState = .listening(host: host, port: boundPort)
        logSink.record(APIServerBridgeNote(
            kind: .listening(host: host, port: boundPort, modelID: identity.modelID)))
        // A model switch that landed while the listener was coming up was
        // ignored by `apply`; check once more now that it can act.
        if let current = await arbiter.loadedSession?.modelDirectory,
           current != session.modelDirectory {
            stopForModelChange(current)
        }
    }

    /// Closes the backend to new generations, drains the bridge's leases,
    /// then shuts the listener down. The order is load-bearing: `shutdown()`
    /// awaits every in-flight request task, and those sit on the decode
    /// stream until the wire cancel is answered; a request the coordinator
    /// releases after the drain must find the backend already stopping so
    /// the shutdown never waits on a generation that started during it.
    private func performStop(reason: String) async {
        guard case .listening = runState, let server else { return }
        runState = .stopping
        if let backend {
            await backend.beginStopping()
            await backend.cancelAllRequests()
            // Requests the coordinator releases from its queue now answer
            // 503 at once; wait for that as well, so `shutdown()` finds no
            // request task between the queue and the backend.
            let deadline = ContinuousClock.now + .seconds(5)
            while ContinuousClock.now < deadline {
                let live = await backend.liveRequestCount
                let queued = await server.queuedRequestCount
                if live == 0, queued == 0 { break }
                try? await Task.sleep(for: .milliseconds(10))
            }
        }
        var reason = reason
        do {
            try await server.shutdown()
        } catch {
            reason += " (shutdown reported: \(Self.describe(error)))"
        }
        self.server = nil
        self.backend = nil
        servedIdentity = nil
        servedModelDirectory = nil
        runState = .stopped
        logSink.record(APIServerBridgeNote(kind: .stopped(reason: reason)))
    }

    private func fail(_ message: String) {
        runState = .failed(message)
        logSink.record(APIServerBridgeNote(kind: .startFailed(message)))
    }

    private func stopForModelChange(_ directory: URL) {
        let name = directory.lastPathComponent
        logSink.record(APIServerBridgeNote(kind: .modelChanged(name)))
        enqueue { await self.performStop(reason: "the model changed to \(name)") }
    }

    // MARK: Activity

    /// The model id and dialect are frozen into the running server, so a
    /// different model directory stops it; an unload keeps it listening
    /// (`/health` 200, completions 503) and a reload of the same directory
    /// changes nothing.
    private func apply(_ activity: AppInferenceActivity) {
        let previous = inferenceActivity
        inferenceActivity = activity
        guard case .listening = runState, let served = servedModelDirectory else { return }
        if let session = activity.loadedSession {
            if session.modelDirectory != served {
                stopForModelChange(session.modelDirectory)
            }
        } else if previous.loadedSession != nil {
            logSink.record(APIServerBridgeNote(kind: .modelUnloaded))
        }
    }

    private func drainLog() {
        let snapshot = logSink.snapshot
        logEntries = snapshot.entries
        droppedBefore = snapshot.droppedBefore
    }

    // MARK: Error rendering

    static func bindFailureMessage(_ error: any Error, host: String, port: Int) -> String {
        if let error = error as? IOError {
            switch error.errnoCode {
            case EADDRINUSE:
                return "Port \(port) is already in use on \(host). Choose another port, or stop the program using it."
            case EACCES:
                return "Port \(port) needs administrator privileges. Choose a port above 1023."
            case EADDRNOTAVAIL:
                return "\(host) is not an address of this Mac."
            default:
                return "Could not listen on \(host):\(port): \(error)"
            }
        }
        return "Could not listen on \(host):\(port): \(describe(error))"
    }

    /// `ServerArgumentError`, `IOError`, and `ModelError` all describe
    /// themselves; interpolation uses that description verbatim.
    static func describe(_ error: any Error) -> String {
        "\(error)"
    }
}
