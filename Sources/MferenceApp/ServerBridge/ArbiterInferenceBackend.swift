import Foundation
import Synchronization
import Mference
import MferenceAppCore
import MferenceServerCore

/// The result of fitting a translated request to the loaded context window.
public struct APIServerPromptMeasurement: Equatable, Sendable {
    /// The request with only the messages that fit.
    public let request: AppGenerationRequest
    public let removedMessageCount: Int
    public let promptTokenCount: Int

    public init(request: AppGenerationRequest, removedMessageCount: Int, promptTokenCount: Int) {
        self.request = request
        self.removedMessageCount = removedMessageCount
        self.promptTokenCount = promptTokenCount
    }
}

/// Fits a translated request to the context window and counts its prompt
/// tokens. The live implementation uses the install's tokenizer; tests
/// supply a counter.
public typealias APIServerPromptMeasurer =
    @Sendable (AppGenerationRequest) async throws -> APIServerPromptMeasurement

/// `ServerInferenceBackend` over the app's decode session. `prepare` runs
/// entirely app-side with no IPC (it must: it runs outside the coordinator
/// gate, concurrently with another request's generation); `generate` takes a
/// lease from the arbiter, streams through it, and releases it.
public actor ArbiterInferenceBackend: ServerInferenceBackend {
    /// Carried from `prepare` to `generate` in `PreparedGeneration.payload`,
    /// so nothing is re-derived once the response status is committed.
    struct Plan: Sendable {
        let request: AppGenerationRequest
        let session: AppLoadedSession
        let promptTokenCount: Int
        let stopStrings: [String]
        let substitutions: [APIServerSubstitution]
    }

    private let arbiter: AppInferenceArbiter
    private let log: APIServerLogSink
    private let policy: APIServerSamplingPolicy
    private let busyTimeout: Duration
    private let measurePrompt: APIServerPromptMeasurer
    /// Requests inside `generate`: waiting for the lease or decoding.
    private var live: Set<UUID> = []
    private var activeRequest: UUID?
    /// Set by `beginStopping()`; `generate` then refuses every request at
    /// once. `APIServerModel` builds a fresh backend per start, so a new
    /// listener never inherits it.
    private var stopping = false

    public init(arbiter: AppInferenceArbiter,
                log: APIServerLogSink,
                policy: APIServerSamplingPolicy,
                busyTimeout: Duration = .seconds(120),
                measurePrompt: @escaping APIServerPromptMeasurer = {
                    try await ArbiterInferenceBackend.measureWithModelTokenizer($0)
                }) {
        self.arbiter = arbiter
        self.log = log
        self.policy = policy
        self.busyTimeout = busyTimeout
        self.measurePrompt = measurePrompt
    }

    /// Requests waiting for the lease or decoding.
    public var liveRequestCount: Int { live.count }

    /// True while a request holds the decode session.
    public var hasActiveRequest: Bool { activeRequest != nil }

    /// True after `beginStopping()`.
    public var isStopping: Bool { stopping }

    // MARK: ServerInferenceBackend

    public func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration {
        guard let session = await arbiter.loadedSession else {
            log.record(APIServerBridgeNote(kind: .modelNotLoaded))
            throw ServerRequestError.unavailable("no model is loaded")
        }
        // Substitutions are carried in the plan and logged by `run` once the
        // lease is held and the session check has passed: a request refused
        // earlier (400 here, 429 at the busy timeout, 503 for a session that
        // went away) must not leave "pinned" rows that suggest a rewritten
        // request was served.
        let (translated, substitutions) = try ArbiterRequestTranslation.translate(
            request, session: session, policy: policy)
        let measured: APIServerPromptMeasurement
        do {
            measured = try await measurePrompt(translated)
        } catch let error as AppInferenceError {
            throw Self.map(error)
        }
        // The chat UI trims silently; an OpenAI client expects a status code,
        // not a shortened conversation.
        guard measured.removedMessageCount == 0 else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the loaded model's \(session.maxContextTokens)-token context; "
                    + "\(measured.removedMessageCount) message(s) would have to be dropped",
                param: "messages",
                code: "context_length_exceeded")
        }
        let plan = Plan(request: measured.request,
                        session: session,
                        promptTokenCount: measured.promptTokenCount,
                        stopStrings: request.generationConfig.stopStrings,
                        substitutions: substitutions)
        return PreparedGeneration(request: request, payload: plan)
    }

    public func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        guard let plan = prepared.payload as? Plan else {
            throw ServerRequestError.unavailable("the request was not prepared by the in-app server")
        }
        // A request released from the coordinator's queue after the stop
        // drain must not start a fresh generation the shutdown then waits on.
        guard !stopping else { throw Self.serverStopping }
        let id = UUID()
        let owner = AppInferenceOwner.http(id)
        live.insert(id)
        defer {
            live.remove(id)
            if activeRequest == id { activeRequest = nil }
        }

        if let holder = await arbiter.activity.activeOwner {
            log.record(APIServerBridgeNote(kind: .waitingForSession(behind: holder)))
        }
        let ticket: AppInferenceLeaseTicket
        do {
            ticket = try await arbiter.acquire(owner, timeout: busyTimeout)
        } catch {
            throw Self.map(error)
        }
        // `beginStopping()` can land while this request is between the guard
        // above and its place in the arbiter's queue, where
        // `cancelAllRequests()` has nothing to reach. The flag is consistent
        // here (`generate` resumes on this actor) and the lease is held, so
        // this check is the durable one: the ticket goes straight back.
        if stopping {
            await arbiter.release(ticket)
            throw Self.serverStopping
        }
        activeRequest = id
        let outcome: Result<ServerCompletion, any Error>
        do {
            outcome = .success(try await run(plan, owner: owner, onEvent: onEvent))
        } catch {
            outcome = .failure(error)
        }
        await arbiter.release(ticket)
        return try outcome.get()
    }

    // MARK: Control

    /// First step of a stop: from here on `generate` answers 503 without
    /// touching the arbiter, so nothing queued behind the drained requests
    /// can start a generation.
    public func beginStopping() {
        stopping = true
    }

    /// Cancels the request that holds the decode session, if any. Lease
    /// scoped: a chat generation is never touched.
    @discardableResult
    public func cancelActiveRequest() async -> Bool {
        guard let id = activeRequest else { return false }
        await arbiter.cancel(owner: .http(id))
        return true
    }

    /// Cancels every request waiting for or holding the session. Queued
    /// requests are dequeued without touching the wire; the active one gets
    /// the wire cancel and runs to its terminal event.
    public func cancelAllRequests() async {
        for id in live {
            await arbiter.cancel(owner: .http(id))
        }
    }

    // MARK: Generation

    private func run(_ plan: Plan,
                     owner: AppInferenceOwner,
                     onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void)
        async throws -> ServerCompletion {
        // `prepare` ran outside the gate; the model may have been unloaded or
        // reloaded while this request waited.
        guard let session = await arbiter.loadedSession, session == plan.session else {
            throw ServerRequestError.unavailable(
                "the model was unloaded or reloaded while the request waited")
        }
        // Only now is a rewritten request about to be served.
        for substitution in plan.substitutions {
            log.record(APIServerBridgeNote(kind: .substitution(substitution)))
        }
        let filter = StopFilter(stops: plan.stopStrings)
        let arbiter = self.arbiter
        let log = self.log
        let stream = await arbiter.stream(plan.request, owner: owner) { delta in
            let (visible, matchedNow) = filter.push(delta)
            if !visible.isEmpty { onEvent(.content(visible)) }
            if matchedNow {
                log.record(APIServerBridgeNote(kind: .stopStringMatched))
                // Lease-scoped: reaches the wire only because this owner holds it.
                Task { await arbiter.cancel(owner: owner) }
            }
        }

        var terminal: AppInferenceEvent?
        do {
            try await withTaskCancellationHandler {
                for try await event in stream {
                    switch event {
                    case .finished, .cancelled, .failed:
                        terminal = event
                    case .prefillProgress, .token:
                        // Text arrives un-throttled through `onTextDelta`; the
                        // `.token` cadence is the chat UI's and blanks text.
                        break
                    }
                }
            } onCancel: {
                Task { await arbiter.cancel(owner: owner) }
            }
        } catch {
            if terminal == nil { throw Self.map(error) }
        }
        // A client that closed its socket cancels the request task, and the
        // iterator then ends without a terminal event. Name that cause; the
        // decode session did nothing wrong.
        if terminal == nil, Task.isCancelled {
            throw Self.clientDisconnected
        }

        let tail = filter.finish()
        if !tail.isEmpty { onEvent(.content(tail)) }
        let content = filter.content

        switch terminal {
        case .finished(let diagnostics):
            return Self.completion(content: content,
                                   finishReason: diagnostics.stopReason == .maxTokens ? "length" : "stop",
                                   diagnostics: diagnostics,
                                   plan: plan)
        case .cancelled(let diagnostics):
            // Only a stop-string cancel is a success; anything else must not
            // dress a truncated answer up as a completed one.
            if filter.isStopped {
                return Self.completion(content: content, finishReason: "stop",
                                       diagnostics: diagnostics, plan: plan)
            }
            throw Self.cancelledError()
        case .failed(let error, _):
            throw Self.map(error)
        case .prefillProgress, .token, nil:
            throw ServerRequestError.unavailable("the decode session ended without a result")
        }
    }

    private static func completion(content: String,
                                   finishReason: String,
                                   diagnostics: AppDiagnostics,
                                   plan: Plan) -> ServerCompletion {
        // The service's own count is authoritative (measured after its trim);
        // `prepare`'s count is the fallback. Nothing is cached on this path.
        let promptTokens = diagnostics.promptTokenCount ?? plan.promptTokenCount
        let completionTokens = diagnostics.generatedTokens
        return ServerCompletion(
            content: content,
            toolCalls: [],
            finishReason: finishReason,
            usage: OpenAIUsage(promptTokens: promptTokens,
                               completionTokens: completionTokens,
                               totalTokens: promptTokens + completionTokens,
                               cachedTokens: 0))
    }

    // MARK: Error mapping

    /// Every arbiter and app error becomes a `ServerRequestError` with a real
    /// status, except the ones that are genuinely internal (500, with the
    /// detail in the log) and task cancellation.
    static func map(_ error: any Error) -> any Error {
        switch error {
        case let error as ServerRequestError:
            return error
        case let error as AppInferenceArbiterError:
            switch error {
            case .busyTimeout: return ServerRequestError.queueFull
            case .unavailable(let reason): return ServerRequestError.unavailable(reason)
            case .transportLost: return ServerRequestError.unavailable(error.description)
            case .cancelled: return cancelledError()
            }
        case let error as AppInferenceError:
            switch error {
            case .cancelled:
                return cancelledError()
            case .reloadRequired, .modelNotLoaded, .modelLoadFailed, .tokenizerUnavailable,
                 .modelNotFound, .generationInFlight:
                return ServerRequestError.unavailable(error.userMessage)
            case .contextOverflow:
                return ServerRequestError.invalid(message: error.userMessage,
                                                  param: "messages",
                                                  code: "context_length_exceeded")
            case .invalidRequest(let message):
                return ServerRequestError.invalid(message: message, param: nil,
                                                  code: "invalid_request")
            case .unknown:
                return error
            }
        default:
            return error
        }
    }

    /// The request task is cancelled only when its client went away
    /// (`channelInactive`) or the listener is closing after the stop drain.
    /// Either way the wire cancel that follows is a consequence, not the
    /// cause, so the row must not read as a decode-session failure — and the
    /// same disconnect must produce the same row whether the iterator noticed
    /// the cancellation before or after the service's `.cancelled` event.
    static let clientDisconnected =
        ServerRequestError.unavailable("the client disconnected before the answer completed")

    /// Answered from `beginStopping()` on, both before and after admission.
    static let serverStopping = ServerRequestError.unavailable("the server is stopping")

    private static func cancelledError() -> any Error {
        Task.isCancelled
            ? clientDisconnected
            : ServerRequestError.unavailable("the request was cancelled before it completed")
    }

    // MARK: Prompt measurement

    /// Loads the install's tokenizer (cached process-wide after the first
    /// load) and fits the request to the loaded context window.
    public static func measureWithModelTokenizer(
        _ request: AppGenerationRequest
    ) async throws -> APIServerPromptMeasurement {
        let tokenizer: MFTokenizer
        do {
            tokenizer = try await MFTokenizer.load(forModelDirectory: request.modelDirectory)
        } catch {
            throw AppInferenceError.tokenizerUnavailable("\(error)")
        }
        return try measure(request) { messages in
            let rendered = try tokenizer.applyChatTemplate(messages.map { message in
                let role: MFTokenizer.Role = switch message.role {
                case .system: .system
                case .user: .user
                case .assistant: .assistant
                }
                return MFTokenizer.Message(role: role, content: message.content)
            })
            return tokenizer.encode(rendered, addBOS: false).count
        }
    }

    /// The fit itself, over any token counter. Throws `AppInferenceError`.
    public static func measure(
        _ request: AppGenerationRequest,
        tokenCount: ([AppGenerationMessage]) throws -> Int
    ) throws -> APIServerPromptMeasurement {
        let fit = try AppGenerationContextWindow.fit(
            messages: request.messages,
            maximumPromptTokens: request.maxContextTokens,
            maxNewTokens: request.maxNewTokens,
            tokenCount: tokenCount)
        var fitted = request
        fitted.messages = fit.messages
        return APIServerPromptMeasurement(request: fitted,
                                          removedMessageCount: fit.removedMessageCount,
                                          promptTokenCount: fit.promptTokenCount)
    }
}

/// `StreamingStopMatcher` behind a mutex: deltas arrive on the client's
/// reader context while the completion is assembled on the backend.
final class StopFilter: Sendable {
    private struct State {
        var matcher: StreamingStopMatcher
        var content = ""
        var matched = false
    }

    private let state: Mutex<State>

    init(stops: [String]) {
        state = Mutex(State(matcher: StreamingStopMatcher(stops: stops)))
    }

    /// Returns the text safe to emit and whether this push completed a stop
    /// string (reported exactly once).
    func push(_ text: String) -> (visible: String, matchedNow: Bool) {
        state.withLock { state in
            let visible = state.matcher.push(text)
            state.content += visible
            let matchedNow = state.matcher.isStopped && !state.matched
            if matchedNow { state.matched = true }
            return (visible, matchedNow)
        }
    }

    func finish() -> String {
        state.withLock { state in
            let tail = state.matcher.finish()
            state.content += tail
            return tail
        }
    }

    var isStopped: Bool { state.withLock { $0.matcher.isStopped } }

    var content: String { state.withLock { $0.content } }
}
