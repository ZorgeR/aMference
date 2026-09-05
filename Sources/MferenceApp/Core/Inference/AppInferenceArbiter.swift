import Foundation

/// Who is using, or waiting for, the app's single decode session.
public enum AppInferenceOwner: Hashable, Sendable {
    case chat
    case http(UUID)
}

/// What the loaded session can serve: the load key plus the sampling the app
/// loaded it for. Published by `AppModel` after every successful load and
/// cleared on unload.
public struct AppLoadedSession: Equatable, Sendable {
    public let modelDirectory: URL
    public let maxContextTokens: Int
    public let runtimeOptions: AppRuntimeOptions
    /// The loaded session's sampling mode (`temperature != 0`).
    public let forceLogitsHead: Bool
    public let temperature: Float
    public let topK: Int?
    public let topP: Float?

    public init(modelDirectory: URL,
                maxContextTokens: Int,
                runtimeOptions: AppRuntimeOptions,
                forceLogitsHead: Bool,
                temperature: Float,
                topK: Int?,
                topP: Float?) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }
}

/// Display-only view of the arbiter's state.
public struct AppInferenceActivity: Equatable, Sendable {
    public let activeOwner: AppInferenceOwner?
    public let queuedChat: Bool
    public let queuedHTTPCount: Int
    public let loadedSession: AppLoadedSession?

    public init(activeOwner: AppInferenceOwner?,
                queuedChat: Bool,
                queuedHTTPCount: Int,
                loadedSession: AppLoadedSession?) {
        self.activeOwner = activeOwner
        self.queuedChat = queuedChat
        self.queuedHTTPCount = queuedHTTPCount
        self.loadedSession = loadedSession
    }

    public static let idle = AppInferenceActivity(
        activeOwner: nil, queuedChat: false, queuedHTTPCount: 0, loadedSession: nil)
}

public enum AppInferenceArbiterError: Error, Equatable, Sendable, CustomStringConvertible {
    /// `acquire` waited for its whole timeout behind another owner.
    case busyTimeout
    /// Admission is closed (a load or unload is running); the string is the
    /// reason, suitable for display.
    case unavailable(String)
    /// The waiting owner was cancelled before it was admitted.
    case cancelled
    /// The watchdog shut the transport down under the active generation.
    /// The loaded session is gone with it: the arbiter publishes
    /// `loadedSession == nil`, and the owner of the model must reload.
    case transportLost

    /// What the chat UI and the load state show after the watchdog fires.
    public static let transportLostMessage =
        "The decode service stopped responding and was shut down; reload the model."

    public var description: String {
        switch self {
        case .busyTimeout: return "The decode session stayed busy for too long."
        case .unavailable(let reason): return reason
        case .cancelled: return "Generation cancelled."
        case .transportLost: return Self.transportLostMessage
        }
    }

    /// The `AppInferenceError` the chat UI shows for this failure.
    public var inferenceError: AppInferenceError {
        switch self {
        case .cancelled: return .cancelled
        case .busyTimeout, .unavailable, .transportLost: return .unknown(description)
        }
    }
}

/// Sole owner of the app's `AppInferenceClient`. The decode pipe is
/// one-generation-at-a-time by transport construction (two readers corrupt
/// the frame stream), so every consumer — the chat UI today, HTTP requests
/// later — goes through a lease handed out here.
///
/// Admission: FIFO within a class, `.chat` ahead of any queued `.http`, and
/// never preempting a lease that is already decoding. Cancellation is
/// lease-scoped: a queued owner is dequeued without touching the wire; only
/// the active owner's cancel reaches `client.cancel()`. Load and unload go
/// through `withExclusiveSession`, which drains the active generation to its
/// terminal event before the lifecycle call runs on the same handle.
///
/// An optional watchdog fails the active generation when it goes silent for
/// `watchdog` *after its first event*: `DecodeFrameCodec.read` blocks with no
/// deadline and Swift cancellation cannot interrupt it, so the transport is
/// shut down instead and the lease fails with `.transportLost`. The clock
/// never starts at registration: the service sends one progress frame per
/// prefill chunk, and the first one can follow minutes of first-touch expert
/// hashing on a large install.
public actor AppInferenceArbiter {
    private enum LeaseSource {
        case held(AppInferenceLeaseTicket)
        case queued(AppInferenceLeaseWaiter)
    }

    private let client: any AppInferenceClient
    /// `nil` disables the watchdog.
    private let watchdogDuration: Duration?
    private let broadcaster: AppInferenceActivityBroadcaster
    private var active: AppInferenceActiveLease?
    private var waiters = AppInferenceWaiterQueue()
    private var forwardingTasks: [UUID: Task<Void, Never>] = [:]
    /// The reason admission is closed while an exclusive session runs.
    private var exclusiveReason: String?
    private var exclusiveWaiters: [CheckedContinuation<Void, Never>] = []
    private var loadedSessionValue: AppLoadedSession?

    /// Exactly one arbiter may exist per client: two arbiters over one
    /// `DecodeServiceInferenceClient` are two readers on the same pipe. The
    /// composition root creates the arbiter once and hands the same instance
    /// to every consumer (`AppModel`, `APIServerModel`).
    ///
    /// - Parameter watchdog: silence budget between events of one
    ///   generation, measured from its first event; `nil` (the default)
    ///   disables the watchdog.
    public init(client: any AppInferenceClient, watchdog: Duration? = nil) {
        self.client = client
        self.watchdogDuration = watchdog
        self.broadcaster = AppInferenceActivityBroadcaster(initial: .idle)
    }

    deinit {
        broadcaster.finishAll()
    }

    // MARK: State

    public var loadedSession: AppLoadedSession? { loadedSessionValue }

    public var activity: AppInferenceActivity { currentActivity }

    /// Each call returns an independent stream. Its first element is the
    /// current activity; every later element is a change.
    public nonisolated var activityStream: AsyncStream<AppInferenceActivity> {
        broadcaster.subscribe()
    }

    public func publishLoadedSession(_ session: AppLoadedSession?) {
        loadedSessionValue = session
        publishActivity()
    }

    // MARK: Leases

    /// Waits for the session. FIFO within the owner's class, `.chat` ahead of
    /// queued `.http`. A `timeout` that elapses first throws `.busyTimeout`;
    /// cancelling the waiting task throws `.cancelled`; a running exclusive
    /// session throws `.unavailable` immediately.
    public func acquire(_ owner: AppInferenceOwner, timeout: Duration?) async throws
        -> AppInferenceLeaseTicket {
        let waiter = try enqueue(owner: owner, releasesOnGenerationEnd: false,
                                 generationID: nil)
        if let timeout, waiters.contains(id: waiter.id) {
            let id = waiter.id
            let timer = Task { [weak self] in
                try? await Task.sleep(for: timeout)
                await self?.withdraw(waiterID: id, error: .busyTimeout)
            }
            if !waiters.setTimeout(timer, for: id) { timer.cancel() }
        }
        let future = waiter.future
        let id = waiter.id
        return try await withTaskCancellationHandler {
            try await future.value
        } onCancel: {
            Task { [weak self] in await self?.withdraw(waiterID: id, error: .cancelled) }
        }
    }

    /// Returns the session. If a generation is still in flight the release
    /// takes effect when it ends. A revoked or unknown ticket is ignored.
    public func release(_ ticket: AppInferenceLeaseTicket) {
        guard let lease = active, lease.ticket == ticket else { return }
        if lease.generation != nil {
            active?.releaseRequested = true
            return
        }
        endActiveLease()
    }

    /// Runs one generation for `owner`. An owner that already holds a ticket
    /// uses it; otherwise a lease is acquired first (with no timeout) and
    /// released when the stream terminates. The stream fails with an
    /// `AppInferenceArbiterError` for arbiter-side outcomes (dequeued by
    /// `cancel`, admission closed, `.transportLost` from the watchdog) and
    /// with whatever the client threw otherwise. `onTextDelta` receives every text fragment
    /// un-throttled when the client supports `AppInferenceDeltaStreaming`.
    public func stream(_ request: AppGenerationRequest,
                       owner: AppInferenceOwner,
                       onTextDelta: @escaping @Sendable (String) -> Void = { _ in })
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream<AppInferenceEvent, Error>.makeStream()
        let generationID = UUID()
        let source: LeaseSource
        if let lease = active, lease.ticket.owner == owner {
            if let reason = exclusiveReason {
                continuation.finish(throwing: AppInferenceArbiterError.unavailable(reason))
                return stream
            }
            guard lease.generation == nil, lease.pendingGenerationID == nil else {
                continuation.finish(throwing: AppInferenceArbiterError.unavailable(
                    "A generation is already in flight for this owner."))
                return stream
            }
            active?.pendingGenerationID = generationID
            source = .held(lease.ticket)
        } else {
            do {
                source = .queued(try enqueue(owner: owner, releasesOnGenerationEnd: true,
                                             generationID: generationID))
            } catch {
                continuation.finish(throwing: error)
                return stream
            }
        }
        continuation.onTermination = { [weak self] termination in
            guard case .cancelled = termination else { return }
            Task { await self?.consumerLeft(generationID: generationID) }
        }
        forwardingTasks[generationID] = Task { [weak self] in
            await self?.runGeneration(generationID, request: request, source: source,
                                      continuation: continuation, onTextDelta: onTextDelta)
        }
        return stream
    }

    /// Lease-scoped cancel. A queued owner is dequeued and its `acquire` or
    /// stream fails with `.cancelled`; the wire is never touched. The active
    /// owner's in-flight generation gets `client.cancel()`; an active owner
    /// whose generation has not started yet ends it before it touches the
    /// client. Any other owner is a no-op.
    public func cancel(owner: AppInferenceOwner) {
        if let waiter = waiters.remove(owner: owner) {
            waiter.timeout?.cancel()
            waiter.future.resolve(.failure(AppInferenceArbiterError.cancelled))
            publishActivity()
            return
        }
        guard let lease = active, lease.ticket.owner == owner else { return }
        if lease.generation != nil {
            client.cancel()
        } else {
            active?.cancelRequested = true
        }
    }

    // MARK: Exclusive sessions

    /// Load/unload barrier: stop admitting → cancel the active generation →
    /// await its terminal event → fail queued waiters with `.unavailable` →
    /// run `body` → resume admitting. Sessions serialise among themselves.
    /// Whatever ticket was active is revoked; `body` runs with nobody on the
    /// client. Rethrows `body`'s error and honours task cancellation while
    /// waiting for the barrier.
    public func withExclusiveSession<T: Sendable>(
        reason: String,
        _ body: @Sendable () async throws -> T
    ) async throws -> T {
        while exclusiveReason != nil {
            await withCheckedContinuation { exclusiveWaiters.append($0) }
        }
        exclusiveReason = reason
        defer { endExclusiveSession() }
        publishActivity()

        if let lease = active {
            if let generation = lease.generation {
                client.cancel()
                await generation.completion.wait()
            }
            if active?.ticket == lease.ticket {
                active = nil
            }
        }
        for waiter in waiters.removeAll() {
            waiter.timeout?.cancel()
            waiter.future.resolve(.failure(AppInferenceArbiterError.unavailable(reason)))
        }
        publishActivity()
        try Task.checkCancellation()
        return try await body()
    }

    private func endExclusiveSession() {
        exclusiveReason = nil
        admitNext()
        publishActivity()
        if !exclusiveWaiters.isEmpty {
            exclusiveWaiters.removeFirst().resume()
        }
    }

    // MARK: Admission

    private func enqueue(owner: AppInferenceOwner, releasesOnGenerationEnd: Bool,
                         generationID: UUID?) throws -> AppInferenceLeaseWaiter {
        if let exclusiveReason {
            throw AppInferenceArbiterError.unavailable(exclusiveReason)
        }
        if active?.ticket.owner == owner || waiters.contains(owner: owner) {
            throw AppInferenceArbiterError.unavailable(
                "This owner already holds or awaits the decode session.")
        }
        let waiter = AppInferenceLeaseWaiter(
            id: UUID(), owner: owner,
            releasesOnGenerationEnd: releasesOnGenerationEnd,
            generationID: generationID,
            future: AppInferenceLeaseFuture())
        waiters.append(waiter)
        admitNext()
        publishActivity()
        return waiter
    }

    private func admitNext() {
        guard active == nil, exclusiveReason == nil,
              let waiter = waiters.popNext() else { return }
        waiter.timeout?.cancel()
        let ticket = AppInferenceLeaseTicket(owner: waiter.owner, id: waiter.id)
        active = AppInferenceActiveLease(
            ticket: ticket,
            releasesOnGenerationEnd: waiter.releasesOnGenerationEnd,
            pendingGenerationID: waiter.generationID)
        waiter.future.resolve(.success(ticket))
    }

    private func withdraw(waiterID: UUID, error: AppInferenceArbiterError) {
        guard let waiter = waiters.remove(id: waiterID) else { return }
        waiter.timeout?.cancel()
        waiter.future.resolve(.failure(error))
        publishActivity()
    }

    private func endActiveLease() {
        active = nil
        admitNext()
        publishActivity()
    }

    // MARK: Generations

    private func runGeneration(
        _ generationID: UUID,
        request: AppGenerationRequest,
        source: LeaseSource,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
        onTextDelta: @escaping @Sendable (String) -> Void
    ) async {
        let ticket: AppInferenceLeaseTicket
        switch source {
        case .held(let held):
            ticket = held
        case .queued(let waiter):
            do {
                ticket = try await waiter.future.value
            } catch {
                forwardingTasks[generationID] = nil
                continuation.finish(throwing: error)
                return
            }
        }
        guard let lease = active, lease.ticket == ticket,
              lease.pendingGenerationID == generationID else {
            forwardingTasks[generationID] = nil
            continuation.finish(throwing: AppInferenceArbiterError.unavailable(
                "The decode session lease was revoked before the generation started."))
            return
        }

        let record = AppInferenceGenerationRecord(id: generationID, continuation: continuation)
        active?.pendingGenerationID = nil
        active?.generation = record
        if lease.cancelRequested {
            // Cancelled between admission and start: the client is never touched.
            finishGeneration(generationID, ticket: ticket,
                             outcome: .failure(AppInferenceArbiterError.cancelled))
            return
        }

        let events = makeClientStream(request, onTextDelta: onTextDelta)
        do {
            for try await event in events {
                // After the watchdog concluded this generation the loop only
                // drains whatever the client still delivers.
                guard active?.generation?.id == generationID else { continue }
                active?.generation?.lastEvent = .now
                startWatchdogIfNeeded(generationID: generationID, ticket: ticket)
                continuation.yield(event)
            }
            finishGeneration(generationID, ticket: ticket, outcome: .success(()))
        } catch {
            finishGeneration(generationID, ticket: ticket, outcome: .failure(error))
        }
    }

    /// The idle clock starts with the first event of a generation, never at
    /// registration (see the type comment).
    private func startWatchdogIfNeeded(generationID: UUID, ticket: AppInferenceLeaseTicket) {
        guard watchdogDuration != nil, active?.generation?.id == generationID,
              active?.generation?.watchdog == nil else { return }
        active?.generation?.watchdog = Task { [weak self] in
            await self?.runWatchdog(generationID: generationID, ticket: ticket)
        }
    }

    private func makeClientStream(
        _ request: AppGenerationRequest,
        onTextDelta: @escaping @Sendable (String) -> Void
    ) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        if let streaming = client as? any AppInferenceDeltaStreaming {
            return streaming.generate(request, onTextDelta: onTextDelta)
        }
        return client.generate(request)
    }

    /// Idempotent: the watchdog and the forwarding loop can both arrive here.
    private func finishGeneration(_ generationID: UUID,
                                  ticket: AppInferenceLeaseTicket,
                                  outcome: Result<Void, any Error>) {
        forwardingTasks[generationID] = nil
        guard var lease = active, lease.ticket == ticket,
              let record = lease.generation, record.id == generationID else { return }
        record.watchdog?.cancel()
        lease.generation = nil
        lease.cancelRequested = false
        active = lease
        switch outcome {
        case .success: record.continuation.finish()
        case .failure(let error): record.continuation.finish(throwing: error)
        }
        record.completion.finish()
        if lease.releasesOnGenerationEnd || lease.releaseRequested {
            endActiveLease()
        } else {
            publishActivity()
        }
    }

    /// The consumer of a `stream()` stopped listening before the terminal
    /// event. Queued: withdrawn without touching the wire. Not yet started:
    /// ends before touching the client. In flight: `client.cancel()` so the
    /// service stops promptly; the forwarding loop still drains to the
    /// terminal event so the pipe stays consistent.
    private func consumerLeft(generationID: UUID) {
        if let waiter = waiters.remove(generationID: generationID) {
            waiter.timeout?.cancel()
            waiter.future.resolve(.failure(AppInferenceArbiterError.cancelled))
            publishActivity()
            return
        }
        guard let lease = active else { return }
        if lease.generation?.id == generationID {
            client.cancel()
        } else if lease.pendingGenerationID == generationID {
            active?.cancelRequested = true
        }
    }

    // MARK: Watchdog

    private func runWatchdog(generationID: UUID, ticket: AppInferenceLeaseTicket) async {
        guard let watchdogDuration else { return }
        while true {
            guard let lease = active, lease.ticket == ticket,
                  let record = lease.generation, record.id == generationID,
                  let lastEvent = record.lastEvent else { return }
            let idle = ContinuousClock.now - lastEvent
            if idle >= watchdogDuration {
                fireWatchdog(generationID: generationID, ticket: ticket)
                return
            }
            do {
                try await Task.sleep(for: watchdogDuration - idle)
            } catch {
                return
            }
        }
    }

    /// Cancel on the wire, fail the lease with `.transportLost`, drop the
    /// published session, then tear the transport down off the actor:
    /// `DecodeServiceInferenceClient.shutdown()` polls the child for up to
    /// ~750 ms and must never stall admission, cancel, or activity
    /// publication while it does.
    private func fireWatchdog(generationID: UUID, ticket: AppInferenceLeaseTicket) {
        guard let lease = active, lease.ticket == ticket,
              lease.generation?.id == generationID else { return }
        client.cancel()
        forwardingTasks[generationID]?.cancel()
        finishGeneration(generationID, ticket: ticket,
                         outcome: .failure(AppInferenceArbiterError.transportLost))
        loadedSessionValue = nil
        publishActivity()
        if let transport = client as? any AppInferenceTransportControlling {
            Task.detached { transport.shutdown() }
        }
    }

    // MARK: Activity

    private var currentActivity: AppInferenceActivity {
        AppInferenceActivity(activeOwner: active?.ticket.owner,
                             queuedChat: waiters.queuedChat,
                             queuedHTTPCount: waiters.queuedHTTPCount,
                             loadedSession: loadedSessionValue)
    }

    private func publishActivity() {
        broadcaster.publish(currentActivity)
    }
}
