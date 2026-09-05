import Foundation
import Synchronization

/// Proof that an owner holds the decode session. The `id` is opaque; a
/// ticket the arbiter has revoked (see `withExclusiveSession`) is ignored by
/// every later call that presents it.
public struct AppInferenceLeaseTicket: Hashable, Sendable {
    public let owner: AppInferenceOwner
    let id: UUID
}

// MARK: - One-shot rendezvous types

/// Resolves exactly once with the granted ticket or the reason the wait
/// ended. Registered synchronously on the arbiter, so a `cancel(owner:)`
/// that lands before the waiting task has even started still finds it.
final class AppInferenceLeaseFuture: Sendable {
    private struct State {
        var result: Result<AppInferenceLeaseTicket, any Error>?
        var continuation: CheckedContinuation<AppInferenceLeaseTicket, any Error>?
    }

    private let state = Mutex(State())

    func resolve(_ result: Result<AppInferenceLeaseTicket, any Error>) {
        let continuation = state.withLock { state
            -> CheckedContinuation<AppInferenceLeaseTicket, any Error>? in
            guard state.result == nil else { return nil }
            state.result = result
            defer { state.continuation = nil }
            return state.continuation
        }
        continuation?.resume(with: result)
    }

    var value: AppInferenceLeaseTicket {
        get async throws {
            try await withCheckedThrowingContinuation { continuation in
                let ready = state.withLock { state
                    -> Result<AppInferenceLeaseTicket, any Error>? in
                    if let result = state.result { return result }
                    state.continuation = continuation
                    return nil
                }
                if let ready { continuation.resume(with: ready) }
            }
        }
    }
}

/// A once-only "done" signal; `withExclusiveSession` waits on it for the
/// active generation's terminal event.
final class AppInferenceCompletion: Sendable {
    private struct State {
        var isDone = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    private let state = Mutex(State())

    var isDone: Bool { state.withLock { $0.isDone } }

    func wait() async {
        await withCheckedContinuation { continuation in
            let resumeNow = state.withLock { state -> Bool in
                if state.isDone { return true }
                state.waiters.append(continuation)
                return false
            }
            if resumeNow { continuation.resume() }
        }
    }

    func finish() {
        let waiters = state.withLock { state -> [CheckedContinuation<Void, Never>] in
            state.isDone = true
            defer { state.waiters.removeAll() }
            return state.waiters
        }
        for waiter in waiters { waiter.resume() }
    }
}

// MARK: - Arbiter bookkeeping

struct AppInferenceLeaseWaiter {
    let id: UUID
    let owner: AppInferenceOwner
    /// A lease granted to `stream()` is released when that generation ends;
    /// one granted to `acquire()` lives until `release(_:)`.
    let releasesOnGenerationEnd: Bool
    /// Set when the waiter was created by `stream()`, so the stream's
    /// termination can withdraw it without knowing the owner.
    let generationID: UUID?
    let future: AppInferenceLeaseFuture
    var timeout: Task<Void, Never>?
}

/// FIFO within a class; `.chat` is admitted ahead of every queued `.http`.
struct AppInferenceWaiterQueue {
    private var chat: [AppInferenceLeaseWaiter] = []
    private var http: [AppInferenceLeaseWaiter] = []

    var queuedChat: Bool { !chat.isEmpty }
    var queuedHTTPCount: Int { http.count }
    var isEmpty: Bool { chat.isEmpty && http.isEmpty }

    func contains(owner: AppInferenceOwner) -> Bool {
        chat.contains { $0.owner == owner } || http.contains { $0.owner == owner }
    }

    func contains(id: UUID) -> Bool {
        chat.contains { $0.id == id } || http.contains { $0.id == id }
    }

    mutating func append(_ waiter: AppInferenceLeaseWaiter) {
        switch waiter.owner {
        case .chat: chat.append(waiter)
        case .http: http.append(waiter)
        }
    }

    mutating func popNext() -> AppInferenceLeaseWaiter? {
        if !chat.isEmpty { return chat.removeFirst() }
        if !http.isEmpty { return http.removeFirst() }
        return nil
    }

    mutating func remove(id: UUID) -> AppInferenceLeaseWaiter? {
        remove { $0.id == id }
    }

    mutating func remove(owner: AppInferenceOwner) -> AppInferenceLeaseWaiter? {
        remove { $0.owner == owner }
    }

    mutating func remove(generationID: UUID) -> AppInferenceLeaseWaiter? {
        remove { $0.generationID == generationID }
    }

    mutating func setTimeout(_ task: Task<Void, Never>, for id: UUID) -> Bool {
        if let index = chat.firstIndex(where: { $0.id == id }) {
            chat[index].timeout = task
            return true
        }
        if let index = http.firstIndex(where: { $0.id == id }) {
            http[index].timeout = task
            return true
        }
        return false
    }

    mutating func removeAll() -> [AppInferenceLeaseWaiter] {
        defer {
            chat.removeAll()
            http.removeAll()
        }
        return chat + http
    }

    private mutating func remove(
        where predicate: (AppInferenceLeaseWaiter) -> Bool
    ) -> AppInferenceLeaseWaiter? {
        if let index = chat.firstIndex(where: predicate) {
            return chat.remove(at: index)
        }
        if let index = http.firstIndex(where: predicate) {
            return http.remove(at: index)
        }
        return nil
    }
}

/// A generation the arbiter is forwarding from the client to one consumer.
struct AppInferenceGenerationRecord {
    let id: UUID
    let continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation
    let completion = AppInferenceCompletion()
    var lastEvent: ContinuousClock.Instant
    var watchdog: Task<Void, Never>?
}

struct AppInferenceActiveLease {
    let ticket: AppInferenceLeaseTicket
    let releasesOnGenerationEnd: Bool
    /// `release(_:)` arrived while a generation was in flight; honoured when
    /// that generation ends.
    var releaseRequested = false
    /// `cancel(owner:)` arrived after admission but before the generation
    /// touched the client; the generation ends without ever starting.
    var cancelRequested = false
    /// The generation `stream()` has scheduled but not yet started.
    var pendingGenerationID: UUID?
    var generation: AppInferenceGenerationRecord?
}

// MARK: - Activity fan-out

/// Publishes `AppInferenceActivity` changes to any number of independent
/// `AsyncStream`s. Lives outside the actor so `activityStream` can be
/// `nonisolated` and register a subscriber synchronously.
final class AppInferenceActivityBroadcaster: Sendable {
    private struct State {
        var latest: AppInferenceActivity
        var subscribers: [UUID: AsyncStream<AppInferenceActivity>.Continuation] = [:]
    }

    private let state: Mutex<State>

    init(initial: AppInferenceActivity) {
        state = Mutex(State(latest: initial))
    }

    var latest: AppInferenceActivity {
        state.withLock { $0.latest }
    }

    /// The first element is the current activity; every later element is a
    /// change. Registration and the snapshot happen under one lock so a
    /// concurrent `publish` can never be observed out of order.
    func subscribe() -> AsyncStream<AppInferenceActivity> {
        AsyncStream { continuation in
            let id = UUID()
            continuation.onTermination = { [self] _ in
                state.withLock { $0.subscribers[id] = nil }
            }
            state.withLock { state in
                state.subscribers[id] = continuation
                continuation.yield(state.latest)
            }
        }
    }

    func publish(_ activity: AppInferenceActivity) {
        state.withLock { state in
            guard state.latest != activity else { return }
            state.latest = activity
            for continuation in state.subscribers.values {
                continuation.yield(activity)
            }
        }
    }

    func finishAll() {
        let continuations = state.withLock { state
            -> [AsyncStream<AppInferenceActivity>.Continuation] in
            defer { state.subscribers.removeAll() }
            return Array(state.subscribers.values)
        }
        for continuation in continuations { continuation.finish() }
    }
}
