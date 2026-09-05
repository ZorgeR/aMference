import Foundation
import Synchronization
import Testing
@testable import MferenceAppCore

@Suite struct AppInferenceArbiterTests {
    private static let request = AppGenerationRequest(
        modelDirectory: URL(fileURLWithPath: "/tmp/arbiter.gturbo"), prompt: "hello")

    // MARK: Admission order

    @Test func leasesAreFIFOWithinAClass() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        let (a, b, c) = (UUID(), UUID(), UUID())

        let first = try await arbiter.acquire(.http(a), timeout: nil)
        let second = Task { try await arbiter.acquire(.http(b), timeout: nil) }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 1 }
        let third = Task { try await arbiter.acquire(.http(c), timeout: nil) }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 2 }
        #expect(await arbiter.activity.activeOwner == .http(a))

        await arbiter.release(first)
        let secondTicket = try await second.value
        #expect(secondTicket.owner == .http(b))
        #expect(await arbiter.activity.activeOwner == .http(b))
        #expect(await arbiter.activity.queuedHTTPCount == 1)

        await arbiter.release(secondTicket)
        let thirdTicket = try await third.value
        #expect(thirdTicket.owner == .http(c))
        await arbiter.release(thirdTicket)
        #expect(await arbiter.activity == .idle)
    }

    @Test func chatIsAdmittedAheadOfQueuedHTTP() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        let (a, b) = (UUID(), UUID())

        let first = try await arbiter.acquire(.http(a), timeout: nil)
        let http = Task { try await arbiter.acquire(.http(b), timeout: nil) }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 1 }
        let chat = Task { try await arbiter.acquire(.chat, timeout: nil) }
        try await waitUntil { await arbiter.activity.queuedChat }

        await arbiter.release(first)
        let chatTicket = try await chat.value
        #expect(chatTicket.owner == .chat)
        #expect(await arbiter.activity.activeOwner == .chat)
        #expect(await arbiter.activity.queuedHTTPCount == 1)
        #expect(!(await arbiter.activity.queuedChat))

        await arbiter.release(chatTicket)
        let httpTicket = try await http.value
        #expect(httpTicket.owner == .http(b))
        await arbiter.release(httpTicket)
    }

    @Test func acquireTimesOutBehindABusySession() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        let ticket = try await arbiter.acquire(.http(UUID()), timeout: nil)

        await #expect(throws: AppInferenceArbiterError.busyTimeout) {
            _ = try await arbiter.acquire(.http(UUID()), timeout: .milliseconds(20))
        }
        #expect(await arbiter.activity.queuedHTTPCount == 0)
        #expect(await arbiter.activity.activeOwner == ticket.owner)
        await arbiter.release(ticket)
    }

    // MARK: No preemption

    @Test func chatNeverPreemptsAnActiveGeneration() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let httpID = UUID()
        let httpTicket = try await arbiter.acquire(.http(httpID), timeout: nil)
        let httpEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .http(httpID)))
        }
        try await waitUntil { client.generateCount == 1 }

        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { await arbiter.activity.queuedChat }
        try await Task.sleep(for: .milliseconds(20))
        #expect(client.cancelCount == 0)
        #expect(client.generateCount == 1)
        #expect(await arbiter.activity.activeOwner == .http(httpID))

        client.emitToken("http")
        client.finishCurrent()
        let http = try await httpEvents.value
        #expect(http.contains(.token(AppTokenEvent(index: 0, textDelta: "http", elapsedDecodeSeconds: 0.01))))
        // The explicit lease outlives its generation; chat waits for release.
        try await Task.sleep(for: .milliseconds(20))
        #expect(await arbiter.activity.activeOwner == .http(httpID))
        #expect(client.generateCount == 1)

        await arbiter.release(httpTicket)
        try await waitUntil { client.generateCount == 2 }
        #expect(await arbiter.activity.activeOwner == .chat)
        client.emitToken("chat")
        client.finishCurrent()
        let chat = try await chatEvents.value
        #expect(chat.contains(.token(AppTokenEvent(index: 0, textDelta: "chat", elapsedDecodeSeconds: 0.01))))
        #expect(client.maxConcurrentStreams == 1)
        #expect(await arbiter.activity == .idle)
    }

    // MARK: Lease-scoped cancellation

    @Test func cancellingAQueuedOwnerNeverTouchesTheWire() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let httpID = UUID()
        let httpEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .http(httpID)))
        }
        try await waitUntil { client.generateCount == 1 }
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { await arbiter.activity.queuedChat }

        await arbiter.cancel(owner: .chat)

        await #expect(throws: AppInferenceArbiterError.cancelled) {
            _ = try await chatEvents.value
        }
        #expect(client.cancelCount == 0)
        #expect(!(await arbiter.activity.queuedChat))
        #expect(await arbiter.activity.activeOwner == .http(httpID))

        client.finishCurrent()
        _ = try await httpEvents.value
        #expect(await arbiter.activity == .idle)
    }

    @Test func cancellingTheActiveOwnerReachesTheClient() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }

        await arbiter.cancel(owner: .chat)

        let events = try await chatEvents.value
        #expect(client.cancelCount == 1)
        guard case .cancelled = events.last else {
            Issue.record("expected a cancelled terminal event, received \(events)")
            return
        }
        #expect(await arbiter.activity == .idle)
    }

    @Test func cancellingAnUnrelatedOwnerIsANoOp() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }

        await arbiter.cancel(owner: .http(UUID()))
        #expect(client.cancelCount == 0)

        client.finishCurrent()
        _ = try await chatEvents.value
    }

    @Test func abandonedStreamCancelsTheWireAndStillReleases() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let consumer = Task {
            for try await event in await arbiter.stream(Self.request, owner: .chat) {
                if case .token = event { break }
            }
        }
        try await waitUntil { client.generateCount == 1 }
        client.emitToken("x")
        try await consumer.value

        try await waitUntil { client.cancelCount == 1 }
        try await waitUntil { await arbiter.activity == .idle }
    }

    // MARK: Exclusive sessions

    @Test func exclusiveSessionAwaitsTheTerminalEventBeforeItsBody() async throws {
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client)
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }
        let queued = Task { try await arbiter.acquire(.http(UUID()), timeout: nil) }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 1 }

        let bodyRan = Flag()
        let exclusive = Task {
            try await arbiter.withExclusiveSession(reason: "Reloading model") {
                bodyRan.set()
                return 42
            }
        }
        try await waitUntil { client.cancelCount == 1 }
        try await Task.sleep(for: .milliseconds(30))
        #expect(!bodyRan.isSet, "body must wait for the terminal event")
        #expect(await arbiter.activity.activeOwner == .chat)
        // Admission is closed for the whole drain; the already-queued waiter
        // is failed once the terminal event has arrived.
        await #expect(throws: AppInferenceArbiterError.unavailable("Reloading model")) {
            _ = try await arbiter.acquire(.http(UUID()), timeout: nil)
        }
        #expect(await arbiter.activity.queuedHTTPCount == 1)

        client.finishCurrent(.cancelled)
        await #expect(throws: AppInferenceArbiterError.unavailable("Reloading model")) {
            _ = try await queued.value
        }
        #expect(try await exclusive.value == 42)
        #expect(bodyRan.isSet)
        let events = try await chatEvents.value
        guard case .cancelled = events.last else {
            Issue.record("expected the drained generation to end cancelled, received \(events)")
            return
        }
        #expect(await arbiter.activity == .idle)
        let ticket = try await arbiter.acquire(.chat, timeout: nil)
        await arbiter.release(ticket)
    }

    @Test func exclusiveSessionRevokesAnIdleExplicitLease() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let ticket = try await arbiter.acquire(.http(UUID()), timeout: nil)

        let result = try await arbiter.withExclusiveSession(reason: "Unloading") {
            await arbiter.activity.activeOwner == nil
        }
        #expect(result)
        #expect(client.cancelCount == 0)
        await arbiter.release(ticket)
        #expect(await arbiter.activity == .idle)
    }

    @Test func exclusiveSessionsSerialise() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        let firstStarted = Flag()
        let firstMayFinish = Flag()
        let secondStarted = Flag()
        let first = Task {
            try await arbiter.withExclusiveSession(reason: "first") {
                firstStarted.set()
                while !firstMayFinish.isSet { try await Task.sleep(for: .milliseconds(2)) }
            }
        }
        try await waitUntil { firstStarted.isSet }
        let second = Task {
            try await arbiter.withExclusiveSession(reason: "second") { secondStarted.set() }
        }
        try await Task.sleep(for: .milliseconds(30))
        #expect(!secondStarted.isSet)
        firstMayFinish.set()
        try await first.value
        try await second.value
        #expect(secondStarted.isSet)
    }

    @Test func exclusiveSessionRethrowsAndReopensAdmission() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        await #expect(throws: AppInferenceError.modelLoadFailed("synthetic")) {
            try await arbiter.withExclusiveSession(reason: "Loading") {
                throw AppInferenceError.modelLoadFailed("synthetic")
            }
        }
        let ticket = try await arbiter.acquire(.chat, timeout: nil)
        await arbiter.release(ticket)
    }

    // MARK: Watchdog

    private static let session = AppLoadedSession(
        modelDirectory: URL(fileURLWithPath: "/tmp/arbiter.gturbo"),
        maxContextTokens: 4_096, runtimeOptions: AppRuntimeOptions(),
        forceLogitsHead: true, temperature: 0.2, topK: 64, topP: 0.95)

    @Test func watchdogIsOffByDefault() {
        #expect(AppInferenceArbiter(client: ScriptedInferenceClient()).watchdogDuration == nil)
        #expect(AppInferenceArbiter(client: ScriptedInferenceClient(),
                                    watchdog: .seconds(180)).watchdogDuration == .seconds(180))
    }

    /// The behavioural half: a short silence after the first event is not a
    /// teardown. It is bounded by what a test can wait for, so it is not
    /// proof of the default; `watchdogIsOffByDefault` asserts that.
    @Test func withoutAWatchdogSilenceAfterTheFirstEventIsNotATeardown() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }
        client.emitToken("first")
        try await Task.sleep(for: .milliseconds(120))
        #expect(client.cancelCount == 0)
        #expect(client.disconnectCount == 0)
        #expect(client.shutdownCount == 0)
        client.finishCurrent()
        let events = try await chatEvents.value
        guard case .finished = events.last else {
            Issue.record("expected a finished terminal event, received \(events)")
            return
        }
    }

    @Test func watchdogClockDoesNotStartBeforeTheFirstEvent() async throws {
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(40))
        await arbiter.publishLoadedSession(Self.session)
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }

        // Several budgets of silence before any event: the service is still
        // inside its first prefill chunk (first-touch hashing included).
        try await Task.sleep(for: .milliseconds(160))
        #expect(client.cancelCount == 0)
        #expect(client.disconnectCount == 0)
        #expect(client.shutdownCount == 0)
        #expect(await arbiter.activity.activeOwner == .chat)
        #expect(await arbiter.loadedSession == Self.session)

        // The first event starts the clock; the same silence now fires it.
        client.emitToken("first")
        var failure: AppInferenceArbiterError?
        do {
            _ = try await chatEvents.value
        } catch let error as AppInferenceArbiterError {
            failure = error
        }
        #expect(failure == .transportLost)
        #expect(client.cancelCount == 1)
        #expect(client.disconnectCount == 1)
        try await waitUntil { client.shutdownCount == 1 }
        #expect(await arbiter.loadedSession == nil)
        try await waitUntil { await arbiter.activity == .idle }
    }

    @Test func watchdogFailsTheLeaseWithTransportLostAndPublishesNoSession() async throws {
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(40))
        await arbiter.publishLoadedSession(Self.session)
        let recorder = ActivityRecorder(arbiter.activityStream)
        try await waitUntil { recorder.count == 1 }
        let httpID = UUID()
        let httpEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .http(httpID)))
        }
        try await waitUntil { client.generateCount == 1 }
        client.emitToken("first")

        await #expect(throws: AppInferenceArbiterError.transportLost) {
            _ = try await httpEvents.value
        }
        #expect(AppInferenceArbiterError.transportLost.description
                == AppInferenceArbiterError.transportLostMessage)
        #expect(AppInferenceArbiterError.transportLost.inferenceError
                == .unknown(AppInferenceArbiterError.transportLostMessage))
        // The lease is released and the session is gone with the transport.
        try await waitUntil { await arbiter.activity == .idle }
        #expect(recorder.snapshot.last == .idle)
        #expect(recorder.snapshot.contains { $0.activeOwner == .http(httpID) && $0.loadedSession == Self.session })
        #expect(client.disconnectCount == 1)
        try await waitUntil { client.shutdownCount == 1 }
        #expect(client.maxConcurrentStreams == 1)
    }

    /// A waiter queued behind the watchdogged generation is admitted only
    /// once the pipe is dead: `disconnect()` runs on the actor before the
    /// lease is released, so the next owner fails on the client's missing
    /// handles instead of opening a second reader on a transport that is
    /// being torn down.
    @Test func watchdogKillsThePipeBeforeTheNextOwnerIsAdmitted() async throws {
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(40))
        await arbiter.publishLoadedSession(Self.session)
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }
        let httpID = UUID()
        let httpEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .http(httpID)))
        }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 1 }

        client.emitToken("first")
        await #expect(throws: AppInferenceArbiterError.transportLost) {
            _ = try await chatEvents.value
        }
        await #expect(throws: AppInferenceError.modelNotLoaded) {
            _ = try await httpEvents.value
        }
        #expect(client.disconnectCount == 1)
        #expect(client.generateCount == 1, "no second generation reached the transport")
        #expect(client.refusedGenerateCount == 1)
        #expect(client.maxConcurrentStreams == 1)
        #expect(client.cancelCount == 1)
        #expect(await arbiter.loadedSession == nil)
        try await waitUntil { client.shutdownCount == 1 }
        try await waitUntil { await arbiter.activity == .idle }
    }

    /// The same teardown with an explicit `acquire` waiter behind it, the
    /// shape an HTTP request takes: the ticket is handed out with the session
    /// already gone, so the bridge's session check refuses the request before
    /// it touches the client.
    @Test func watchdogDropsTheSessionBeforeAnExplicitWaiterIsAdmitted() async throws {
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(40))
        await arbiter.publishLoadedSession(Self.session)
        let recorder = ActivityRecorder(arbiter.activityStream)
        try await waitUntil { recorder.count == 1 }
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }
        let httpID = UUID()
        let queued = Task { try await arbiter.acquire(.http(httpID), timeout: nil) }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 1 }

        client.emitToken("first")
        await #expect(throws: AppInferenceArbiterError.transportLost) {
            _ = try await chatEvents.value
        }
        let ticket = try await queued.value
        #expect(ticket.owner == .http(httpID))
        #expect(await arbiter.loadedSession == nil)
        #expect(client.disconnectCount == 1)
        // The admission that granted the ticket already carried no session.
        try await waitUntil { recorder.snapshot.contains { $0.activeOwner == .http(httpID) } }
        #expect(recorder.snapshot.filter { $0.activeOwner == .http(httpID) }
            .allSatisfy { $0.loadedSession == nil })
        await arbiter.release(ticket)
        #expect(client.generateCount == 1)
        try await waitUntil { client.shutdownCount == 1 }
        try await waitUntil { await arbiter.activity == .idle }
    }

    @Test func watchdogIsResetByEvents() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(60))
        let chatEvents = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }
        for index in 0..<5 {
            try await Task.sleep(for: .milliseconds(25))
            client.emitToken("t", index: index)
        }
        client.finishCurrent()
        let events = try await chatEvents.value
        #expect(client.disconnectCount == 0)
        #expect(client.shutdownCount == 0)
        #expect(events.filter { if case .token = $0 { true } else { false } }.count == 5)
    }

    // MARK: Explicit tickets and streams

    @Test func streamReusesAHeldTicketAndKeepsTheLease() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let httpID = UUID()
        let ticket = try await arbiter.acquire(.http(httpID), timeout: nil)
        let deltas = TextDeltaRecorder()
        let events = Task {
            try await collect(await arbiter.stream(Self.request, owner: .http(httpID)) {
                deltas.append($0)
            })
        }
        try await waitUntil { client.generateCount == 1 }
        client.emitToken("a")
        client.finishCurrent()
        _ = try await events.value
        #expect(await arbiter.activity.activeOwner == .http(httpID))

        await arbiter.release(ticket)
        #expect(await arbiter.activity == .idle)
        // A revoked or released ticket is ignored.
        await arbiter.release(ticket)
        #expect(await arbiter.activity == .idle)
    }

    @Test func releaseDuringAGenerationTakesEffectWhenItEnds() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let httpID = UUID()
        let ticket = try await arbiter.acquire(.http(httpID), timeout: nil)
        let events = Task {
            try await collect(await arbiter.stream(Self.request, owner: .http(httpID)))
        }
        try await waitUntil { client.generateCount == 1 }

        await arbiter.release(ticket)
        #expect(await arbiter.activity.activeOwner == .http(httpID))

        client.finishCurrent()
        _ = try await events.value
        try await waitUntil { await arbiter.activity == .idle }
    }

    @Test func aSecondStreamForTheSameOwnerIsRejected() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let first = Task {
            try await collect(await arbiter.stream(Self.request, owner: .chat))
        }
        try await waitUntil { client.generateCount == 1 }

        var failure: AppInferenceArbiterError?
        do {
            _ = try await collect(await arbiter.stream(Self.request, owner: .chat))
        } catch let error as AppInferenceArbiterError {
            failure = error
        }
        guard case .unavailable = failure else {
            Issue.record("expected unavailable, received \(String(describing: failure))")
            return
        }
        client.finishCurrent()
        _ = try await first.value
    }

    // MARK: Activity

    @Test func activityStreamFansOutToEverySubscriber() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        let session = AppLoadedSession(
            modelDirectory: URL(fileURLWithPath: "/tmp/arbiter.gturbo"),
            maxContextTokens: 4_096, runtimeOptions: AppRuntimeOptions(),
            forceLogitsHead: true, temperature: 0.2, topK: 64, topP: 0.95)
        let first = ActivityRecorder(arbiter.activityStream)
        let second = ActivityRecorder(arbiter.activityStream)
        try await waitUntil { first.count == 1 && second.count == 1 }
        #expect(first.snapshot.first == .idle)

        await arbiter.publishLoadedSession(session)
        let ticket = try await arbiter.acquire(.chat, timeout: nil)
        await arbiter.release(ticket)
        await arbiter.publishLoadedSession(nil)

        let expected: [AppInferenceActivity] = [
            .idle,
            AppInferenceActivity(activeOwner: nil, queuedChat: false, queuedHTTPCount: 0,
                                 loadedSession: session),
            AppInferenceActivity(activeOwner: .chat, queuedChat: false, queuedHTTPCount: 0,
                                 loadedSession: session),
            AppInferenceActivity(activeOwner: nil, queuedChat: false, queuedHTTPCount: 0,
                                 loadedSession: session),
            .idle,
        ]
        try await waitUntil { first.count == expected.count && second.count == expected.count }
        #expect(first.snapshot == expected)
        #expect(second.snapshot == expected)
        #expect(await arbiter.loadedSession == nil)
    }

    @Test func aLateSubscriberStartsFromTheCurrentActivity() async throws {
        let arbiter = AppInferenceArbiter(client: ScriptedInferenceClient())
        let ticket = try await arbiter.acquire(.http(UUID()), timeout: nil)
        let recorder = ActivityRecorder(arbiter.activityStream)
        try await waitUntil { recorder.count == 1 }
        #expect(recorder.snapshot.first?.activeOwner == ticket.owner)
        await arbiter.release(ticket)
        try await waitUntil { recorder.count == 2 }
        #expect(recorder.snapshot.last == .idle)
    }

    // MARK: Helpers

    private func collect(_ stream: AsyncThrowingStream<AppInferenceEvent, Error>)
        async throws -> [AppInferenceEvent] {
        var events: [AppInferenceEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }
}

private func waitUntil(_ predicate: @Sendable () async -> Bool,
                       timeout: Duration = .seconds(5)) async throws {
    let deadline = ContinuousClock.now + timeout
    while ContinuousClock.now < deadline {
        if await predicate() { return }
        try await Task.sleep(for: .milliseconds(2))
    }
    Issue.record("timed out waiting for condition")
    throw ArbiterTestError.timedOut
}

private enum ArbiterTestError: Error {
    case timedOut
}

private final class Flag: Sendable {
    private let value = Mutex(false)
    var isSet: Bool { value.withLock { $0 } }
    func set() { value.withLock { $0 = true } }
}

private final class ActivityRecorder: Sendable {
    private final class Values: Sendable {
        let list = Mutex<[AppInferenceActivity]>([])
    }

    private let values: Values
    private let task: Task<Void, Never>

    init(_ stream: AsyncStream<AppInferenceActivity>) {
        let values = Values()
        self.values = values
        task = Task {
            for await activity in stream {
                values.list.withLock { $0.append(activity) }
            }
        }
    }

    deinit { task.cancel() }

    var count: Int { values.list.withLock { $0.count } }
    var snapshot: [AppInferenceActivity] { values.list.withLock { $0 } }
}
