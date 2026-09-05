import Foundation
import Synchronization
import Testing
import MferenceAppCore
import MferenceServerBridge
import MferenceServerCore

private final class Counter: Sendable {
    private let count = Mutex(0)
    func increment() { count.withLock { $0 += 1 } }
    var value: Int { count.withLock { $0 } }
}

@Suite("API server log sink")
struct APIServerLogSinkTests {
    private static func routed(_ path: String = "/health") -> ServerLogEvent {
        ServerLogEvent(method: "GET", path: path, kind: .routed(status: 200))
    }

    @Test func ringDropsTheOldestAndCountsTheDrops() {
        let sink = APIServerLogSink(capacity: 5, flushDelay: .milliseconds(1))
        for _ in 1...8 { sink.record(Self.routed()) }
        let snapshot = sink.snapshot
        #expect(snapshot.entries.map(\.id) == [4, 5, 6, 7, 8])
        #expect(snapshot.droppedBefore == 3)

        sink.clear()
        #expect(sink.snapshot == .empty)
        // Ids keep counting after a clear, so a row never changes identity.
        sink.record(Self.routed())
        #expect(sink.snapshot.entries.map(\.id) == [9])
        #expect(sink.snapshot.droppedBefore == 0)
    }

    @Test func flushIsCoalescedIntoOneMainActorHop() async throws {
        let sink = APIServerLogSink(capacity: 500, flushDelay: .milliseconds(20))
        let flushes = Counter()
        sink.setFlushHandler {
            MainActor.assertIsolated()
            flushes.increment()
        }
        for _ in 0..<200 { sink.record(Self.routed()) }
        // Never synchronous: the event loop must not run observers.
        #expect(flushes.value == 0)
        try await waitUntil { flushes.value == 1 }
        try await Task.sleep(for: .milliseconds(80))
        #expect(flushes.value == 1)

        sink.record(APIServerBridgeNote(kind: .stopStringMatched))
        try await waitUntil { flushes.value == 2 }
    }

    @Test func recordNeverBlocksTheCaller() async throws {
        let sink = APIServerLogSink(capacity: 500, flushDelay: .milliseconds(5))
        let flushes = Counter()
        sink.setFlushHandler { flushes.increment() }
        let start = ContinuousClock.now
        await Task.detached {
            for _ in 0..<5000 { sink.record(Self.routed()) }
        }.value
        #expect(ContinuousClock.now - start < .seconds(2))
        let snapshot = sink.snapshot
        #expect(snapshot.entries.count == 500)
        #expect(snapshot.droppedBefore == 4500)
        try await waitUntil { flushes.value >= 1 }
    }

    @Test func bridgeNotesInterleaveWithServerEventsInOrder() {
        let sink = APIServerLogSink(capacity: 10, flushDelay: .milliseconds(1))
        sink.record(Self.routed("/v1/models"))
        sink.record(APIServerBridgeNote(kind: .waitingForSession(behind: .chat)))
        sink.record(ServerLogEvent(id: "chatcmpl-1", method: "POST", path: "/v1/chat/completions",
                                   requestedModel: "test-model",
                                   kind: .requestStarted(method: "POST", path: "/v1/chat/completions",
                                                         streaming: true)))
        let entries = sink.snapshot.entries
        #expect(entries.map(\.id) == [1, 2, 3])
        #expect(entries[0].serverEvent?.path == "/v1/models")
        #expect(entries[1].bridgeNote == .waitingForSession(behind: .chat))
        #expect(entries[2].serverEvent?.id == "chatcmpl-1")
    }

    @Test func noHandlerMeansRowsStillAccumulate() async throws {
        let sink = APIServerLogSink(capacity: 3, flushDelay: .milliseconds(1))
        sink.record(Self.routed())
        try await Task.sleep(for: .milliseconds(20))
        sink.record(Self.routed())
        #expect(sink.snapshot.entries.count == 2)
    }

    @Test func presentationFlagsPollingNoiseAndSeverity() {
        func entry(_ payload: APIServerLogEntry.Payload) -> APIServerLogEntry {
            APIServerLogEntry(id: 1, payload: payload)
        }
        let health = APIServerLogEntryPresentation.resolve(entry(.server(Self.routed("/health"))))
        #expect(health.isPollingNoise)
        #expect(health.summary == "GET /health 200")
        #expect(health.severity == .neutral)
        let models = APIServerLogEntryPresentation.resolve(entry(.server(Self.routed("/v1/models"))))
        #expect(models.isPollingNoise)
        let missing = APIServerLogEntryPresentation.resolve(entry(.server(ServerLogEvent(
            method: "GET", path: "/v1/nope", kind: .routed(status: 404)))))
        #expect(!missing.isPollingNoise)

        let rejected = APIServerLogEntryPresentation.resolve(entry(.server(ServerLogEvent(
            method: "POST", path: "/v1/chat/completions", requestedModel: "gpt-4o",
            kind: .requestRejected(status: 404, code: "model_not_found",
                                   message: "requested model is not available")))))
        #expect(rejected.summary == "POST /v1/chat/completions 404 model_not_found")
        #expect(rejected.detail == "requested model is not available · model: gpt-4o")
        #expect(rejected.severity == .warning)
        #expect(!rejected.isPollingNoise)

        let failed = APIServerLogEntryPresentation.resolve(entry(.server(ServerLogEvent(
            id: "chatcmpl-9", method: "POST", path: "/v1/chat/completions",
            kind: .requestFailed(status: 500, streaming: true, detail: "boom")))))
        #expect(failed.severity == .error)
        #expect(failed.requestID == "chatcmpl-9")
        #expect(failed.detail == "boom")
        #expect(failed.summary.contains("500"))

        let completed = APIServerLogEntryPresentation.resolve(entry(.server(ServerLogEvent(
            id: "chatcmpl-2", method: "POST", path: "/v1/chat/completions",
            kind: .requestCompleted(duration: .milliseconds(1500), promptTokens: 12,
                                    cachedTokens: 0, completionTokens: 34,
                                    finishReason: "stop")))))
        #expect(completed.summary == "POST /v1/chat/completions 200 · 1.5s · prompt 12 · completion 34 · stop")
        #expect(completed.severity == .success)

        let substitution = APIServerLogEntryPresentation.resolve(entry(.bridge(APIServerBridgeNote(
            kind: .substitution(APIServerSubstitution(
                parameter: "temperature", requested: "0.7", applied: "0", reason: "greedy session"))))))
        #expect(substitution.summary == "temperature pinned: 0.7 → 0")
        #expect(substitution.detail == "greedy session")
        #expect(substitution.severity == .warning)

        let listening = APIServerLogEntryPresentation.resolve(entry(.bridge(APIServerBridgeNote(
            kind: .listening(host: "127.0.0.1", port: 8080, modelID: "test-model")))))
        #expect(listening.summary == "Listening on http://127.0.0.1:8080/v1")
        #expect(listening.severity == .success)
    }
}
