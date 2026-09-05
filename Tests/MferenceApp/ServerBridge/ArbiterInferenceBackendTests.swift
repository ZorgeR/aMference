import Foundation
import Synchronization
import Testing
import Mference
import MferenceAppCore
import MferenceServerBridge
import MferenceServerCore

/// Drives a real `MferenceHTTPServer` on an ephemeral port through
/// `APIServerModel` and `ArbiterInferenceBackend`, over an arbiter whose
/// client is the model-free fake. This is the headless proof: what `curl`
/// gets from a started `APIServerModel`, asserted byte by byte.
@Suite("Arbiter inference backend over HTTP", .serialized)
@MainActor
struct ArbiterInferenceBackendTests {
    private static let response = "alpha beta gamma delta epsilon"

    private static func body(_ extra: String = "",
                             messages: String = #"[{"role":"user","content":"hello there"}]"#) -> String {
        #"{"model":"test-model","messages":\#(messages)\#(extra)}"#
    }

    @Test func streamingDeliversContentDeltasThenStopAndDone() async throws {
        let harness = try await BridgeHarness.start(
            client: BridgeFakeInferenceClient(response: Self.response))
        let (status, frames) = try await streamCompletion(
            Self.body(#","stream":true,"stream_options":{"include_usage":true}"#),
            port: harness.port)
        #expect(status == 200)
        let deltas = contentDeltas(frames)
        // One chunk per delivered piece: the un-throttled hook, not the chat cadence.
        #expect(deltas.count == 5)
        #expect(deltas.joined() == Self.response)
        #expect(finishReasons(frames) == ["stop"])
        #expect(frames.last == "[DONE]")
        let usageFrame = try #require(frames.first { $0.contains(#""usage""#) })
        #expect(usageFrame.contains(#""completion_tokens":5"#))
        #expect(usageFrame.contains(#""cached_tokens":0"#))
        // The role chunk comes first, before any content.
        #expect(frames.first?.contains(#""role":"assistant""#) == true)
        await harness.stop()
    }

    @Test func nonStreamingAccumulatesTheWholeAnswer() async throws {
        let harness = try await BridgeHarness.start(
            client: BridgeFakeInferenceClient(response: Self.response))
        let reply = try await completion(Self.body(), port: harness.port)
        #expect(reply.status == 200)
        #expect(messageContent(reply) == Self.response)
        #expect(finishReason(reply) == "stop")
        #expect(reply.json["model"] as? String == "test-model")
        let usage = usage(reply)
        // The fake reports no prompt count, so `prepare`'s measurement is used.
        #expect(usage["prompt_tokens"] as? Int == 2)
        #expect(usage["completion_tokens"] as? Int == 5)
        #expect(usage["total_tokens"] as? Int == 7)
        #expect((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 0)

        let request = try #require(harness.client.lastRequest)
        #expect(request.messages == [AppGenerationMessage(role: .user, content: "hello there")])
        #expect(request.modelDirectory == BridgeFixtures.modelDirectory)
        #expect(request.maxContextTokens == 4096)
        #expect(request.temperature == 0.2)
        await harness.stop()
    }

    @Test func serviceReportedPromptCountWins() async throws {
        let harness = try await BridgeHarness.start(
            client: BridgeFakeInferenceClient(response: Self.response, promptTokenCount: 41))
        let reply = try await completion(Self.body(), port: harness.port)
        #expect(usage(reply)["prompt_tokens"] as? Int == 41)
        await harness.stop()
    }

    @Test func stopStringTruncatesAndCancelsTheLease() async throws {
        let client = BridgeFakeInferenceClient(response: "alpha beta STOP gamma delta")
        let harness = try await BridgeHarness.start(client: client)
        let reply = try await completion(Self.body(#","stop":["STOP"]"#), port: harness.port)
        #expect(reply.status == 200)
        #expect(messageContent(reply) == "alpha beta ")
        #expect(finishReason(reply) == "stop")
        #expect(client.cancelCount == 1)
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.contains { $0.bridgeNote == .stopStringMatched }
            }
        }

        // Streaming: no chunk ever carries the stop string or what follows it.
        let (status, frames) = try await streamCompletion(
            Self.body(#","stream":true,"stop":"STOP""#), port: harness.port)
        #expect(status == 200)
        #expect(contentDeltas(frames).joined() == "alpha beta ")
        #expect(!frames.joined().contains("gamma"))
        #expect(finishReasons(frames) == ["stop"])
        #expect(client.cancelCount == 2)
        #expect(await harness.arbiter.activity.activeOwner == nil)
        await harness.stop()
    }

    @Test func maxTokensEndsWithLength() async throws {
        let harness = try await BridgeHarness.start(
            client: BridgeFakeInferenceClient(response: Self.response))
        let reply = try await completion(Self.body(#","max_tokens":2"#), port: harness.port)
        #expect(reply.status == 200)
        #expect(messageContent(reply) == "alpha beta")
        #expect(finishReason(reply) == "length")
        #expect(usage(reply)["completion_tokens"] as? Int == 2)
        await harness.stop()
    }

    @Test func noLoadedSessionAnswers503ButHealthStaysUp() async throws {
        let harness = try await BridgeHarness.start()
        await harness.arbiter.publishLoadedSession(nil)
        let reply = try await completion(Self.body(), port: harness.port)
        #expect(reply.status == 503)
        #expect(errorCode(reply) == "service_unavailable")
        #expect(errorMessage(reply) == "no model is loaded")
        #expect(harness.client.generateCount == 0)

        // A stream is rejected before the head, so it keeps the status too.
        let (status, frames) = try await streamCompletion(Self.body(#","stream":true"#),
                                                          port: harness.port)
        #expect(status == 503)
        #expect(frames.isEmpty)

        #expect(try await send("GET", "/health", port: harness.port, contentType: nil).status == 200)
        let models = try await send("GET", "/v1/models", port: harness.port, contentType: nil)
        #expect(models.status == 200)
        #expect(models.text.contains(#""id":"test-model""#))
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.contains { $0.bridgeNote == .modelNotLoaded }
            }
        }
        await harness.stop()
    }

    @Test func busySessionAnswers429AfterTheBusyTimeout() async throws {
        let harness = try await BridgeHarness.start(busyTimeout: .milliseconds(150))
        let chat = try await harness.arbiter.acquire(.chat, timeout: nil)
        let started = ContinuousClock.now
        let reply = try await completion(Self.body(), port: harness.port)
        #expect(reply.status == 429)
        #expect(errorCode(reply) == "queue_full")
        #expect(ContinuousClock.now - started >= .milliseconds(150))
        #expect(harness.client.generateCount == 0)
        let isFailure: @Sendable (ServerLogEvent) -> Bool = { event in
            if case .requestFailed = event.kind { return true }
            return false
        }
        // The observable copy drains on a coalesced tick; wait for both rows.
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.contains {
                    $0.bridgeNote == .waitingForSession(behind: .chat)
                } && harness.model.logEntries.contains { $0.serverEvent.map(isFailure) == true }
            }
        }
        let failed = try #require(harness.model.logEntries.compactMap(\.serverEvent).last(where: isFailure))
        if case .requestFailed(let status, _, let detail) = failed.kind {
            #expect(status == 429)
            #expect(detail.hasPrefix("queue_full"))
        }
        await harness.arbiter.release(chat)
        #expect(await harness.arbiter.activity.activeOwner == nil)
        await harness.stop()
    }

    @Test func requestWaitsForChatThenCompletes() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response, pieceDelay: .milliseconds(20))
        let harness = try await BridgeHarness.start(client: client, busyTimeout: .seconds(5))
        let chatRequest = AppGenerationRequest(modelDirectory: BridgeFixtures.modelDirectory,
                                               prompt: "chat first")
        let chatStream = await harness.arbiter.stream(chatRequest, owner: .chat)
        let chatDone = Task {
            var events = 0
            for try await _ in chatStream { events += 1 }
            return events
        }
        try await waitUntil { await harness.arbiter.activity.activeOwner == .chat }

        let reply = try await completion(Self.body(), port: harness.port)
        #expect(reply.status == 200)
        #expect(messageContent(reply) == Self.response)
        #expect(try await chatDone.value > 0)
        #expect(client.generateCount == 2)
        #expect(client.cancelCount == 0)
        #expect(await harness.arbiter.activity == AppInferenceActivity(
            activeOwner: nil, queuedChat: false, queuedHTTPCount: 0,
            loadedSession: BridgeFixtures.session()))
        await harness.stop()
    }

    @Test func toolsAreRefusedWith400() async throws {
        let harness = try await BridgeHarness.start()
        let reply = try await completion(
            Self.body(#","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]"#),
            port: harness.port)
        #expect(reply.status == 400)
        #expect(errorCode(reply) == "tools_not_supported")
        #expect(harness.client.generateCount == 0)

        let accepted = try await completion(Self.body(#","tools":[]"#), port: harness.port)
        #expect(accepted.status == 200)
        await harness.stop()
    }

    @Test func unknownModelIs404NamingTheSentModel() async throws {
        let harness = try await BridgeHarness.start()
        let reply = try await send("POST", "/v1/chat/completions", port: harness.port,
                                   body: #"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
        #expect(reply.status == 404)
        #expect(errorCode(reply) == "model_not_found")
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.contains { $0.serverEvent?.requestedModel == "gpt-4o" }
            }
        }
        let rejected = try #require(harness.model.logEntries.compactMap(\.serverEvent)
            .first { $0.requestedModel == "gpt-4o" })
        #expect(rejected.kind == .requestRejected(status: 404, code: "model_not_found",
                                                  message: "requested model is not available"))
        #expect(APIServerLogEntryPresentation.resolve(
            APIServerLogEntry(id: 1, payload: .server(rejected))).detail?.contains("gpt-4o") == true)
        await harness.stop()
    }

    @Test func modelsRouteListsTheServedID() async throws {
        let harness = try await BridgeHarness.start()
        let reply = try await send("GET", "/v1/models", port: harness.port, contentType: nil)
        #expect(reply.status == 200)
        let data = reply.json["data"] as? [[String: Any]]
        #expect(data?.map { $0["id"] as? String } == ["test-model"])
        await harness.stop()
    }

    @Test func samplingMismatchIs400UnlessPinned() async throws {
        let rejecting = try await BridgeHarness.start(
            session: BridgeFixtures.session(forceLogitsHead: false))
        let reply = try await completion(Self.body(), port: rejecting.port)
        #expect(reply.status == 400)
        #expect(errorCode(reply) == "sampling_mode_mismatch")
        #expect(rejecting.client.generateCount == 0)
        await rejecting.stop()

        let pinning = try await BridgeHarness.start(
            session: BridgeFixtures.session(forceLogitsHead: false), policy: .pinToSession)
        let pinned = try await completion(Self.body(#","temperature":0.9"#), port: pinning.port)
        #expect(pinned.status == 200)
        #expect(pinning.client.lastRequest?.temperature == 0)
        #expect(pinning.client.lastRequest?.isPureGreedy == true)
        try await waitUntil {
            await MainActor.run {
                pinning.model.logEntries.contains {
                    if case .substitution(let row) = $0.bridgeNote {
                        return row.parameter == "temperature" && row.requested == "0.9"
                    }
                    return false
                }
            }
        }
        await pinning.stop()
    }

    @Test func promptThatWouldDropHistoryIs400() async throws {
        // Six words of context; the earlier turn would have to go.
        let harness = try await BridgeHarness.start(session: BridgeFixtures.session(maxContext: 6))
        let long = Self.body(messages: #"""
        [{"role":"user","content":"one two three four five"},
         {"role":"assistant","content":"six"},
         {"role":"user","content":"seven eight"}]
        """#)
        let reply = try await completion(long, port: harness.port)
        #expect(reply.status == 400)
        #expect(errorCode(reply) == "context_length_exceeded")
        #expect(errorMessage(reply)?.contains("2 message(s)") == true)

        // A single turn that fits is served.
        let short = try await completion(Self.body(), port: harness.port)
        #expect(short.status == 200)

        // A single turn that cannot fit at all is the same 400.
        let huge = Self.body(messages: #"[{"role":"user","content":"a b c d e f g h"}]"#)
        let overflow = try await completion(huge, port: harness.port)
        #expect(overflow.status == 400)
        #expect(errorCode(overflow) == "context_length_exceeded")
        await harness.stop()
    }

    @Test func prepareDuringAnotherGenerationNeverTouchesTheClient() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response, pieceDelay: .milliseconds(30))
        let arbiter = AppInferenceArbiter(client: client)
        await arbiter.publishLoadedSession(BridgeFixtures.session())
        let sink = APIServerLogSink(flushDelay: .milliseconds(1))
        let backend = ArbiterInferenceBackend(arbiter: arbiter, log: sink,
                                              policy: .rejectOnMismatch,
                                              measurePrompt: BridgeFixtures.wordCountMeasurer)
        let chatStream = await arbiter.stream(
            AppGenerationRequest(modelDirectory: BridgeFixtures.modelDirectory, prompt: "busy"),
            owner: .chat)
        let chat = Task { for try await _ in chatStream {} }
        try await waitUntil { await arbiter.activity.activeOwner == .chat }
        #expect(client.generateCount == 1)

        let prepared = try await backend.prepare(try BridgeFixtures.validated(Self.body()))
        #expect(prepared.promptIDs.isEmpty)
        #expect(prepared.payload != nil)
        #expect(client.generateCount == 1)
        #expect(client.cancelCount == 0)
        #expect(await arbiter.activity.activeOwner == .chat)

        try await chat.value
        #expect(await arbiter.activity.activeOwner == nil)
    }

    @Test func cancelActiveRequestEndsTheStreamInBand() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response, pieceDelay: .milliseconds(60))
        let harness = try await BridgeHarness.start(client: client)
        let streaming = Task {
            try await streamCompletion(Self.body(#","stream":true"#), port: harness.port)
        }
        try await waitUntil { await harness.arbiter.activity.activeOwner != nil }
        try await waitUntil { await MainActor.run { harness.model.canCancelActiveRequest } }
        harness.model.cancelActiveRequest()
        let (status, frames) = try await streaming.value
        #expect(status == 200)
        #expect(frames.last == "[DONE]")
        #expect(frames.contains { $0.contains(#""code":"service_unavailable""#) })
        #expect(finishReasons(frames).isEmpty)
        #expect(client.cancelCount == 1)
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.contains { $0.bridgeNote == .activeRequestCancelled }
            }
        }
        await harness.stop()
    }

    @Test func stopRefusesARequestQueuedInTheCoordinator() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response, pieceDelay: .milliseconds(60))
        let harness = try await BridgeHarness.start(client: client)
        let first = Task {
            try await streamCompletion(Self.body(#","stream":true"#), port: harness.port)
        }
        try await waitUntil { await harness.arbiter.activity.activeOwner != nil }
        let second = Task { try await completion(Self.body(), port: harness.port) }
        // The second request is rendered and parked in the coordinator's
        // queue behind the first; its lifecycle row is the last observable
        // step before the wait.
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.filter { entry in
                    if case .requestStarted = entry.serverEvent?.kind { return true }
                    return false
                }.count == 2
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        #expect(client.generateCount == 1)

        await harness.stop()
        #expect(harness.model.runState == .stopped)
        let reply = try await second.value
        #expect(reply.status == 503)
        #expect(errorCode(reply) == "service_unavailable")
        #expect(errorMessage(reply) == "the server is stopping")
        // Nothing started a fresh generation during the stop.
        #expect(client.generateCount == 1)
        #expect(client.cancelCount == 1)
        let (status, frames) = try await first.value
        #expect(status == 200)
        #expect(frames.last == "[DONE]")
        #expect(await harness.arbiter.activity.activeOwner == nil)
    }

    /// `beginStopping()` lands while a request is parked ahead of its lease
    /// (inside `acquire`, holding nothing yet). A request that has not
    /// reached the arbiter's queue is out of `cancelAllRequests()`'s reach,
    /// so the check that counts is the one after admission: once the chat
    /// releases, the request is admitted, hands its ticket straight back,
    /// and answers 503 without a generation ever starting.
    @Test func stopRefusesARequestAdmittedAfterStoppingBegan() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response)
        let arbiter = AppInferenceArbiter(client: client)
        await arbiter.publishLoadedSession(BridgeFixtures.session())
        let sink = APIServerLogSink(flushDelay: .milliseconds(1))
        let backend = ArbiterInferenceBackend(arbiter: arbiter, log: sink,
                                              policy: .rejectOnMismatch,
                                              measurePrompt: BridgeFixtures.wordCountMeasurer)
        let chat = try await arbiter.acquire(.chat, timeout: nil)
        let prepared = try await backend.prepare(try BridgeFixtures.validated(Self.body()))
        let request = Task { try await backend.generate(prepared) { _ in } }
        try await waitUntil { await arbiter.activity.queuedHTTPCount == 1 }
        #expect(await backend.liveRequestCount == 1)

        await backend.beginStopping()
        await arbiter.release(chat)

        await #expect(throws: ServerRequestError.unavailable("the server is stopping")) {
            _ = try await request.value
        }
        #expect(client.generateCount == 0)
        #expect(client.cancelCount == 0)
        #expect(await backend.liveRequestCount == 0)
        #expect(!(await backend.hasActiveRequest))
        try await waitUntil { await arbiter.activity.activeOwner == nil }
        #expect(await arbiter.activity.queuedHTTPCount == 0)
        #expect(!sink.snapshot.entries.contains { $0.bridgeNote == .activeRequestCancelled })
    }

    @Test func clientDisconnectIsLoggedAsTheCause() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response, pieceDelay: .milliseconds(60))
        let harness = try await BridgeHarness.start(client: client)
        let sawContent = ContentFlag()
        let url = harness.completionURL
        let consumer = Task {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(Self.body(#","stream":true"#).utf8)
            let (bytes, _) = try await URLSession.shared.bytes(for: request)
            for try await line in bytes.lines where line.contains(#""content""#) {
                sawContent.set()
            }
        }
        try await waitUntil { sawContent.isSet }
        // Cancelling the URLSession task closes the socket mid-stream.
        consumer.cancel()
        _ = try? await consumer.value

        let disconnected: @Sendable (APIServerLogEntry) -> String? = { entry in
            if case .requestFailed(let status, let streaming, let detail) = entry.serverEvent?.kind,
               status == 503, streaming {
                return detail
            }
            return nil
        }
        try await waitUntil {
            await MainActor.run { harness.model.logEntries.contains { disconnected($0) != nil } }
        }
        let detail = harness.model.logEntries.compactMap(disconnected).first
        #expect(detail == "service_unavailable: the client disconnected before the answer completed")
        // The wire cancel followed the disconnect (the backend's cancellation
        // handler and the dropped consumer stream each issue one; the second
        // reaches a session that is already cancelling) and the lease is
        // released.
        try await waitUntil { client.cancelCount >= 1 }
        try await waitUntil { await harness.arbiter.activity.activeOwner == nil }
        #expect(client.generateCount == 1)
        await harness.stop()
    }

    @Test func cancelActiveRequestWithNothingActiveRecordsNoRow() async throws {
        let harness = try await BridgeHarness.start()
        #expect(!harness.model.canCancelActiveRequest)
        harness.model.cancelActiveRequest()
        try await Task.sleep(for: .milliseconds(60))
        #expect(!harness.model.logSink.snapshot.entries.contains {
            $0.bridgeNote == .activeRequestCancelled
        })
        await harness.stop()
    }

    @Test func aRejectedRequestLeavesNoSubstitutionRow() async throws {
        let harness = try await BridgeHarness.start(
            session: BridgeFixtures.session(maxContext: 6, forceLogitsHead: false),
            policy: .pinToSession,
            busyTimeout: .milliseconds(150))
        let isSubstitution: @Sendable (APIServerLogEntry) -> Bool = { entry in
            if case .substitution = entry.bridgeNote { return true }
            return false
        }
        // Pinned temperature, but the conversation does not fit: 400, and no
        // row claims a rewritten request was served.
        let rejected = try await completion(Self.body(#","temperature":0.9"#, messages: #"""
        [{"role":"user","content":"one two three four five"},
         {"role":"assistant","content":"six"},
         {"role":"user","content":"seven eight"}]
        """#), port: harness.port)
        #expect(rejected.status == 400)
        #expect(errorCode(rejected) == "context_length_exceeded")
        try await Task.sleep(for: .milliseconds(60))
        #expect(!harness.model.logSink.snapshot.entries.contains(where: isSubstitution))
        #expect(harness.client.generateCount == 0)

        // Pinned temperature, but the chat holds the session past the busy
        // timeout: 429, and again no row for a request that never generated.
        let chat = try await harness.arbiter.acquire(.chat, timeout: nil)
        let busy = try await completion(Self.body(#","temperature":0.9"#), port: harness.port)
        #expect(busy.status == 429)
        #expect(errorCode(busy) == "queue_full")
        await harness.arbiter.release(chat)
        try await Task.sleep(for: .milliseconds(60))
        #expect(!harness.model.logSink.snapshot.entries.contains(where: isSubstitution))
        #expect(harness.client.generateCount == 0)

        // A served request still gets its row.
        let served = try await completion(Self.body(#","temperature":0.9"#), port: harness.port)
        #expect(served.status == 200)
        try await waitUntil {
            await MainActor.run { harness.model.logEntries.contains(where: isSubstitution) }
        }
        await harness.stop()
    }

    @Test func watchdogTeardownAnswers503AndDropsTheSession() async throws {
        // One prefill event, then a gap longer than the budget.
        let client = BridgeFakeInferenceClient(response: Self.response,
                                               pieceDelay: .milliseconds(120),
                                               prefillSteps: 1)
        let harness = try await BridgeHarness.start(client: client, watchdog: .milliseconds(40))
        let reply = try await completion(Self.body(), port: harness.port)
        #expect(reply.status == 503)
        #expect(errorCode(reply) == "service_unavailable")
        #expect(errorMessage(reply) == AppInferenceArbiterError.transportLostMessage)
        #expect(client.cancelCount == 1)
        try await waitUntil {
            await MainActor.run {
                harness.model.inferenceActivity.loadedSession == nil
                    && harness.model.logEntries.contains { $0.bridgeNote == .modelUnloaded }
            }
        }
        #expect(harness.model.runState.isListening)
        // Until a reload publishes a session, completions are refused.
        let refused = try await completion(Self.body(), port: harness.port)
        #expect(refused.status == 503)
        #expect(errorMessage(refused) == "no model is loaded")
        #expect(client.generateCount == 1)
        await harness.stop()
    }

    @Test func stopDrainsAnInFlightRequestBeforeShuttingDown() async throws {
        let client = BridgeFakeInferenceClient(response: Self.response, pieceDelay: .milliseconds(60))
        let harness = try await BridgeHarness.start(client: client)
        let streaming = Task {
            try await streamCompletion(Self.body(#","stream":true"#), port: harness.port)
        }
        try await waitUntil { await harness.arbiter.activity.activeOwner != nil }
        let stopStarted = ContinuousClock.now
        await harness.stop()
        #expect(ContinuousClock.now - stopStarted < .seconds(4))
        #expect(harness.model.runState == .stopped)
        #expect(client.cancelCount == 1)
        #expect(await harness.arbiter.activity.activeOwner == nil)
        let (status, frames) = try await streaming.value
        #expect(status == 200)
        #expect(frames.last == "[DONE]")
    }
}

private final class ContentFlag: Sendable {
    private let state = Mutex(false)
    var isSet: Bool { state.withLock { $0 } }
    func set() { state.withLock { $0 = true } }
}
