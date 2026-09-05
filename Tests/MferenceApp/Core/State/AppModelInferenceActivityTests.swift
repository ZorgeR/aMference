import Foundation
import Testing
@testable import MferenceAppCore

/// `AppModel` over a shared arbiter: chat waits behind an API lease without
/// preempting it, shows the queued phase while it waits, and its lifecycle
/// calls run behind the arbiter's barrier.
@Suite struct AppModelInferenceActivityTests {
    @MainActor
    @Test func chatQueuesBehindAnAPILeaseAndProceedsWhenItIsReleased() async throws {
        let directory = try makeCompleteModelInstall("arbiter-queued")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 1_000_000)
        client.prefillSteps = 0
        let arbiter = AppInferenceArbiter(client: client)
        let model = AppModel(modelDirectory: directory, client: client, arbiter: arbiter)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        let httpID = UUID()
        let ticket = try await arbiter.acquire(.http(httpID), timeout: nil)
        #expect(model.inferenceActivity.activeOwner == nil || model.inferenceActivity.activeOwner == .http(httpID))

        model.promptText = "wait your turn"
        model.run()
        for _ in 0..<200 where model.phase != .queued {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(model.isRunning)
        #expect(model.phase == .queued)
        #expect(model.inferenceActivity.queuedChat)
        #expect(model.inferenceActivity.activeOwner == .http(httpID))
        #expect(model.presentation.label == "Waiting for an API request")
        #expect(model.presentation.showsActivity)
        #expect(model.liveTokenCount == 0)

        await arbiter.release(ticket)
        for _ in 0..<400 where model.isRunning {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(!model.isRunning)
        #expect(model.phase == .idle)
        #expect(model.error == nil)
        #expect(model.outputText.contains("one two three"))
        #expect(model.inferenceActivity == .idle)
    }

    @MainActor
    @Test func stopWhileQueuedDequeuesChatWithoutTouchingTheClient() async throws {
        let mock = MockInferenceClient(response: "unused", tokenDelayNanos: 1_000_000)
        let client = RecordingInferenceClient(wrapping: mock)
        let arbiter = AppInferenceArbiter(client: client)
        let model = readyModel(client: client, arbiter: arbiter)
        let ticket = try await arbiter.acquire(.http(UUID()), timeout: nil)

        model.promptText = "never runs"
        model.run()
        for _ in 0..<200 where model.phase != .queued {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.phase == .queued)

        model.cancel()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        #expect(client.cancelCount == 0)
        #expect(client.generateCount == 0)
        #expect(!model.inferenceActivity.queuedChat)
        await arbiter.release(ticket)
    }

    @MainActor
    @Test func loadPublishesTheSessionAndUnloadClearsIt() async throws {
        let directory = try makeCompleteModelInstall("arbiter-session")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let model = AppModel(modelDirectory: directory, client: client, arbiter: arbiter)
        model.temperature = 0.7
        model.topKEnabled = true
        model.topK = 40
        model.topPEnabled = false

        model.loadModel()
        for _ in 0..<200 where !model.loadState.isReady {
            try await Task.sleep(for: .milliseconds(5))
        }
        for _ in 0..<200 where model.inferenceActivity.loadedSession == nil {
            try await Task.sleep(for: .milliseconds(5))
        }

        let session = try #require(await arbiter.loadedSession)
        #expect(session.modelDirectory == directory.standardizedFileURL)
        #expect(session.maxContextTokens == model.maxContextTokens)
        #expect(session.runtimeOptions == model.runtimeOptions)
        #expect(session.forceLogitsHead)
        #expect(session.temperature == 0.7)
        #expect(session.topK == 40)
        #expect(session.topP == nil)
        #expect(model.inferenceActivity.loadedSession == session)

        model.unloadModel()
        for _ in 0..<200 where model.loadState != .notLoaded {
            try await Task.sleep(for: .milliseconds(5))
        }
        for _ in 0..<200 where model.inferenceActivity.loadedSession != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(await arbiter.loadedSession == nil)
        #expect(model.inferenceActivity.loadedSession == nil)
    }

    @MainActor
    @Test func unloadRevokesAnActiveAPILease() async throws {
        let directory = try makeCompleteModelInstall("arbiter-drain")
        defer { try? FileManager.default.removeItem(at: directory) }
        let lifecycle = MockLifecycleInferenceClient()
        let arbiter = AppInferenceArbiter(client: lifecycle)
        let model = AppModel(modelDirectory: directory, client: lifecycle, arbiter: arbiter)
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        #expect(model.canUnloadModel)

        // An API request holds the session; the model is unloaded under it.
        let httpID = UUID()
        let ticket = try await arbiter.acquire(.http(httpID), timeout: nil)
        model.unloadModel()
        for _ in 0..<200 where model.loadState != .notLoaded {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(model.loadState == .notLoaded)
        #expect(await arbiter.activity.activeOwner == nil)
        // The revoked ticket is inert.
        await arbiter.release(ticket)
        #expect(await arbiter.activity == .idle)
    }

    // MARK: Watchdog teardown

    /// The arbiter's watchdog shut the decode service down under a chat
    /// generation: the run ends with the transport-lost error, and the load
    /// state stops claiming a ready model so the UI offers Load again.
    @MainActor
    @Test func watchdogTeardownUnderAChatGenerationFailsTheLoadState() async throws {
        let directory = try makeCompleteModelInstall("arbiter-watchdog-chat")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(40))
        let model = AppModel(modelDirectory: directory, client: client, arbiter: arbiter)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        await arbiter.publishLoadedSession(Self.session(for: model, directory: directory))
        for _ in 0..<200 where model.inferenceActivity.loadedSession == nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.canRun == false)
        model.promptText = "stall after the first token"
        #expect(model.canRun)

        model.run()
        for _ in 0..<200 where client.generateCount == 0 {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.isRunning)
        client.emitToken("first")
        for _ in 0..<400 where model.isRunning {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(!model.isRunning)
        #expect(model.error == .unknown(AppInferenceArbiterError.transportLostMessage))
        #expect(model.loadState
                == .failed(.modelLoadFailed(AppInferenceArbiterError.transportLostMessage)))
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.liveMemoryBytes == nil)
        #expect(model.canLoadModel)
        #expect(model.presentation.primaryAction == .retryLoad)
        for _ in 0..<200 where model.inferenceActivity.loadedSession != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.inferenceActivity.loadedSession == nil)
        #expect(await arbiter.loadedSession == nil)
        for _ in 0..<200 where client.shutdownCount == 0 {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(client.disconnectCount == 1)
        #expect(client.shutdownCount == 1)
    }

    /// The same teardown under an API request: the chat was idle, so only the
    /// published session tells the app the model is gone.
    @MainActor
    @Test func watchdogTeardownUnderAnAPIRequestFailsTheLoadState() async throws {
        let directory = try makeCompleteModelInstall("arbiter-watchdog-http")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = ScriptedInferenceClient()
        client.cancelEmitsTerminal = false
        let arbiter = AppInferenceArbiter(client: client, watchdog: .milliseconds(40))
        let model = AppModel(modelDirectory: directory, client: client, arbiter: arbiter)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        await arbiter.publishLoadedSession(Self.session(for: model, directory: directory))
        for _ in 0..<200 where model.inferenceActivity.loadedSession == nil {
            try await Task.sleep(for: .milliseconds(5))
        }

        let httpID = UUID()
        let request = AppGenerationRequest(modelDirectory: directory, prompt: "api")
        let http = Task {
            var count = 0
            for try await _ in await arbiter.stream(request, owner: .http(httpID)) { count += 1 }
            return count
        }
        for _ in 0..<200 where client.generateCount == 0 {
            try await Task.sleep(for: .milliseconds(5))
        }
        client.emitToken("first")
        await #expect(throws: AppInferenceArbiterError.transportLost) {
            _ = try await http.value
        }

        for _ in 0..<200 where !model.loadState.isFailed {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.loadState
                == .failed(.modelLoadFailed(AppInferenceArbiterError.transportLostMessage)))
        #expect(model.loadedRuntimeKey == nil)
        #expect(!model.isRunning)
        #expect(model.canLoadModel)
        #expect(model.inferenceActivity.loadedSession == nil)
    }

    /// A ready state that never had a published session (fixtures set it
    /// directly) is not a lost transport: an activity change must not fail it.
    @MainActor
    @Test func activityWithoutASessionNeverFailsAReadyModel() async throws {
        let client = ScriptedInferenceClient()
        let arbiter = AppInferenceArbiter(client: client)
        let model = readyModel(client: client, arbiter: arbiter)
        let ticket = try await arbiter.acquire(.http(UUID()), timeout: nil)
        for _ in 0..<100 where model.inferenceActivity.activeOwner == nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        await arbiter.release(ticket)
        for _ in 0..<100 where model.inferenceActivity.activeOwner != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.loadState.isReady)
    }

    // MARK: Unload ordering

    /// The session is cleared inside the unload barrier: the first admission
    /// after it lifts (what an HTTP `generate` takes first) never sees the
    /// old session, so a request that raced the unload is refused with no
    /// model rather than sent to an unloaded service.
    ///
    /// The probe is a race by construction. Against the wrong ordering (the
    /// session cleared after the barrier lifts) it catches the stale session
    /// only when its admission and its read both land before the late
    /// `publishLoadedSession(nil)` is serviced, which one cycle misses more
    /// often than not. The cycle therefore repeats, reloading in between:
    /// 30 rounds fail the wrong ordering with high probability, while the
    /// right ordering passes every round deterministically.
    @MainActor
    @Test func unloadClearsTheSessionBeforeAdmissionReopens() async throws {
        let directory = try makeCompleteModelInstall("arbiter-unload-order")
        defer { try? FileManager.default.removeItem(at: directory) }
        let lifecycle = MockLifecycleInferenceClient()
        let arbiter = AppInferenceArbiter(client: lifecycle)
        let model = AppModel(modelDirectory: directory, client: lifecycle, arbiter: arbiter)

        for round in 1...30 {
            model.loadModel()
            for _ in 0..<200 where !model.loadState.isReady {
                try await Task.sleep(for: .milliseconds(5))
            }
            for _ in 0..<200 where model.inferenceActivity.loadedSession == nil {
                try await Task.sleep(for: .milliseconds(5))
            }
            #expect(await arbiter.loadedSession != nil, "round \(round): no session after load")

            lifecycle.suspendUnloads = true
            model.unloadModel()
            await lifecycle.waitForUnloadStart(round)
            if round == 1 {
                // Behind the barrier every admission is refused outright.
                await #expect(throws: AppInferenceArbiterError.unavailable("Unloading model")) {
                    _ = try await arbiter.acquire(.http(UUID()), timeout: nil)
                }
            }

            // Spin on admission; the first ticket granted is the first thing
            // through after the barrier lifts.
            let httpID = UUID()
            let probe = Task { () throws -> AppLoadedSession? in
                while true {
                    do {
                        let ticket = try await arbiter.acquire(.http(httpID), timeout: nil)
                        let seen = await arbiter.loadedSession
                        await arbiter.release(ticket)
                        return seen
                    } catch let error as AppInferenceArbiterError {
                        guard case .unavailable = error else { throw error }
                        try await Task.sleep(for: .microseconds(200))
                    }
                }
            }
            lifecycle.releaseUnloads()
            let seen = try await probe.value
            #expect(seen == nil, "round \(round): admitted over a stale session")

            for _ in 0..<200 where model.loadState != .notLoaded {
                try await Task.sleep(for: .milliseconds(5))
            }
            #expect(model.loadState == .notLoaded, "round \(round)")
        }

        for _ in 0..<200 where model.inferenceActivity.loadedSession != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.inferenceActivity.loadedSession == nil)
        #expect(await arbiter.activity == .idle)
    }

    // MARK: Helpers

    @MainActor
    private static func session(for model: AppModel, directory: URL) -> AppLoadedSession {
        AppLoadedSession(modelDirectory: directory,
                         maxContextTokens: model.maxContextTokens,
                         runtimeOptions: model.runtimeOptions,
                         forceLogitsHead: true,
                         temperature: Float(model.temperature),
                         topK: model.topK,
                         topP: Float(model.topP))
    }

    @MainActor
    private func readyModel(client: any AppInferenceClient,
                            arbiter: AppInferenceArbiter) -> AppModel {
        let model = AppModel(client: client, arbiter: arbiter)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        return model
    }
}
