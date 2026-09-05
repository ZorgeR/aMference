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
