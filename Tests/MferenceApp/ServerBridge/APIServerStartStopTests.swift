import Foundation
import Testing
import MferenceAppCore
import MferenceServerBridge
import MferenceServerCore

@Suite("API server start/stop", .serialized)
@MainActor
struct APIServerStartStopTests {
    @Test func eachStartBuildsAFreshServerAndReadsThePortFromTheChannel() async throws {
        let harness = try await BridgeHarness.start()
        let first = harness.port
        #expect(first != 0)
        #expect(harness.model.port == 0)
        #expect(harness.model.baseURL == "http://127.0.0.1:\(first)/v1")
        #expect(harness.model.servedIdentity == BridgeFixtures.identity)
        #expect(try await send("GET", "/health", port: first, contentType: nil).status == 200)

        await harness.stop()
        #expect(harness.model.runState == .stopped)
        #expect(harness.model.servedIdentity == nil)
        #expect(harness.model.baseURL == nil)

        // A reused `MferenceHTTPServer` cannot start again: its group is gone.
        harness.model.start()
        await harness.model.awaitIdle()
        guard case .listening(let host, let second) = harness.model.runState else {
            Issue.record("second start failed: \(harness.model.runState)")
            return
        }
        #expect(host == "127.0.0.1")
        #expect(second != 0)
        #expect(harness.model.startCount == 2)
        #expect(try await send("GET", "/health", port: second, contentType: nil).status == 200)
        await harness.stop()
        #expect(harness.model.runState == .stopped)
    }

    @Test func startAndStopAreSerialized() async throws {
        let harness = try await BridgeHarness.start()
        harness.model.stop()
        harness.model.start()
        harness.model.stop()
        harness.model.start()
        harness.model.stop()
        await harness.model.awaitIdle()
        #expect(harness.model.runState == .stopped)
        #expect(harness.model.startCount == 3)

        try await waitUntil { await MainActor.run { harness.model.logEntries.count >= 6 } }
        let lifecycle = harness.model.logEntries.compactMap { entry -> String? in
            switch entry.bridgeNote {
            case .listening: "listening"
            case .stopped: "stopped"
            default: nil
            }
        }
        #expect(lifecycle == ["listening", "stopped", "listening", "stopped", "listening", "stopped"])
    }

    @Test func startWithoutALoadedSessionFailsInEnglish() async throws {
        let arbiter = AppInferenceArbiter(client: BridgeFakeInferenceClient())
        let model = APIServerModel(arbiter: arbiter, environment: BridgeFixtures.environment(),
                                   logFlushDelay: .milliseconds(10))
        #expect(!model.canStart)
        model.start()
        await model.awaitIdle()
        #expect(model.runState == .failed("Load a model before starting the server."))
        #expect(model.presentation.label == "Could not start")
        #expect(model.presentation.primaryAction == nil)
        #expect(model.logSink.snapshot.entries.last?.bridgeNote
                == .startFailed("Load a model before starting the server."))
    }

    @Test func bindFailureIsRenderedAsPlainEnglish() async throws {
        let first = try await BridgeHarness.start()
        let arbiter = AppInferenceArbiter(client: BridgeFakeInferenceClient())
        await arbiter.publishLoadedSession(BridgeFixtures.session())
        let second = APIServerModel(arbiter: arbiter, environment: BridgeFixtures.environment(),
                                    logFlushDelay: .milliseconds(10))
        second.port = first.port
        second.start()
        await second.awaitIdle()
        guard case .failed(let message) = second.runState else {
            Issue.record("expected a bind failure, got \(second.runState)")
            await first.stop()
            return
        }
        #expect(message == "Port \(first.port) is already in use on 127.0.0.1. Choose another port, or stop the program using it.")
        #expect(second.canStart)
        await first.stop()

        // The port is free again; the same model recovers.
        second.start()
        await second.awaitIdle()
        #expect(second.runState.isListening)
        second.stop()
        await second.awaitIdle()
    }

    @Test func identityAndHostFailuresSurfaceVerbatim() async throws {
        let arbiter = AppInferenceArbiter(client: BridgeFakeInferenceClient())
        await arbiter.publishLoadedSession(BridgeFixtures.session())
        let identityFailure = APIServerModel(
            arbiter: arbiter,
            environment: BridgeFixtures.environment(resolveIdentity: { _ in
                throw ServerArgumentError.invalid("manifest.json is unreadable")
            }),
            logFlushDelay: .milliseconds(10))
        identityFailure.start()
        await identityFailure.awaitIdle()
        #expect(identityFailure.runState == .failed("manifest.json is unreadable"))

        let hostFailure = APIServerModel(
            arbiter: arbiter,
            environment: APIServerEnvironment(
                resolveIdentity: { _ in BridgeFixtures.identity },
                measurePrompt: BridgeFixtures.wordCountMeasurer,
                resolveHost: { _ in
                    throw ServerArgumentError.invalid(
                        "tailscale reported 2 IPv4 addresses; refusing to guess which to bind")
                }),
            logFlushDelay: .milliseconds(10))
        hostFailure.bindMode = .tailnet
        hostFailure.start()
        await hostFailure.awaitIdle()
        #expect(hostFailure.runState
                == .failed("tailscale reported 2 IPv4 addresses; refusing to guess which to bind"))
    }

    @Test func aDifferentModelDirectoryStopsTheServer() async throws {
        let harness = try await BridgeHarness.start()
        let other = BridgeFixtures.session(directory: URL(fileURLWithPath: "/tmp/other.gturbo"))
        await harness.arbiter.publishLoadedSession(other)
        try await waitUntil { await MainActor.run { harness.model.runState == .stopped } }
        await harness.model.awaitIdle()
        let notes = harness.model.logSink.snapshot.entries.compactMap(\.bridgeNote)
        #expect(notes.contains(.modelChanged("other.gturbo")))
        if case .stopped(let reason) = try #require(notes.last) {
            #expect(reason == "the model changed to other.gturbo")
        } else {
            Issue.record("expected a stopped row, got \(String(describing: notes.last))")
        }
    }

    @Test func unloadingKeepsTheServerListening() async throws {
        let harness = try await BridgeHarness.start()
        await harness.arbiter.publishLoadedSession(nil)
        try await waitUntil { await MainActor.run { harness.model.inferenceActivity.loadedSession == nil } }
        #expect(harness.model.runState.isListening)
        #expect(harness.model.presentation.label == "Listening · no model loaded")
        #expect(try await send("GET", "/health", port: harness.port, contentType: nil).status == 200)
        #expect(harness.model.logSink.snapshot.entries.compactMap(\.bridgeNote).contains(.modelUnloaded))

        // Reloading the same directory changes nothing.
        await harness.arbiter.publishLoadedSession(BridgeFixtures.session())
        try await waitUntil { await MainActor.run { harness.model.inferenceActivity.loadedSession != nil } }
        #expect(harness.model.runState.isListening)
        #expect(harness.model.canStop)
        await harness.stop()
    }

    @Test func logDrainsToObservableStateAndClears() async throws {
        let harness = try await BridgeHarness.start()
        _ = try await send("GET", "/health", port: harness.port, contentType: nil)
        try await waitUntil {
            await MainActor.run {
                harness.model.logEntries.contains { $0.serverEvent?.path == "/health" }
            }
        }
        #expect(harness.model.logEntries.first?.bridgeNote != nil)
        harness.model.clearLog()
        #expect(harness.model.logEntries.isEmpty)
        #expect(harness.model.droppedBefore == 0)
        await harness.stop()
    }
}
