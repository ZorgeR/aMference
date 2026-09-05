import Foundation
import Testing
import MferenceDecodeProtocol
@testable import MferenceAppCore

/// Regression net for ⌘. (Stop). Cancellation reaches the decode service
/// through two paths: `AppModel.cancel()` and the stream's `onTermination`
/// when a consumer stops listening early. Narrowing `onTermination` to
/// early termination only (so normal completion no longer posts a global
/// wire cancel) must leave both paths working.
@Suite struct DecodeServiceCancelRegressionTests {
    @Test func explicitCancelReachesTheWireWhileGenerating() async throws {
        let service = FakeDecodeService()
        let client = service.makeClient()
        defer { client.shutdown() }

        let events = client.generate(service.makeRequest())
        var iterator = events.makeAsyncIterator()
        let generation = try #require(try await service.nextCommand().generationRequest)
        try service.emitSnapshot(generationID: generation.generationID, sequence: 1, text: "Hel")
        #expect(try await iterator.next() == .token(AppTokenEvent(
            index: 0, textDelta: "Hel", elapsedDecodeSeconds: 0)))

        client.cancel()

        let next = try await service.nextCommand()
        #expect(next.isCancel, "expected a wire cancel, received \(next)")
        try service.emitTerminal(.cancelled, generationID: generation.generationID, tokenCount: 1)
        let terminal = try await iterator.next()
        guard case .cancelled(let diagnostics) = terminal else {
            Issue.record("expected a cancelled terminal event, received \(String(describing: terminal))")
            return
        }
        #expect(diagnostics.stopReason == .cancelled)
        #expect(try await iterator.next() == nil)
    }

    @Test func normalCompletionSendsNoWireCancel() async throws {
        let service = FakeDecodeService()
        let client = service.makeClient()
        defer { client.shutdown() }

        let events = client.generate(service.makeRequest())
        let generation = try #require(try await service.nextCommand().generationRequest)
        try service.emitSnapshot(generationID: generation.generationID, sequence: 1, text: "Hi")
        try service.emitTerminal(.finished, generationID: generation.generationID, tokenCount: 1)
        var collected: [AppInferenceEvent] = []
        for try await event in events { collected.append(event) }
        guard case .finished = collected.last else {
            Issue.record("expected a finished terminal event, received \(collected)")
            return
        }

        // The next frame the service hears must be the unload the test sends
        // now, not a cancel left behind by the finished stream's teardown.
        let unload = Task { await client.unload() }
        let next = try await service.nextCommand()
        let unloadID = try #require(next.unloadRequestID, "expected unload, received \(next)")
        try service.emit(DecodeServiceEvent(kind: .unloaded, generationID: unloadID))
        await unload.value
    }

    @Test func abandonedStreamStillCancelsOnTheWire() async throws {
        let service = FakeDecodeService()
        let client = service.makeClient()
        defer { client.shutdown() }

        // The service side runs concurrently: the consumer below blocks on
        // the first event, which only exists once the fake has answered.
        let serviceSide = Task {
            let generation = try #require(try await service.nextCommand().generationRequest)
            try service.emitSnapshot(generationID: generation.generationID, sequence: 1, text: "Hel")
            return generation.generationID
        }
        // The stream is a temporary, as in `AppModel.launchGeneration`, so
        // leaving the loop drops its last reference and terminates it.
        for try await event in client.generate(service.makeRequest()) {
            if case .token = event { break }
        }

        let generationID = try await serviceSide.value
        let next = try await service.nextCommand()
        #expect(next.isCancel, "expected a wire cancel after the consumer left, received \(next)")
        try service.emitTerminal(.cancelled, generationID: generationID, tokenCount: 1)
    }

    @MainActor
    @Test func appModelCancelReachesTheClient() async throws {
        let mock = MockInferenceClient(response: "one two three four five",
                                       tokenDelayNanos: 20_000_000)
        mock.prefillSteps = 0
        let client = RecordingInferenceClient(wrapping: mock)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        model.promptText = "stop me"
        model.run()
        for _ in 0..<200 where model.liveTokenCount == 0 {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.liveTokenCount > 0)

        model.cancel()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(!model.isRunning)
        #expect(client.cancelCount == 1)
        #expect(model.error == .cancelled)
    }
}
