import Foundation
import Testing
import MferenceDecodeProtocol
@testable import MferenceAppCore

@Suite struct DecodeServiceDeltaStreamingTests {
    @Test func everySnapshotReachesTheDeltaHookWhileTokenEventsStayThrottled() async throws {
        let service = FakeDecodeService()
        let client = service.makeClient()
        defer { client.shutdown() }
        let recorder = TextDeltaRecorder()
        let pieces = ["Hello", ",", " world", "!", " done"]

        let events = client.generate(service.makeRequest()) { recorder.append($0) }
        let generation = try #require(try await service.nextCommand().generationRequest)
        for (offset, piece) in pieces.enumerated() {
            try service.emitSnapshot(generationID: generation.generationID,
                                     sequence: UInt64(offset + 1), text: piece)
        }
        try service.emitTerminal(.finished, generationID: generation.generationID,
                                 tokenCount: pieces.count)

        var tokenEvents: [AppTokenEvent] = []
        var finished = false
        for try await event in events {
            switch event {
            case .token(let token): tokenEvents.append(token)
            case .finished: finished = true
            default: break
            }
        }

        #expect(finished)
        #expect(recorder.snapshot == pieces)
        #expect(client.generationTranscriptMailbox.completeText == pieces.joined())
        // The event path is unchanged: only the first visible chunk carries
        // text; later `.token` events are metric ticks at most every 0.5 s.
        #expect(!tokenEvents.isEmpty)
        #expect(tokenEvents.count <= pieces.count)
        #expect(tokenEvents.filter { !$0.textDelta.isEmpty }.count == 1)
        #expect(tokenEvents.map(\.textDelta).joined() == "Hello")
    }

    @Test func plainGenerateKeepsWorkingWithoutAHook() async throws {
        let service = FakeDecodeService()
        let client = service.makeClient()
        defer { client.shutdown() }

        let events = client.generate(service.makeRequest())
        let generation = try #require(try await service.nextCommand().generationRequest)
        try service.emitSnapshot(generationID: generation.generationID, sequence: 1, text: "A")
        try service.emitTerminal(.finished, generationID: generation.generationID, tokenCount: 1)
        var sawText = false
        for try await event in events {
            if case .token(let token) = event, token.textDelta == "A" { sawText = true }
        }
        #expect(sawText)
        #expect(client.generationTranscriptMailbox.completeText == "A")
    }
}
