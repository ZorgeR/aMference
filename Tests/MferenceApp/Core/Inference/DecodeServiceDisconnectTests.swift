import Foundation
import Testing
import MferenceDecodeProtocol
@testable import MferenceAppCore

/// `disconnect()` is the half of a teardown the arbiter's watchdog runs on
/// its actor before it releases the lease: from then on the client answers
/// as if nothing were loaded, whatever a later `shutdown()` still has to
/// reap.
@Suite struct DecodeServiceDisconnectTests {
    @Test func disconnectLeavesNoHandlesForTheNextGeneration() async throws {
        let service = FakeDecodeService()
        let client = service.makeClient()
        defer { client.shutdown() }

        // A generation is on the pipe, its reader blocked on the next frame.
        let first = client.generate(service.makeRequest())
        var firstIterator = first.makeAsyncIterator()
        _ = try #require(try await service.nextCommand().generationRequest)

        client.disconnect()

        // The next owner finds no connection and writes nothing.
        var refusal: AppInferenceError?
        do {
            for try await _ in client.generate(service.makeRequest()) {}
        } catch let error as AppInferenceError {
            refusal = error
        }
        #expect(refusal == .modelNotLoaded)
        #expect(client.currentInferenceMemoryBytes == nil)
        // The command pipe is closed: the service reads EOF, never a second
        // `generate` frame. `cancel()` has nothing to write to either.
        client.cancel()
        await #expect(throws: DecodeFrameError.unexpectedEOF) {
            _ = try await service.nextCommand(timeout: .seconds(2))
        }

        // The first stream ends with an error, never a frame. (Closing the
        // service side stands in for the reaped process's EOF where the
        // closed read handle alone has not woken the reader.)
        service.closeEvents()
        await #expect(throws: (any Error).self) {
            _ = try await firstIterator.next()
        }

        // Both halves are idempotent, in either order.
        client.disconnect()
        client.shutdown()
        client.shutdown()
        #expect(client.currentInferenceMemoryBytes == nil)
    }
}
