import Foundation
import Synchronization
import MferenceDecodeProtocol
@testable import MferenceAppCore

/// Stands in for `MferenceDecodeService` on the far side of the pipe. The
/// client under test writes command frames into `commands`; the test answers
/// by writing event frames into `events`. No process is spawned.
final class FakeDecodeService: @unchecked Sendable {
    let loadedDirectory: URL
    private let commands = Pipe()
    private let events = Pipe()
    private let eventWrites = Mutex(())

    init(loadedDirectory: URL = FileManager.default.temporaryDirectory) {
        self.loadedDirectory = loadedDirectory
    }

    deinit {
        try? events.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    func makeClient() -> DecodeServiceInferenceClient {
        DecodeServiceInferenceClient(
            transportInput: commands.fileHandleForWriting,
            transportOutput: events.fileHandleForReading,
            loadedDirectory: loadedDirectory)
    }

    func makeRequest(prompt: String = "hello") -> AppGenerationRequest {
        AppGenerationRequest(modelDirectory: loadedDirectory, prompt: prompt)
    }

    /// Reads the next command frame on a background thread and fails after
    /// `timeout` so a missing frame surfaces as a test failure, not a hang.
    func nextCommand(timeout: Duration = .seconds(5)) async throws -> DecodeServiceCommand {
        let handle = commands.fileHandleForReading
        let box = Mutex<Result<DecodeServiceCommand, Error>?>(nil)
        Thread {
            let result = Result { try DecodeFrameCodec.read(DecodeServiceCommand.self, from: handle) }
            box.withLock { $0 = result }
        }.start()
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if let result = box.withLock({ $0 }) { return try result.get() }
            try await Task.sleep(for: .milliseconds(2))
        }
        throw FakeDecodeServiceError.timedOutWaitingForCommand
    }

    func emit(_ event: DecodeServiceEvent) throws {
        try eventWrites.withLock { _ in
            try events.fileHandleForWriting.write(
                contentsOf: DecodeFrameCodec.encode(event))
        }
    }

    func emitSnapshot(generationID: UUID, sequence: UInt64, text: String) throws {
        try emit(DecodeServiceEvent(
            kind: .snapshot, generationID: generationID, sequence: sequence,
            textDelta: text, tokenCount: Int(sequence)))
    }

    func emitTerminal(_ kind: DecodeServiceEventKind, generationID: UUID,
                      tokenCount: Int) throws {
        let stopReason: String = switch kind {
        case .cancelled: "cancelled"
        case .failed: "failed"
        default: "eos"
        }
        try emit(DecodeServiceEvent(
            kind: kind, generationID: generationID, tokenCount: tokenCount,
            stopReason: stopReason))
    }

    /// Closes the event pipe so a reader blocked on it sees EOF.
    func closeEvents() {
        try? events.fileHandleForWriting.close()
    }
}

enum FakeDecodeServiceError: Error {
    case timedOutWaitingForCommand
}

extension DecodeServiceCommand {
    var generationRequest: DecodeGenerationRequest? {
        if case .generate(let request) = self { return request }
        return nil
    }

    var isCancel: Bool {
        if case .cancel = self { return true }
        return false
    }

    var unloadRequestID: UUID? {
        if case .unload(let id) = self { return id }
        return nil
    }
}

/// Collects `onTextDelta` fragments from a `@Sendable` hook.
final class TextDeltaRecorder: Sendable {
    private let deltas = Mutex<[String]>([])

    func append(_ delta: String) {
        deltas.withLock { $0.append(delta) }
    }

    var snapshot: [String] {
        deltas.withLock { $0 }
    }
}
