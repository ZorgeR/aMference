import Foundation
import Testing
import MferenceDecodeProtocol

/// Exercises `FrameChannelClaim` on private pipes only; the test process's
/// own fd 1 and fd 2 are never touched.
@Suite struct FrameChannelClaimTests {
    @Test func claimedChannelCarriesFramesWhileStrayStdoutWritesLandOnStderr() throws {
        let frames = Pipe()
        let diagnostics = Pipe()
        let originalStdout = frames.fileHandleForWriting.fileDescriptor

        let channel = try FrameChannelClaim.claim(
            standardOutput: originalStdout,
            standardError: diagnostics.fileHandleForWriting.fileDescriptor)
        #expect(channel.fileDescriptor != originalStdout)

        let event = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            sequence: 1,
            textDelta: "hello",
            tokenCount: 1)
        try channel.write(contentsOf: DecodeFrameCodec.encode(event))
        // What a `print()` in the runtime does after the claim: it writes to
        // the original stdout descriptor number, not to the private channel.
        try frames.fileHandleForWriting.write(
            contentsOf: Data("stray print() output\n".utf8))

        try channel.close()
        try frames.fileHandleForWriting.close()
        try diagnostics.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: frames.fileHandleForReading)
        #expect(decoded.kind == .snapshot)
        #expect(decoded.sequence == 1)
        #expect(decoded.textDelta == "hello")
        #expect(decoded.tokenCount == 1)
        #expect(frames.fileHandleForReading.readDataToEndOfFile().isEmpty)

        let stray = String(
            decoding: diagnostics.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self)
        #expect(stray == "stray print() output\n")
    }

    @Test func claimReportsErrnoWhenTheOutputDescriptorIsInvalid() {
        #expect(throws: FrameChannelClaimError.duplicateFailed(errno: EBADF)) {
            _ = try FrameChannelClaim.claim(standardOutput: -1, standardError: -1)
        }
    }

    @Test func failedRedirectLeavesTheOriginalDescriptorIntact() throws {
        let frames = Pipe()
        let originalStdout = frames.fileHandleForWriting.fileDescriptor

        #expect(throws: FrameChannelClaimError.redirectFailed(errno: EBADF)) {
            _ = try FrameChannelClaim.claim(
                standardOutput: originalStdout, standardError: -1)
        }

        try frames.fileHandleForWriting.write(contentsOf: Data("still open\n".utf8))
        try frames.fileHandleForWriting.close()
        let received = String(
            decoding: frames.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self)
        #expect(received == "still open\n")
    }
}
