import Foundation
import Darwin

/// Why claiming the frame channel failed. The payload is the `errno` the
/// failing syscall left behind.
public enum FrameChannelClaimError: Error, Equatable, Sendable, CustomStringConvertible {
    /// `dup(standardOutput)` failed; nothing was changed.
    case duplicateFailed(errno: Int32)
    /// `dup2(standardError, standardOutput)` failed; the private duplicate was
    /// closed again and the original descriptor is untouched.
    case redirectFailed(errno: Int32)

    public var description: String {
        switch self {
        case .duplicateFailed(let errno):
            return "could not duplicate the frame channel descriptor: "
                + "\(String(cString: strerror(errno))) (errno \(errno))"
        case .redirectFailed(let errno):
            return "could not redirect stdout onto stderr: "
                + "\(String(cString: strerror(errno))) (errno \(errno))"
        }
    }
}

/// Detaches the decode service's frame channel from the process's stdout.
///
/// The Mac app reads length-prefixed `DecodeServiceEvent` frames from the
/// child's stdout and throws on any sequence gap instead of resyncing, so a
/// single stray byte on fd 1 — a `print()` somewhere in the runtime, a Swift
/// runtime message, a Metal validation-layer line — corrupts the whole
/// session. `claim` makes that impossible by construction: it duplicates the
/// original stdout descriptor into a fresh private descriptor that only the
/// frame codec ever writes to, then points the original fd number at stderr.
/// Anything that writes to "stdout" afterwards lands in the stderr stream,
/// where the parent already expects diagnostics.
///
/// The descriptors are parameters so the behaviour can be tested on pipes
/// without touching the test process's own fd 1 and fd 2.
public enum FrameChannelClaim {
    /// Duplicates `standardOutput` into a private descriptor and returns a
    /// handle over it (closed on dealloc), then makes `standardOutput` an
    /// alias of `standardError`. The private descriptor is marked
    /// close-on-exec so a spawned child cannot inherit the frame channel.
    public static func claim(standardOutput: Int32 = STDOUT_FILENO,
                             standardError: Int32 = STDERR_FILENO) throws -> FileHandle {
        let channel = retryingOnInterrupt { dup(standardOutput) }
        guard channel >= 0 else {
            throw FrameChannelClaimError.duplicateFailed(errno: errno)
        }
        _ = fcntl(channel, F_SETFD, FD_CLOEXEC)
        guard retryingOnInterrupt({ dup2(standardError, standardOutput) }) >= 0 else {
            let failure = errno
            _ = close(channel)
            throw FrameChannelClaimError.redirectFailed(errno: failure)
        }
        return FileHandle(fileDescriptor: channel, closeOnDealloc: true)
    }

    private static func retryingOnInterrupt(_ call: () -> Int32) -> Int32 {
        while true {
            let result = call()
            if result >= 0 || errno != EINTR { return result }
        }
    }
}
