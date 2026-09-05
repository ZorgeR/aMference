import Foundation

/// Writes one human-readable diagnostic line to stderr.
///
/// The runtime must never write to stdout. Inside `MferenceDecodeService`
/// fd 1 is the length-prefixed frame channel to the Mac app, and the
/// app-side reader treats any stray byte as a corrupted stream rather than
/// resyncing; the CLI and server likewise reserve stdout for their own
/// output. Swift's `print` is therefore off-limits under `Sources/Mference/` —
/// route every message through this helper instead. A failed write is
/// swallowed on purpose: a diagnostic must never take the runtime down.
func runtimeDiagnostic(_ message: String) {
    try? FileHandle.standardError.write(contentsOf: Data((message + "\n").utf8))
}
