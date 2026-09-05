import Foundation
import MferenceServerCore

/// The bind picker's rows. `ServerBindMode` is deliberately not
/// `CaseIterable` in the server module, and deliberately has no wildcard or
/// LAN case; the bridge only names the two that exist.
public struct APIServerBindOption: Identifiable, Equatable, Sendable {
    public let mode: ServerBindMode
    public let label: String
    public let help: String
    /// True when the endpoint is reachable from another machine. There is no
    /// authentication, so the pane must warn while such an option is selected.
    public let isExposedBeyondThisMac: Bool

    public var id: String { mode.rawValue }

    public static let loopback = APIServerBindOption(
        mode: .loopback,
        label: "This Mac only",
        help: "Listens on 127.0.0.1. Only programs on this Mac can connect.",
        isExposedBeyondThisMac: false)

    public static let tailnet = APIServerBindOption(
        mode: .tailnet,
        label: "Tailnet",
        help: "Listens on this Mac's Tailscale IPv4 address only. Requires the tailscale CLI on PATH and a connected Tailscale.",
        isExposedBeyondThisMac: true)

    public static let all: [APIServerBindOption] = [.loopback, .tailnet]

    public static func option(for mode: ServerBindMode) -> APIServerBindOption {
        switch mode {
        case .loopback: .loopback
        case .tailnet: .tailnet
        }
    }

    /// Shown while an exposed option is selected.
    public static let exposureWarning =
        "Anyone on your tailnet can use this model. There is no authentication."
}
