import Foundation
import Mference

/// The API identity of the model the app has loaded: the family-derived
/// model id `/v1/models` lists and the chat dialect the request validator
/// applies. Resolved from the install's manifest and tokenizer sidecar only;
/// it never instantiates `ServerModelSession`, whose only entry point loads
/// a second copy of the weights.
public struct APIServerModelIdentity: Equatable, Sendable {
    public let family: ModelFamily
    public let modelID: String
    public let chatDialect: ChatDialect

    public init(family: ModelFamily, modelID: String, chatDialect: ChatDialect) {
        self.family = family
        self.modelID = modelID
        self.chatDialect = chatDialect
    }

    /// Family → API model id, kept identical to
    /// `ServerModelSession.defaultModelID` so a client configured against the
    /// standalone server works against the app without edits.
    public static func modelID(for family: ModelFamily) -> String {
        switch family {
        case .gemma4: return "gemma-4-26b-a4b-it"
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen38: return "qwen3.8-27b-4bit"
        case .deepseekV4Flash: return "deepseek-v4-flash-2bit-dq"
        case .inklingSmall: return "inkling-small-4bit"
        case .maple: return "maple-preview-2bit-mlx"
        // Unreachable while the capability gate stands: `peekFamily` refuses
        // this family by name.
        case .qwen38flashnext: return "qwen3.8-flash-next-int4g64"
        }
    }

    /// Reads `manifest.json -> arch.family` and the tokenizer's dialect. The
    /// tokenizer is cached per source by a process-wide actor, so after the
    /// app's own load this is pure CPU.
    public static func resolve(modelDirectory: URL) async throws -> APIServerModelIdentity {
        let family = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelDirectory)
        return APIServerModelIdentity(family: family,
                                      modelID: modelID(for: family),
                                      chatDialect: tokenizer.dialect)
    }
}
