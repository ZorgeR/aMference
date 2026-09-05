import Foundation
import Mference
import MferenceAppCore
import MferenceServerCore

/// What the in-app server does when a request's sampling mode (greedy versus
/// sampled) differs from the one the app loaded the model for. The decode
/// service refuses a request whose mode differs from the loaded session
/// (`RealInferenceClient` compares the whole load key), so the bridge has to
/// decide before the request reaches the wire.
public enum APIServerSamplingPolicy: String, CaseIterable, Identifiable, Sendable {
    /// Answer `400 sampling_mode_mismatch` naming the loaded mode. The default:
    /// the client learns what to change instead of getting a silently
    /// different answer.
    case rejectOnMismatch
    /// Rewrite temperature, top-p, top-k and repetition penalty to the loaded
    /// session's values and log one substitution row per changed parameter.
    case pinToSession

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .rejectOnMismatch: "Reject with 400"
        case .pinToSession: "Pin to the loaded session"
        }
    }

    public var help: String {
        switch self {
        case .rejectOnMismatch:
            "Requests whose sampling mode differs from the loaded model are refused with a 400 that names the loaded mode."
        case .pinToSession:
            "Sampling parameters are rewritten to the loaded model's values; every substitution is logged."
        }
    }
}

/// One parameter the bridge rewrote under `.pinToSession`.
public struct APIServerSubstitution: Equatable, Sendable {
    public let parameter: String
    public let requested: String
    public let applied: String
    public let reason: String

    public init(parameter: String, requested: String, applied: String, reason: String) {
        self.parameter = parameter
        self.requested = requested
        self.applied = applied
        self.reason = reason
    }
}

/// Pure translation from a validated OpenAI request to the app's generation
/// request. Every rejection is a `ServerRequestError`, so it reaches the
/// client with a real status and code; anything else would collapse to the
/// generic 500 envelope. Runs with no IPC and no tokenizer.
public enum ArbiterRequestTranslation {
    public static func translate(_ request: ValidatedChatRequest,
                                 session: AppLoadedSession,
                                 policy: APIServerSamplingPolicy)
        throws -> (request: AppGenerationRequest, substitutions: [APIServerSubstitution]) {
        // Tools are token-ID-driven inside the decode service, which runs with
        // an empty tool set and throws on any call. An empty `tools: []` is
        // accepted because many clients send it unconditionally.
        guard request.tools.isEmpty else {
            throw invalid("tools are not supported by the in-app server; use the standalone MferenceServer",
                          "tools", "tools_not_supported")
        }
        let config = request.generationConfig
        if config.seed != nil {
            throw invalid("seed is not supported by the in-app server", "seed", "unsupported_parameter")
        }
        guard config.repetitionPenalty >= 1 else {
            throw invalid("repetition_penalty must be at least 1 on the in-app server",
                          "repetition_penalty", "invalid_value")
        }

        let messages = try translateMessages(request.messages)

        var temperature = config.temperature
        var topK = config.topK
        var topP = config.topP
        var repetitionPenalty = config.repetitionPenalty
        var substitutions: [APIServerSubstitution] = []

        // The service's load key carries only the mode, not the values, so a
        // request that agrees on the mode keeps its own temperature/top-k/top-p.
        let requestSampled = !(temperature == 0 && repetitionPenalty == 1)
        if requestSampled != session.forceLogitsHead {
            switch policy {
            case .rejectOnMismatch:
                throw invalid(mismatchMessage(session: session),
                              "temperature", "sampling_mode_mismatch")
            case .pinToSession:
                let reason = session.forceLogitsHead
                    ? "the loaded session samples; greedy requests cannot be served without a reload"
                    : "the loaded session decodes greedily; sampled requests cannot be served without a reload"
                func pin<T: Equatable>(_ parameter: String, _ value: inout T, to applied: T,
                                       describe: (T) -> String) {
                    guard value != applied else { return }
                    substitutions.append(APIServerSubstitution(
                        parameter: parameter, requested: describe(value),
                        applied: describe(applied), reason: reason))
                    value = applied
                }
                if session.forceLogitsHead {
                    pin("temperature", &temperature, to: session.temperature, describe: format)
                    pin("top_k", &topK, to: session.topK, describe: format)
                    pin("top_p", &topP, to: session.topP, describe: format)
                } else {
                    pin("temperature", &temperature, to: 0, describe: format)
                    pin("repetition_penalty", &repetitionPenalty, to: 1, describe: format)
                }
            }
        }
        // Top-p below 1 needs top-k on this runtime; the validator defaults
        // top-k, so this only guards a hand-built request.
        if temperature > 0, let topP, topP < 1, topK == nil {
            throw invalid("top_p below 1 requires top_k", "top_p", "invalid_value")
        }

        let translated = AppGenerationRequest(
            modelDirectory: session.modelDirectory,
            messages: messages,
            maxNewTokens: request.maximumCompletionTokens,
            maxContextTokens: session.maxContextTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            runtimeOptions: session.runtimeOptions)
        return (translated, substitutions)
    }

    /// The wire carries system/user/assistant only. `developer` folds into
    /// `system` for every dialect; every system message merges into one
    /// leading message; tool traffic is refused with a code of its own.
    static func translateMessages(_ messages: [MFTokenizer.Message]) throws -> [AppGenerationMessage] {
        var guidance: [String] = []
        var conversation: [AppGenerationMessage] = []
        for message in messages {
            guard message.toolCalls.isEmpty else {
                throw invalid("assistant tool calls are not supported by the in-app server",
                              "messages", "tool_messages_not_supported")
            }
            switch message.role {
            case .tool:
                throw invalid("tool messages are not supported by the in-app server",
                              "messages", "tool_messages_not_supported")
            case .system, .developer:
                if let content = message.content, !content.isEmpty {
                    guidance.append(content)
                }
            case .user, .assistant:
                guard let content = message.content,
                      !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw invalid("message content must not be empty", "messages", "invalid_message")
                }
                conversation.append(AppGenerationMessage(
                    role: message.role == .user ? .user : .assistant,
                    content: content))
            }
        }
        guard let last = conversation.last else {
            throw invalid("messages must contain a user turn", "messages", "invalid_message")
        }
        guard last.role == .user else {
            throw invalid("a trailing assistant message (prefill) is not supported by the in-app server",
                          "messages", "assistant_prefill_not_supported")
        }
        guard conversation.first?.role == .user else {
            throw invalid("the conversation must begin with a user message",
                          "messages", "invalid_message")
        }
        let merged = guidance.joined(separator: "\n\n")
        if merged.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return conversation
        }
        return [AppGenerationMessage(role: .system, content: merged)] + conversation
    }

    static func mismatchMessage(session: AppLoadedSession) -> String {
        if session.forceLogitsHead {
            return "the loaded model samples at temperature \(format(session.temperature)); "
                + "this request asked for greedy decoding (temperature 0 and repetition_penalty 1). "
                + "Reload the model with temperature 0, or enable pin-to-session in the API Server pane."
        }
        return "the loaded model decodes greedily (temperature 0); this request asked for sampling. "
            + "Reload the model with a non-zero temperature, or enable pin-to-session in the API Server pane."
    }

    private static func format(_ value: Float) -> String {
        value == value.rounded() ? String(Int(value)) : String(value)
    }

    private static func format(_ value: Int?) -> String {
        value.map(String.init) ?? "off"
    }

    private static func format(_ value: Float?) -> String {
        value.map(format) ?? "off"
    }

    private static func invalid(_ message: String, _ param: String?, _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
