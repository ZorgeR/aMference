import Foundation
import Testing
import Mference
import MferenceAppCore
import MferenceServerBridge
import MferenceServerCore

@Suite("Arbiter request translation")
struct ArbiterRequestTranslationTests {
    private static let sampled = BridgeFixtures.session(forceLogitsHead: true)
    private static let greedy = BridgeFixtures.session(forceLogitsHead: false)

    private static func body(_ extra: String = "",
                             messages: String = #"[{"role":"user","content":"hi there"}]"#) -> String {
        #"{"model":"test-model","messages":\#(messages)\#(extra)}"#
    }

    private struct Rejection {
        let name: String
        let json: String
        var dialect: ChatDialect = .chatml
        var session: AppLoadedSession = sampled
        var policy: APIServerSamplingPolicy = .rejectOnMismatch
        let code: String
        let param: String?
    }

    private static let toolHistory = #"""
    [{"role":"user","content":"hi"},
     {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
     {"role":"tool","tool_call_id":"c1","content":"ok"},
     {"role":"user","content":"and?"}]
    """#

    @Test func everyRejectionCarriesItsCode() throws {
        let rows: [Rejection] = [
            Rejection(name: "tools",
                      json: Self.body(#","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]"#),
                      code: "tools_not_supported", param: "tools"),
            Rejection(name: "tool history",
                      json: Self.body(messages: Self.toolHistory),
                      code: "tool_messages_not_supported", param: "messages"),
            Rejection(name: "seed",
                      json: Self.body(#","seed":7"#),
                      code: "unsupported_parameter", param: "seed"),
            Rejection(name: "repetition_penalty below 1",
                      json: Self.body(#","repetition_penalty":0.5"#),
                      code: "invalid_value", param: "repetition_penalty"),
            Rejection(name: "assistant prefill",
                      json: Self.body(messages: #"[{"role":"user","content":"hi"},{"role":"assistant","content":"Sure,"}]"#),
                      code: "assistant_prefill_not_supported", param: "messages"),
            Rejection(name: "sampled request against a greedy session",
                      json: Self.body(),
                      session: Self.greedy,
                      code: "sampling_mode_mismatch", param: "temperature"),
            Rejection(name: "greedy request against a sampled session",
                      json: Self.body(#","temperature":0"#),
                      code: "sampling_mode_mismatch", param: "temperature"),
            Rejection(name: "blank user content",
                      json: Self.body(messages: #"[{"role":"user","content":"   "}]"#),
                      code: "invalid_message", param: "messages"),
            Rejection(name: "system only",
                      json: Self.body(messages: #"[{"role":"system","content":"be terse"}]"#),
                      code: "invalid_message", param: "messages"),
        ]
        for row in rows {
            let request = try BridgeFixtures.validated(row.json, dialect: row.dialect)
            do {
                _ = try ArbiterRequestTranslation.translate(request, session: row.session,
                                                            policy: row.policy)
                Issue.record("\(row.name): expected a rejection")
            } catch let error as ServerRequestError {
                guard case .invalid(_, let param, let code) = error else {
                    Issue.record("\(row.name): expected .invalid, got \(error)")
                    continue
                }
                #expect(code == row.code, "\(row.name)")
                #expect(param == row.param, "\(row.name)")
            }
        }
    }

    @Test func emptyToolsArrayIsAccepted() throws {
        let request = try BridgeFixtures.validated(Self.body(#","tools":[]"#))
        let (translated, substitutions) = try ArbiterRequestTranslation.translate(
            request, session: Self.sampled, policy: .rejectOnMismatch)
        #expect(substitutions.isEmpty)
        #expect(translated.messages == [AppGenerationMessage(role: .user, content: "hi there")])
    }

    @Test func mismatchMessageNamesTheLoadedMode() throws {
        let request = try BridgeFixtures.validated(Self.body())
        #expect(throws: ServerRequestError.self) {
            try ArbiterRequestTranslation.translate(request, session: Self.greedy,
                                                    policy: .rejectOnMismatch)
        }
        do {
            _ = try ArbiterRequestTranslation.translate(request, session: Self.greedy,
                                                        policy: .rejectOnMismatch)
        } catch let error as ServerRequestError {
            #expect(error.envelope.error.message.contains("decodes greedily"))
        }
        let greedyRequest = try BridgeFixtures.validated(Self.body(#","temperature":0"#))
        do {
            _ = try ArbiterRequestTranslation.translate(greedyRequest, session: Self.sampled,
                                                        policy: .rejectOnMismatch)
        } catch let error as ServerRequestError {
            #expect(error.envelope.error.message.contains("temperature 0.2"))
        }
    }

    @Test func pinToGreedySessionRewritesTemperatureAndPenalty() throws {
        let request = try BridgeFixtures.validated(
            Self.body(#","temperature":0.7,"repetition_penalty":1.1"#))
        let (translated, substitutions) = try ArbiterRequestTranslation.translate(
            request, session: Self.greedy, policy: .pinToSession)
        #expect(translated.temperature == 0)
        #expect(translated.repetitionPenalty == 1)
        #expect(translated.isPureGreedy)
        #expect(substitutions.map(\.parameter) == ["temperature", "repetition_penalty"])
        #expect(substitutions.map(\.requested) == ["0.7", "1.1"])
        #expect(substitutions.map(\.applied) == ["0", "1"])
        #expect(substitutions.allSatisfy { $0.reason.contains("greedily") })
    }

    @Test func pinToSampledSessionRewritesOnlyWhatDiffers() throws {
        // Defaults for top_k/top_p match the session: only temperature moves.
        let request = try BridgeFixtures.validated(Self.body(#","temperature":0"#))
        let (translated, substitutions) = try ArbiterRequestTranslation.translate(
            request, session: Self.sampled, policy: .pinToSession)
        #expect(translated.temperature == 0.2)
        #expect(translated.topK == 64)
        #expect(translated.topP == 0.95)
        #expect(substitutions.map(\.parameter) == ["temperature"])
        #expect(substitutions.first?.requested == "0")
        #expect(substitutions.first?.applied == "0.2")

        let custom = try BridgeFixtures.validated(
            Self.body(#","temperature":0,"top_k":10,"top_p":0.5"#))
        let (pinned, rows) = try ArbiterRequestTranslation.translate(
            custom, session: Self.sampled, policy: .pinToSession)
        #expect(pinned.temperature == 0.2)
        #expect(pinned.topK == 64)
        #expect(pinned.topP == 0.95)
        #expect(rows.map(\.parameter) == ["temperature", "top_k", "top_p"])
        #expect(rows.map(\.requested) == ["0", "10", "0.5"])
        #expect(rows.map(\.applied) == ["0.2", "64", "0.95"])
    }

    @Test func matchingModeKeepsTheRequestsOwnSampling() throws {
        let request = try BridgeFixtures.validated(
            Self.body(#","temperature":0.7,"top_k":10,"top_p":0.5,"max_tokens":77"#))
        for policy in APIServerSamplingPolicy.allCases {
            let (translated, substitutions) = try ArbiterRequestTranslation.translate(
                request, session: Self.sampled, policy: policy)
            #expect(substitutions.isEmpty, "\(policy)")
            #expect(translated.temperature == 0.7)
            #expect(translated.topK == 10)
            #expect(translated.topP == 0.5)
            #expect(translated.maxNewTokens == 77)
        }
        let greedyRequest = try BridgeFixtures.validated(Self.body(#","temperature":0"#))
        let (translated, substitutions) = try ArbiterRequestTranslation.translate(
            greedyRequest, session: Self.greedy, policy: .rejectOnMismatch)
        #expect(substitutions.isEmpty)
        #expect(translated.isPureGreedy)
    }

    @Test func sessionOwnsContextAndRuntimeOptions() throws {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 24
        options.prefillEnabled = false
        let session = AppLoadedSession(
            modelDirectory: URL(fileURLWithPath: "/tmp/other/../other.gturbo"),
            maxContextTokens: 8192,
            runtimeOptions: options,
            forceLogitsHead: true, temperature: 0.2, topK: 64, topP: 0.95)
        let request = try BridgeFixtures.validated(Self.body(#","max_tokens":12"#))
        let (translated, _) = try ArbiterRequestTranslation.translate(
            request, session: session, policy: .rejectOnMismatch)
        #expect(translated.modelDirectory == URL(fileURLWithPath: "/tmp/other.gturbo"))
        #expect(translated.maxContextTokens == 8192)
        #expect(translated.runtimeOptions == options)
        #expect(translated.maxNewTokens == 12)
        #expect(throws: Never.self) { try translated.validate(requireModelDirectory: false) }
    }

    @Test func guidanceFoldsIntoOneLeadingSystemMessage() throws {
        // Gemma keeps `developer` distinct in the validator; the wire has no
        // such role, so the bridge folds it and merges the guidance.
        let json = Self.body(messages: #"""
        [{"role":"system","content":"Be terse."},
         {"role":"developer","content":"Answer in French."},
         {"role":"user","content":"hello"},
         {"role":"assistant","content":"bonjour"},
         {"role":"user","content":"how are you"}]
        """#)
        let request = try BridgeFixtures.validated(json, dialect: .gemma)
        #expect(request.messages.map(\.role) == [.system, .developer, .user, .assistant, .user])
        let (translated, _) = try ArbiterRequestTranslation.translate(
            request, session: Self.sampled, policy: .rejectOnMismatch)
        #expect(translated.messages == [
            AppGenerationMessage(role: .system, content: "Be terse.\n\nAnswer in French."),
            AppGenerationMessage(role: .user, content: "hello"),
            AppGenerationMessage(role: .assistant, content: "bonjour"),
            AppGenerationMessage(role: .user, content: "how are you"),
        ])
        #expect(throws: Never.self) { try translated.validate(requireModelDirectory: false) }
    }

    @Test func developerOnlyGuidanceBecomesSystem() throws {
        let json = Self.body(messages: #"""
        [{"role":"developer","content":"Answer in French."},{"role":"user","content":"hello"}]
        """#)
        let request = try BridgeFixtures.validated(json, dialect: .gemma)
        let (translated, _) = try ArbiterRequestTranslation.translate(
            request, session: Self.sampled, policy: .rejectOnMismatch)
        #expect(translated.messages.first == AppGenerationMessage(role: .system, content: "Answer in French."))
    }
}
