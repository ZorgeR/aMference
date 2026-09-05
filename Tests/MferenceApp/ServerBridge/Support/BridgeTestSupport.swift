import Foundation
import Testing
import Mference
import MferenceAppCore
import MferenceServerBridge
import MferenceServerCore

enum BridgeTestError: Error {
    case timedOut
    case missingPort
}

enum BridgeFixtures {
    static let modelID = "test-model"
    static let modelDirectory = URL(fileURLWithPath: "/tmp/bridge-test.gturbo")
    static let identity = APIServerModelIdentity(family: .qwen36,
                                                 modelID: modelID,
                                                 chatDialect: .chatml)

    /// A loaded session; sampled at the app's defaults unless told otherwise.
    static func session(directory: URL = modelDirectory,
                        maxContext: Int = 4096,
                        forceLogitsHead: Bool = true) -> AppLoadedSession {
        AppLoadedSession(modelDirectory: directory,
                         maxContextTokens: maxContext,
                         runtimeOptions: AppRuntimeOptions(),
                         forceLogitsHead: forceLogitsHead,
                         temperature: forceLogitsHead ? 0.2 : 0,
                         topK: forceLogitsHead ? 64 : nil,
                         topP: forceLogitsHead ? 0.95 : nil)
    }

    /// Counts whitespace-separated words instead of tokens.
    static let wordCountMeasurer: APIServerPromptMeasurer = { request in
        try ArbiterInferenceBackend.measure(request) { messages in
            messages.reduce(0) { $0 + $1.content.split(whereSeparator: \.isWhitespace).count }
        }
    }

    static func wordCount(_ messages: [AppGenerationMessage]) -> Int {
        messages.reduce(0) { $0 + $1.content.split(whereSeparator: \.isWhitespace).count }
    }

    static func environment(
        identity: APIServerModelIdentity = identity,
        resolveIdentity: (@Sendable (URL) async throws -> APIServerModelIdentity)? = nil
    ) -> APIServerEnvironment {
        APIServerEnvironment(
            resolveIdentity: resolveIdentity ?? { _ in identity },
            measurePrompt: wordCountMeasurer,
            resolveHost: { _ in "127.0.0.1" })
    }

    /// Decodes and validates an OpenAI request body the way the server does.
    static func validated(_ json: String,
                          modelID: String = modelID,
                          dialect: ChatDialect = .chatml) throws -> ValidatedChatRequest {
        let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
        return try OpenAIRequestValidator.validate(decoded, modelID: modelID, dialect: dialect)
    }
}

/// An `APIServerModel` over an arbiter and the fake client, started on an
/// ephemeral port.
@MainActor
struct BridgeHarness {
    let client: BridgeFakeInferenceClient
    let arbiter: AppInferenceArbiter
    let model: APIServerModel
    let port: Int

    static func start(client: BridgeFakeInferenceClient = BridgeFakeInferenceClient(),
                      session: AppLoadedSession? = BridgeFixtures.session(),
                      policy: APIServerSamplingPolicy = .rejectOnMismatch,
                      busyTimeout: Duration = .seconds(5),
                      queueLimit: Int = 4,
                      environment: APIServerEnvironment = BridgeFixtures.environment()) async throws -> BridgeHarness {
        let arbiter = AppInferenceArbiter(client: client)
        if let session { await arbiter.publishLoadedSession(session) }
        let model = APIServerModel(arbiter: arbiter, environment: environment,
                                   logFlushDelay: .milliseconds(10))
        model.port = 0
        model.samplingPolicy = policy
        model.busyTimeout = busyTimeout
        model.queueLimit = queueLimit
        model.start()
        await model.awaitIdle()
        guard case .listening(_, let port) = model.runState else {
            Issue.record("server did not start: \(model.runState)")
            throw BridgeTestError.missingPort
        }
        return BridgeHarness(client: client, arbiter: arbiter, model: model, port: port)
    }

    func stop() async {
        model.stop()
        await model.awaitIdle()
    }

    var completionURL: URL {
        URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
    }
}

// MARK: - HTTP helpers

struct HTTPReply {
    let status: Int
    let data: Data
    var text: String { String(decoding: data, as: UTF8.self) }
    var json: [String: Any] {
        (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
    }
}

func send(_ method: String,
          _ path: String,
          port: Int,
          body: String? = nil,
          contentType: String? = "application/json") async throws -> HTTPReply {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
    request.httpMethod = method
    if let contentType {
        request.setValue(contentType, forHTTPHeaderField: "content-type")
    }
    request.httpBody = body.map { Data($0.utf8) }
    let (data, response) = try await URLSession.shared.data(for: request)
    let status = try #require((response as? HTTPURLResponse)?.statusCode)
    return HTTPReply(status: status, data: data)
}

func completion(_ body: String, port: Int) async throws -> HTTPReply {
    try await send("POST", "/v1/chat/completions", port: port, body: body)
}

/// Reads a streaming completion and returns its status plus every `data:`
/// payload, in order, as raw strings.
func streamCompletion(_ body: String, port: Int) async throws -> (status: Int, frames: [String]) {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    request.httpBody = Data(body.utf8)
    let (bytes, response) = try await URLSession.shared.bytes(for: request)
    let status = try #require((response as? HTTPURLResponse)?.statusCode)
    var frames: [String] = []
    for try await line in bytes.lines where line.hasPrefix("data: ") {
        frames.append(String(line.dropFirst("data: ".count)))
    }
    return (status, frames)
}

/// The `content` deltas of a streamed completion's frames.
func contentDeltas(_ frames: [String]) -> [String] {
    frames.compactMap { frame -> String? in
        guard let object = try? JSONSerialization.jsonObject(with: Data(frame.utf8)) as? [String: Any],
              let choices = object["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any] else { return nil }
        return delta["content"] as? String
    }
}

func finishReasons(_ frames: [String]) -> [String] {
    frames.compactMap { frame -> String? in
        guard let object = try? JSONSerialization.jsonObject(with: Data(frame.utf8)) as? [String: Any],
              let choices = object["choices"] as? [[String: Any]] else { return nil }
        return choices.first?["finish_reason"] as? String
    }
}

func messageContent(_ reply: HTTPReply) -> String? {
    let choices = reply.json["choices"] as? [[String: Any]]
    let message = choices?.first?["message"] as? [String: Any]
    return message?["content"] as? String
}

func finishReason(_ reply: HTTPReply) -> String? {
    let choices = reply.json["choices"] as? [[String: Any]]
    return choices?.first?["finish_reason"] as? String
}

func usage(_ reply: HTTPReply) -> [String: Any] {
    reply.json["usage"] as? [String: Any] ?? [:]
}

func errorCode(_ reply: HTTPReply) -> String? {
    let error = reply.json["error"] as? [String: Any]
    return error?["code"] as? String
}

func errorMessage(_ reply: HTTPReply) -> String? {
    let error = reply.json["error"] as? [String: Any]
    return error?["message"] as? String
}

func waitUntil(_ predicate: @Sendable () async -> Bool,
               timeout: Duration = .seconds(5)) async throws {
    let deadline = ContinuousClock.now + timeout
    while ContinuousClock.now < deadline {
        if await predicate() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for condition")
    throw BridgeTestError.timedOut
}

// MARK: - Log helpers

extension APIServerLogEntry {
    var serverEvent: ServerLogEvent? {
        if case .server(let event) = payload { return event }
        return nil
    }

    var bridgeNote: APIServerBridgeNote.Kind? {
        if case .bridge(let note) = payload { return note.kind }
        return nil
    }
}
