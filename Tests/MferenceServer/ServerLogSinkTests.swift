import Foundation
import Synchronization
import Testing
@testable import MferenceServerCore

/// Keeps every event the server records, in order.
private final class RecordingServerLogSink: ServerLogSink, Sendable {
    private let events = Mutex<[ServerLogEvent]>([])

    func record(_ event: ServerLogEvent) {
        events.withLock { $0.append(event) }
    }

    var recorded: [ServerLogEvent] {
        events.withLock { $0 }
    }
}

private struct DecodeFailure: Error {}

/// Answers at once, so the log sees a whole lifecycle.
private actor ScriptedServerBackend: ServerInferenceBackend {
    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4,
                               cachedTokens: 2))
    }
}

/// Parks in `generate` until released, so the queue can be filled.
private actor ParkingBackend: ServerInferenceBackend {
    private var parked: [CheckedContinuation<Void, Never>] = []
    private var released = false

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        if !released {
            await withCheckedContinuation { parked.append($0) }
        }
        return ServerCompletion(
            content: "done",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    /// Latches open: a queued request that reaches `generate` only after the
    /// release must not park again, or shutdown would wait on it forever.
    func releaseAll() {
        released = true
        for continuation in parked { continuation.resume() }
        parked.removeAll()
    }
}

/// Rejects in `prepare` the way an embedding host with no model loaded does.
private actor UnavailableBackend: ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        throw ServerRequestError.unavailable("no model is loaded")
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        Issue.record("generate must not run after prepare fails")
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

/// Carries backend-private state from `prepare` into `generate`.
private actor PayloadBackend: ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        PreparedGeneration(request: request, payload: "ticket-7")
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let ticket = prepared.payload as? String ?? "missing"
        return ServerCompletion(
            content: ticket,
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private struct RunningServer {
    let server: MferenceHTTPServer
    let port: Int
    let sink: RecordingServerLogSink
}

private func startServer(backend: any ServerInferenceBackend,
                         queueLimit: Int = 1) async throws -> RunningServer {
    let sink = RecordingServerLogSink()
    let server = MferenceHTTPServer(
        modelID: "test-model",
        queueLimit: queueLimit,
        backend: backend,
        log: sink)
    let channel = try await server.start(port: 0)
    let port = try #require(channel.localAddress?.port)
    return RunningServer(server: server, port: port, sink: sink)
}

private let completionPath = "/v1/chat/completions"
private let completionBody = Data(#"""
{"model":"test-model","messages":[{"role":"user","content":"hi"}]}
"""#.utf8)

private func send(_ method: String,
                  _ path: String,
                  port: Int,
                  body: Data? = nil,
                  contentType: String? = "application/json") async throws -> (Data, Int) {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
    request.httpMethod = method
    if let contentType {
        request.setValue(contentType, forHTTPHeaderField: "content-type")
    }
    request.httpBody = body
    let (data, response) = try await URLSession.shared.data(for: request)
    let status = try #require((response as? HTTPURLResponse)?.statusCode)
    return (data, status)
}

private func rejection(_ event: ServerLogEvent) -> (status: UInt, code: String)? {
    if case .requestRejected(let status, let code, _) = event.kind {
        return (status, code)
    }
    return nil
}

private func failure(_ event: ServerLogEvent) -> (status: UInt, streaming: Bool, detail: String)? {
    if case .requestFailed(let status, let streaming, let detail) = event.kind {
        return (status, streaming, detail)
    }
    return nil
}

@Suite("Server log sink", .serialized)
struct ServerLogSinkTests {
    @Test func servedRoutesAreLogged() async throws {
        let running = try await startServer(backend: ScriptedServerBackend())

        #expect(try await send("GET", "/health?probe=1", port: running.port, contentType: nil).1 == 200)
        #expect(try await send("GET", "/v1/models", port: running.port, contentType: nil).1 == 200)

        let events = running.sink.recorded
        #expect(events.map(\.kind) == [.routed(status: 200), .routed(status: 200)])
        #expect(events.map(\.method) == ["GET", "GET"])
        #expect(events.map(\.path) == ["/health", "/v1/models"])
        #expect(events.allSatisfy { $0.id == nil && $0.requestedModel == nil })

        try await running.server.shutdown()
    }

    @Test func completionLifecycleCarriesTheRoute() async throws {
        let running = try await startServer(backend: ScriptedServerBackend())

        let (_, status) = try await send("POST", completionPath, port: running.port,
                                         body: completionBody)
        #expect(status == 200)

        let events = running.sink.recorded
        #expect(events.count == 2)
        let started = try #require(events.first)
        #expect(started.kind == .requestStarted(method: "POST", path: completionPath,
                                                streaming: false))
        #expect(started.method == "POST")
        #expect(started.path == completionPath)
        #expect(started.requestedModel == "test-model")
        #expect(started.id?.hasPrefix("chatcmpl-") == true)

        let completed = try #require(events.last)
        #expect(completed.id == started.id)
        #expect(completed.method == "POST")
        #expect(completed.path == completionPath)
        #expect(completed.requestedModel == "test-model")
        if case .requestCompleted(_, let prompt, let cached, let completion, let finish) = completed.kind {
            #expect(prompt == 3)
            #expect(cached == 2)
            #expect(completion == 1)
            #expect(finish == "stop")
        } else {
            Issue.record("expected a completion, got \(completed.kind)")
        }

        try await running.server.shutdown()
    }

    @Test func everySynchronousRejectionIsLogged() async throws {
        let running = try await startServer(backend: ScriptedServerBackend())
        let port = running.port

        let oversized = Data(repeating: UInt8(ascii: " "),
                             count: MferenceHTTPServer.maximumBodyBytes + 1)
        #expect(try await send("POST", completionPath, port: port, body: oversized).1 == 413)
        #expect(try await send("POST", completionPath, port: port, body: completionBody,
                               contentType: "text/plain").1 == 415)
        #expect(try await send("DELETE", "/v1/models", port: port, contentType: nil).1 == 405)
        #expect(try await send("GET", "/v1/nope", port: port, contentType: nil).1 == 404)
        #expect(try await send("POST", completionPath, port: port,
                               body: Data("{not json".utf8)).1 == 400)
        let unsupported = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"n":2}
        """#.utf8)
        #expect(try await send("POST", completionPath, port: port, body: unsupported).1 == 400)
        let wrongModel = Data(#"""
        {"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        #expect(try await send("POST", completionPath, port: port, body: wrongModel).1 == 404)

        let expected: [(method: String, path: String, status: UInt, code: String, model: String?)] = [
            ("POST", completionPath, 413, "request_too_large", nil),
            ("POST", completionPath, 415, "unsupported_media_type", nil),
            ("DELETE", "/v1/models", 405, "method_not_allowed", nil),
            ("GET", "/v1/nope", 404, "not_found", nil),
            ("POST", completionPath, 400, "invalid_json", nil),
            ("POST", completionPath, 400, "unsupported_value", "test-model"),
            ("POST", completionPath, 404, "model_not_found", "gpt-4o"),
        ]
        let events = running.sink.recorded
        #expect(events.count == expected.count)
        for (event, row) in zip(events, expected) {
            #expect(event.id == nil)
            #expect(event.method == row.method)
            #expect(event.path == row.path)
            #expect(event.requestedModel == row.model)
            let rejected = rejection(event)
            #expect(rejected?.status == row.status, "\(row)")
            #expect(rejected?.code == row.code, "\(row)")
        }

        try await running.server.shutdown()
    }

    @Test func queueOverflowIsLoggedWithTheRoute() async throws {
        let backend = ParkingBackend()
        let running = try await startServer(backend: backend, queueLimit: 1)
        let port = running.port

        // One generating, one queued: the gate is now full.
        let first = Task { try await send("POST", completionPath, port: port, body: completionBody).1 }
        let second = Task { try await send("POST", completionPath, port: port, body: completionBody).1 }
        let deadline = ContinuousClock.now + .seconds(2)
        while await running.server.queuedRequestCount != 1, ContinuousClock.now < deadline {
            await Task.yield()
        }
        #expect(await running.server.queuedRequestCount == 1)

        let (data, status) = try await send("POST", completionPath, port: port, body: completionBody)
        #expect(status == 429)
        #expect(String(decoding: data, as: UTF8.self).contains("queue_full"))

        let failed = try #require(running.sink.recorded.last { failure($0)?.status == 429 })
        #expect(failed.id?.hasPrefix("chatcmpl-") == true)
        #expect(failed.method == "POST")
        #expect(failed.path == completionPath)
        #expect(failed.requestedModel == "test-model")
        #expect(failure(failed)?.streaming == false)
        #expect(failure(failed)?.detail == "queue_full: generation queue is full")

        await backend.releaseAll()
        #expect(try await first.value == 200)
        #expect(try await second.value == 200)
        try await running.server.shutdown()
    }

    @Test func unavailableBackendAnswers503AndLogsIt() async throws {
        let running = try await startServer(backend: UnavailableBackend())

        let (data, status) = try await send("POST", completionPath, port: running.port,
                                            body: completionBody)
        #expect(status == 503)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""code":"service_unavailable""#))
        #expect(text.contains(#""type":"server_error""#))
        #expect(text.contains("no model is loaded"))

        // Rejected in `prepare`, before the head: a stream keeps the status too.
        let streaming = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        #expect(try await send("POST", completionPath, port: running.port, body: streaming).1 == 503)

        let failures = running.sink.recorded.compactMap(failure)
        #expect(failures.map(\.status) == [503, 503])
        #expect(failures.map(\.streaming) == [false, false])
        #expect(failures.allSatisfy { $0.detail == "service_unavailable: no model is loaded" })

        try await running.server.shutdown()
    }

    @Test func preparedGenerationPayloadReachesGenerate() async throws {
        let running = try await startServer(backend: PayloadBackend())

        let (data, status) = try await send("POST", completionPath, port: running.port,
                                            body: completionBody)
        #expect(status == 200)
        #expect(String(decoding: data, as: UTF8.self).contains(#""content":"ticket-7""#))

        try await running.server.shutdown()
    }

    @Test func standardErrorSinkReproducesTheLegacyLines() {
        let date = Date(timeIntervalSince1970: 1_772_000_000)
        let stamp = date.formatted(.iso8601)
        let id = "chatcmpl-0123"
        func line(_ kind: ServerLogEvent.Kind) -> String? {
            StandardErrorServerLogSink.line(for: ServerLogEvent(
                id: id, date: date, method: "POST", path: completionPath,
                requestedModel: "test-model", kind: kind))
        }

        #expect(line(.requestStarted(method: "POST", path: completionPath, streaming: true))
                == "[\(stamp)] request \(id) started streaming=true\n")
        #expect(line(.requestCompleted(duration: .milliseconds(1_500),
                                       promptTokens: 12, cachedTokens: 4,
                                       completionTokens: 7, finishReason: "stop"))
                == "[\(stamp)] request \(id) completed in 1.5s prompt=12 cached=4 completion=7 finish=stop\n")
        #expect(line(.requestFailed(status: 500, streaming: false, detail: "DecodeFailure()"))
                == "[\(stamp)] request \(id) failed status=500 streaming=false error=DecodeFailure()\n")
        #expect(line(.streamAborted(reason: "error envelope could not be encoded"))
                == "[\(stamp)] request \(id) stream aborted: error envelope could not be encoded\n")
        #expect(line(.requestRejected(status: 404, code: "not_found", message: "route not found")) == nil)
        #expect(line(.routed(status: 200)) == nil)
    }

    @Test func standardErrorSinkWritesOnlyTheLegacyKinds() {
        let written = Mutex<[Data]>([])
        let sink = StandardErrorServerLogSink { chunk in
            written.withLock { $0.append(chunk) }
        }
        let date = Date(timeIntervalSince1970: 1_772_000_000)

        sink.record(ServerLogEvent(method: "GET", path: "/health", kind: .routed(status: 200)))
        sink.record(ServerLogEvent(method: "GET", path: "/nope",
                                   kind: .requestRejected(status: 404, code: "not_found",
                                                          message: "route not found")))
        sink.record(ServerLogEvent(id: "chatcmpl-1", date: date, method: "POST",
                                   path: completionPath,
                                   kind: .requestStarted(method: "POST", path: completionPath,
                                                         streaming: false)))

        let lines = written.withLock { $0 }.map { String(decoding: $0, as: UTF8.self) }
        #expect(lines == ["[\(date.formatted(.iso8601))] request chatcmpl-1 started streaming=false\n"])
    }

    @Test func serverLogDescribesRequestErrorsByCodeAndMessage() {
        let sink = RecordingServerLogSink()
        let log = ServerLog(sink: sink)

        log.requestFailed(id: "chatcmpl-2", method: "POST", path: completionPath,
                          requestedModel: "test-model", status: 429, streaming: true,
                          error: ServerRequestError.queueFull)
        log.requestFailed(id: "chatcmpl-3", method: "POST", path: completionPath,
                          requestedModel: "test-model", status: 500, streaming: false,
                          error: DecodeFailure())

        let failures = sink.recorded.compactMap(failure)
        #expect(failures.map(\.detail) == ["queue_full: generation queue is full", "DecodeFailure()"])
        #expect(failures.map(\.streaming) == [true, false])
        #expect(sink.recorded.map(\.id) == ["chatcmpl-2", "chatcmpl-3"])
    }

    @Test func unavailableErrorUsesTheServiceUnavailableEnvelope() {
        let envelope = ServerRequestError.unavailable("no model is loaded").envelope
        #expect(envelope.error.code == "service_unavailable")
        #expect(envelope.error.type == "server_error")
        #expect(envelope.error.message == "no model is loaded")
        #expect(envelope.error.param == nil)
    }
}
