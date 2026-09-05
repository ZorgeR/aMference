import Foundation

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// Lets a client validate and trim chat history with the exact tokenizer before
/// the UI commits the pending user turn to durable history.
public protocol AppGenerationRequestPreparing: AppInferenceClient {
    func prepare(_ request: AppGenerationRequest) async throws -> AppGenerationRequest
}

public protocol AppGenerationContextReporting: AppGenerationRequestPreparing {
    func prepareWithContextReport(_ request: AppGenerationRequest) async throws
        -> AppPreparedGenerationRequest
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
}

/// A client that can hand every text fragment the transport delivers to a
/// caller-supplied hook, independent of the throttled `.token` cadence of
/// `AppInferenceEvent`. The hook runs on the client's reader context and
/// must not block.
public protocol AppInferenceDeltaStreaming: AppInferenceClient {
    func generate(_ request: AppGenerationRequest,
                  onTextDelta: @escaping @Sendable (String) -> Void)
        -> AsyncThrowingStream<AppInferenceEvent, Error>
}

/// A client whose transport can be torn down out of band when it stops
/// responding. `AppInferenceArbiter`'s watchdog calls it after `cancel()`
/// when a generation has produced no event for the watchdog duration.
public protocol AppInferenceTransportControlling: AnyObject, AppInferenceClient {
    func shutdown()
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
