# In-app API Server pane — design study

**Date:** 2026-09-03
**Status:** design study, not an accepted plan. Produced by a multi-agent review (3 candidate architectures, 3 judges); file:line citations were verified against commit b1a56e8 on the day. Re-verify before implementing.
**Scope:** serve the model already loaded in the Mac app as the existing OpenAI-compatible server, with an activity log. Batching is out of scope (see `2026-09-03-batched-decode-feasibility.md`).
**Companion:** `docs/superpowers/specs/2026-09-03-app-pages-design.md` (Models + API Server pages).

---

# Implementation Plan — In-App "API Server" Pane for Mference Mac

**Design:** Arbiter-brokered in-app OpenAI server (single decode session, leased)

---

## 1. Verdict

**Feasible.** The pane can serve the model the app already has loaded, with real SSE streaming and a real request log, with **zero changes to the decode wire protocol** (`Sources/MferenceDecodeProtocol/DecodeProtocol.swift`) and no second copy of the weights.

**Headline caveat:** *there is no prompt cache on this path, and there cannot be one without moving the HTTP layer into the decode service.* `ServerPromptCache` is internal to `MferenceServerCore` (`ServerPromptCache.swift:39`), keyed on facts that only exist inside `ServerModelSession`, and the decode service calls `runner.reset()` before **every** generation (`RealInferenceClient.swift:338`). So every HTTP request re-prefills the entire conversation and `usage.prompt_tokens_details.cached_tokens` is always `0`. For multi-turn agent traffic the in-app server will be **materially slower per turn** than the standalone `MferenceServer` binary. This must be stated in the pane, not in a footnote.

Second caveat, smaller but user-visible: **tool calling is impossible on this path.** `StructuredAssistantDecoder` discriminates tool blocks by *special token ID* (`StructuredAssistantDecoder.swift:58-80`), and the decode wire carries only coalesced text (`DecodeProtocol.swift`, `DecodeServiceEvent.textDelta`). The service also constructs the decoder with `allowedTools: []` and throws on any tool call (`RealInferenceClient.swift:341-345`). Requests carrying `tools` get an explicit `400`, never a mystery `500`.

---

## 2. Chosen architecture, and why it won

`MferenceHTTPServer` runs **in the Mac app process**. The weights stay exactly where they are today — inside the `MferenceDecodeService` child process (`DecodeServiceInferenceClient.swift:213-234`). One new actor, `AppInferenceArbiter`, becomes the **sole owner** of the `DecodeServiceInferenceClient` instance, and hands out leases to two consumers: the chat UI (`AppModel`) and each HTTP request. A `ServerInferenceBackend` conformance in a new bridge target forwards HTTP requests through that arbiter.

**Why the arbiter, and not just "add a server":** the decode pipe is not one-generation-at-a-time by *policy*, it is one-at-a-time by *transport construction*. `DecodeServiceInferenceClient.generate` reads frames straight off the single shared `handles.output` in a blocking loop, drops frames whose id does not match (`:101`), and **throws** on any sequence gap rather than resynchronising (`:104-108`). `ensureLoaded` (`:47`) and `unload` (`:66`) read the same handle. Two concurrent readers do not degrade — they interleave 4-byte length headers and corrupt the stream irrecoverably. Making one actor the sole caller is the only structural fix.

**Why it beat the alternatives:**

- **vs. a second `ServerModelSession` in the app** — impossible, not merely undesirable. `ServerModelSession.init` is `private` (`ServerInference.swift:278`); the only entry point `static load` (`:195`) calls `MetalContext()`, `Model.load`, `ForwardRunnerFactory.make`. Any use is a guaranteed second 6.6–20 GB load.
- **vs. hosting the server inside `MferenceDecodeService`** — the highest-fidelity option and the correct *phase 2* (it is the only place token IDs exist, so the only route to tools and the prompt cache), but it requires **four** prerequisite rewrites before a user sees a single log row: a breaking outbound wire change (`DecodeServiceEvent` → an enum frame), a full app-side demux reader rewrite (unsolicited frames break `ensureLoaded`'s single-frame read at `:45-50`), a new single-writer discipline for fd 1 (today `DecodeServiceOutbox.runWriter` and `Entry.write` never overlap because `.generate` blocks the command loop, `Entry.swift:95-107`), and a target split because `MferenceDecodeService` has **no test target** at all (`Package.swift:72-76`). It also puts a network listener and the whole NIO stack inside the crash-sensitive 20 GB process.
- **vs. the simpler app-side "lease + preempt" variant** — that design leaves `ensureLoaded`/`unload` outside the lease by convention, mirrors actor state into a `nonisolated` Mutex for MainActor gates (TOCTOU by construction), and makes chat *preempt* a decoding HTTP request, which on a committed SSE stream can only be reported in-band (`HTTPServer.swift:465-485`) where most OpenAI clients silently render it as a short successful answer.

---

## 3. Files to CREATE

### 3.1 `MferenceServerCore` (server module)

**`Sources/MferenceServer/Core/ServerLogSink.swift`**

The injectable log seam. Required — see §7 for why a backend wrapper provably cannot do this.

```swift
public struct ServerLogEvent: Sendable {
    public enum Kind: Sendable {
        case requestStarted(method: String, path: String, streaming: Bool)
        case requestCompleted(duration: Duration,
                              promptTokens: Int, cachedTokens: Int,
                              completionTokens: Int, finishReason: String)
        /// Synchronous rejections that never reach the backend.
        case requestRejected(status: UInt, code: String, message: String)
        /// A served route that produced no request lifecycle (GET /health, /v1/models).
        case routed(status: UInt)
        case requestFailed(status: UInt, streaming: Bool, detail: String)
        case streamAborted(reason: String)
    }
    public let id: String?            // chatcmpl-… when one exists
    public let date: Date
    public let method: String
    public let path: String
    public let requestedModel: String? // decoded BEFORE validate throws
    public let kind: Kind
    public init(...)
}

public protocol ServerLogSink: Sendable {
    /// MUST NOT block: called on the NIO event loop at several rejection sites.
    func record(_ event: ServerLogEvent)
}

public struct StandardErrorServerLogSink: ServerLogSink, Sendable {
    public init() {}
    public func record(_ event: ServerLogEvent)   // reproduces today's `[<ISO8601>] …` lines verbatim
}
```

### 3.2 `MferenceAppCore` (app core, no NIO)

**`Sources/MferenceApp/Core/Inference/AppInferenceArbiter.swift`**

```swift
public enum AppInferenceOwner: Hashable, Sendable {
    case chat
    case http(UUID)
}

public struct AppLoadedSession: Equatable, Sendable {
    public let modelDirectory: URL
    public let maxContextTokens: Int
    public let runtimeOptions: AppRuntimeOptions
    public let forceLogitsHead: Bool      // the loaded session's sampling mode
    public let temperature: Float
    public let topK: Int?
    public let topP: Float?
}

public struct AppInferenceActivity: Equatable, Sendable {
    public let activeOwner: AppInferenceOwner?
    public let queuedChat: Bool
    public let queuedHTTPCount: Int
    public let loadedSession: AppLoadedSession?
}

public actor AppInferenceArbiter {
    public init(client: any AppInferenceClient)

    /// Same element type AppModel already consumes, so its `for try await` loop is unchanged.
    /// `onTextDelta` is the un-throttled per-frame text hook the SSE path needs.
    public func stream(_ request: AppGenerationRequest,
                       owner: AppInferenceOwner,
                       onTextDelta: @escaping @Sendable (String) -> Void = { _ in })
        -> AsyncThrowingStream<AppInferenceEvent, Error>

    /// Lease-scoped. Dequeues a queued owner without touching the wire;
    /// sends the global wire cancel only when `owner` actually holds the lease.
    public func cancel(owner: AppInferenceOwner)

    /// Load/unload barrier: stop admitting → cancel the active lease →
    /// AWAIT its terminal frame → fail queued waiters → run `body` → publish → resume.
    public func withExclusiveSession<T: Sendable>(
        reason: String,
        _ body: @Sendable () async throws -> T) async throws -> T

    public func publishLoadedSession(_ session: AppLoadedSession?)

    public var loadedSession: AppLoadedSession? { get }
    public var activity: AppInferenceActivity { get }
    public nonisolated var activityStream: AsyncStream<AppInferenceActivity> { get }
}
```

Admission rules, FIFO within a class, `.chat` admitted ahead of any queued `.http`. **No preemption of a request that is already decoding.** An **opt-in** watchdog (`AppInferenceArbiter.init(client:watchdog:)`, default `nil`; the app does not enable it today) covers the wedge case: `DecodeFrameCodec.readExactly` blocks in `handle.read(upToCount:)` with no deadline (`DecodeProtocol.swift:302-312`) and Swift cancellation cannot interrupt it, so a wedge that used to stall one chat would hang everything. When enabled, its idle clock starts at the *first event* of a generation, never at registration (the service sends one frame per prefill chunk, and the first can follow minutes of first-touch expert hashing); on expiry the arbiter cancels on the wire, `disconnect()`s the client's pipe before the lease is released, fails the lease with `AppInferenceArbiterError.transportLost`, publishes `loadedSession = nil`, and reaps the process off the actor (`shutdown()`). Enabling it by default is gated on a service-side heartbeat frame: with one frame per prefill chunk, any fixed budget is either too short for a long chunk or too long to be useful against a wedge.

**`Sources/MferenceApp/Core/Inference/AppInferenceLease.swift`**
Internal `Ticket` / waiter-queue types for the arbiter (kept out of the actor file for readability). Mirrors the shape of `ServerCoordinator` (`ServerInference.swift:61-166`), one level above it.

### 3.3 New target `MferenceServerBridge` — `Sources/MferenceApp/ServerBridge/`

**`ArbiterInferenceBackend.swift`**

```swift
public actor ArbiterInferenceBackend: ServerInferenceBackend {
    public init(arbiter: AppInferenceArbiter,
                log: APIServerLogSink,
                policy: APIServerSamplingPolicy,
                busyTimeout: Duration = .seconds(120))
    public func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration
    public func generate(_ prepared: PreparedGeneration,
                         onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void)
        async throws -> ServerCompletion
}
```

**`ArbiterRequestTranslation.swift`** — a **pure function**, the single most valuable unit test in the feature:

```swift
public enum APIServerSamplingPolicy: String, Sendable { case rejectOnMismatch, pinToSession }

public struct APIServerSubstitution: Equatable, Sendable {
    public let parameter: String, requested: String, applied: String, reason: String
}

public enum ArbiterRequestTranslation {
    public static func translate(_ request: ValidatedChatRequest,
                                 session: AppLoadedSession,
                                 policy: APIServerSamplingPolicy)
        throws -> (request: AppGenerationRequest, substitutions: [APIServerSubstitution])
}
```

**`APIServerModel.swift`** — `@MainActor @Observable public final class APIServerModel`. Owns `runState`, `port`, `bindMode`, the log ring buffer, `start()` / `stop()` / `cancelActiveRequest()`, and the `activityStream` consumer task.

**`APIServerLog.swift`** — `APIServerLogEntry` (Sendable value) + `APIServerLogSink` (a `Sendable` final class conforming to `ServerLogSink`, holding `Mutex<Deque<APIServerLogEntry>>`, cap 500, drop-oldest with a `droppedBefore` counter) + `APIServerLogEntryPresentation.resolve(_:)` (pure).

**`APIServerPresentationState.swift`** — pure `resolve(APIServerSnapshot) -> Self` returning `label / detail / severity / showsActivity / primaryAction`, copied in shape from `AppPresentationState.resolve` (`AppPresentationState.swift:81`) and reusing `AppPresentationSeverity` (`:3-9`).

**`APIServerModelIdentity.swift`** — `ManifestReader.peekFamily(directoryURL:)` (`ManifestReader.swift:533`) → the family→id table copied verbatim from `ServerModelSession.defaultModelID` (`ServerInference.swift:171-181`), plus `chatDialect` from `MFTokenizer.load(forModelDirectory:).dialect` (`Tokenizer.swift:51`). **Never instantiates `ServerModelSession`.**

**`APIServerBindOption.swift`** — `ServerBindMode` is *not* `CaseIterable` (`ServerArguments.swift:102-105`), so the bridge supplies the picker's option list plus label/help/`isExposedBeyondThisMac` for each case.

### 3.4 `MferenceMacPresentation`

**`Sources/MferenceApp/MacPresentation/AppPrimaryPanePresentation.swift`**

```swift
public enum AppPrimaryPane: String, CaseIterable, Identifiable, Sendable {
    case chat, apiServer
    public static let storageKey = "Mference.primaryPane"
    public var id: String { rawValue }
    public var title: String        // "Chat" / "API Server"
    public var systemImage: String  // "bubble.left" / "network"
    public static func resolve(_ stored: String) -> Self { Self(rawValue: stored) ?? .chat }
}
```

Modelled one-for-one on `AppAppearance` (`MferenceMacTheme.swift:4-40`).

### 3.5 `MferenceMac` (SwiftUI, inside the existing target path)

- **`Sources/MferenceApp/Mac/Server/APIServerView.swift`** — the pane. `Form { … }.formStyle(.grouped).scrollContentBackground(.hidden)`, the `InspectorView.swift:9-19` recipe. Header strip needs `.gesture(WindowDragGesture())` because the window is `.hiddenTitleBar` (`MferenceMacApp.swift:47`).
- **`Sources/MferenceApp/Mac/Server/APIServerEndpointSection.swift`** — port `TextField`+`Stepper` (`.fixedSize()`), segmented bind picker, Connect rows (base URL + model id, `.font(.caption.monospaced())`, copy buttons), limits disclosure.
- **`Sources/MferenceApp/Mac/Server/APIServerLogListView.swift`** — `ScrollViewReader` + `LazyVStack` of `.font(.caption.monospacedDigit())` rows; noise filter toggle (hide `/health` + `/v1/models`), text filter, Clear, Copy All; failed rows expand to show `detail`; a "… N entries dropped" separator wherever `droppedBefore > 0`.

### 3.6 Tests to create

| Path | Covers |
|---|---|
| `Tests/MferenceApp/Core/Inference/AppInferenceArbiterTests.swift` | Lease FIFO; chat admitted ahead of queued HTTP; **no preemption**; `cancel(owner:)` on a *queued* owner does not touch the wire; `withExclusiveSession` awaits the terminal frame before running its body; watchdog fires. Uses the existing `Tests/MferenceApp/Core/Support/MockInferenceClient.swift` / `FakeInferenceClient.swift`. |
| `Tests/MferenceApp/Core/Inference/DecodeServiceCancelRegressionTests.swift` | ⌘. still cancels after the `onTermination` narrowing (§4.6). |
| `Tests/MferenceApp/ServerBridge/ArbiterRequestTranslationTests.swift` | Table-driven: every rejection and every substitution in §5. |
| `Tests/MferenceApp/ServerBridge/ArbiterInferenceBackendTests.swift` | Drives a **real** `MferenceHTTPServer` on `port: 0` with a fake client — the pattern `Tests/MferenceServer/HTTPServerTests.swift:216-221` already proves works with model-free backends. Covers SSE deltas, non-streaming accumulation, stop-string truncation, 503 on no-model, 429 on busy timeout. |
| `Tests/MferenceApp/ServerBridge/APIServerLogSinkTests.swift` | Ring cap, drop counting, coalesced flush, non-blocking `record`. |
| `Tests/MferenceApp/ServerBridge/APIServerPresentationStateTests.swift` | Every run state → label/severity/action. |
| `Tests/MferenceApp/ServerBridge/APIServerStartStopTests.swift` | Fresh instance per start; serialized start/stop; bound port read from the channel. |
| `Tests/MferenceServer/ServerLogSinkTests.swift` | Every silent site now emits: 413, 415, 405, 404, `/health`, `/v1/models`, validation 400, `unknownModel` 404 **naming the requested model**, 429. |
| `Tests/MferenceApp/Core/Inference/GreedyHeadEquivalenceTests.swift` | See §4.7 — greedy decode is bit-identical with `forceLogitsHead` true vs false. **Gates the session-key relaxation.** |

---

## 4. Files to MODIFY

### 4.1 `Package.swift`

Add the bridge target and its test target; wire it into the executable. **Do not** add `MferenceServerCore` to `MferenceAppCore` — `MferenceDecodeService` depends on `MferenceAppCore` (`Package.swift:72-76`), so that edge would link SwiftNIO into the process holding the weights.

```swift
.target(
    name: "MferenceServerBridge",
    dependencies: ["MferenceAppCore", "MferenceServerCore"],
    path: "Sources/MferenceApp/ServerBridge"
),
// MferenceMac (Package.swift:92-100): dependencies become
//   ["MferenceAppCore", "MferenceMacPresentation", "MferenceServerBridge"]
.testTarget(
    name: "MferenceServerBridgeTests",
    dependencies: ["MferenceServerBridge", "MferenceAppCore", "MferenceServerCore"],
    path: "Tests/MferenceApp/ServerBridge"
),
```

`Sources/MferenceApp/ServerBridge` is a free path — the existing targets claim `Core`, `MacPresentation`, and `Mac` only.

### 4.2 `Sources/MferenceServer/Core/ServerLog.swift`

`ServerLog` (currently `enum ServerLog` at `:10`, no `public`, four statics funnelling to `FileHandle.standardError` at `:51-54`) becomes a `Sendable` struct wrapping a `ServerLogSink`, keeping the same four method names so the call sites change only their receiver. `StandardErrorServerLogSink` reproduces the existing format byte-for-byte, so `Sources/MferenceServer/Command/main.swift` needs **no edit** and the CLI's stderr is unchanged.

### 4.3 `Sources/MferenceServer/Core/HTTPServer.swift`

1. `init` (`:21-33`) gains `log: any ServerLogSink = StandardErrorServerLogSink()`, threaded through `start`'s local captures (`:36-42`) into `ServerHTTPHandler.init` (`:124-144`).
2. **Emit at every currently-silent site.** Verified silent today: 413 (`:162`), `GET /health` (`:184`), `GET /v1/models` (`:186`), 415 (`:197`), 405 (`:204`), 404 (`:208`), and the synchronous `catch` for validation/JSON (`:294-301`). Only `:242`, `:272`, `:444`, `:473` log at all.
3. Pass `head.method.rawValue` and the already-computed `path` (`:182`) into `handleCompletion` so async rows carry method+path.
4. **Capture `decoded.model` before `OpenAIRequestValidator.validate` throws.** `:218` decodes and `:219` validates in the same `do`. Split them so an `unknownModel` 404 can name what the client actually sent — model matching is strict equality (`OpenAIModels.swift:261`) and this is the single most common integration mistake.
5. `failure(for:)` (`:454-462`) gains one line mapping `ServerRequestError.unavailable` → `503`.
6. `queuedRequestCount` / `hasActiveRequest` / `acceptedConnectionCount` (`:104`, `:108`, `:112`) become `public` — they are currently internal, reachable only via `@testable`, so the pane cannot show queue depth.

### 4.4 `Sources/MferenceServer/Core/OpenAIModels.swift`

Add `case unavailable(String)` to `ServerRequestError` (`:229-245`) with envelope `code: "service_unavailable"`. Without it the only client-visible statuses are 400/404/429/500, and "no model is loaded" would have to masquerade as a 400.

### 4.5 `Sources/MferenceServer/Core/ServerInference.swift`

`PreparedGeneration` (`:28-38`) gains `public let payload: (any Sendable)?` defaulted to `nil`. Required because `ValidatedChatRequest` has **no public init** (`OpenAIModels.swift:248-255`, implicit internal memberwise), so the translated `AppGenerationRequest` cannot otherwise survive `prepare` → `generate`; re-translating inside `generate` would put rejection logic exactly where the contract comment at `:43-46` forbids it. The default keeps all existing callers and the model-free test stubs source-compatible.

### 4.6 `Sources/MferenceApp/Core/Inference/DecodeServiceInferenceClient.swift`

Two changes, both small, both load-bearing:

1. **Add the un-throttled delta hook.** `generate(_:)` becomes `generate(_:onTextDelta:)` with the old signature delegating with a no-op closure. One line beside `generationTranscriptMailbox.append(event.textDelta)` (`:117`) calls `onTextDelta(event.textDelta)`. This is necessary because the current `.token` path throttles to 0.5 s and **blanks the text after the first visible chunk**: `textDelta: beginsVisibleText ? event.textDelta : ""` (`:127`). A backend consuming `AppInferenceEvent` alone would emit essentially no text. Zero behaviour change for the chat UI.
2. **Fix the fire-on-normal-completion cancel.** `continuation.onTermination = { task.cancel(); self?.cancel() }` (`:156-159`) fires on **normal** completion, posting a global payload-free `DecodeServiceCommand.cancel` (`:174-176`; `DecodeProtocol.swift:161`) that `Entry.swift:18` applies out-of-band to whatever is running. Harmless today (single-flight UI), **fatal with a queue**: request N's teardown kills request N+1. Switch on `Termination` and cancel only on `.cancelled` — or, preferred, drop the call entirely and let the arbiter own all cancellation.

### 4.7 `Sources/MferenceApp/Core/Inference/RealInferenceClient.swift`

**The temperature-0 fix.** `run` rebuilds a `SessionLoadKey` from the *request* with `forceLogitsHead: Self.forceLogitsHead(for: request)` = `!request.isPureGreedy` (`:258-259`, `:298-302`) and throws `.reloadRequired` unless it equals the loaded key (`:304`). The app loads with `forceLogitsHead = temperature != 0` (`AppModel.swift:343-345`, `:418`). So a request with `temperature: 0` — the single most common agent setting — against an app loaded at 0.2 fails, and every non-`ServerRequestError` collapses to the opaque 500 envelope (`HTTPServer.swift:459-462`).

Relax `:304` from equality to **compatibility on `forceLogitsHead` only** (`directory`, `maxContext`, `options` stay strict equality):

```swift
guard requestKey.isServable(by: loadedKey) else { throw AppInferenceError.reloadRequired }
// isServable: directory/maxContext/options equal AND
//             (loaded.forceLogitsHead || !request.forceLogitsHead)
```

Rationale: a session that materialises the full logits head can always serve a greedy argmax request; the reverse is not true.

> **This relaxation must be gated on `Tests/MferenceApp/Core/Inference/GreedyHeadEquivalenceTests.swift` passing** — a numerical test that a greedy decode produces an identical token sequence on a session loaded with `forceLogitsHead: true` and one loaded with `false`. I have not read the runner's head fast-path; if the test fails, drop the relaxation and fall back to the `400 sampling_mode_mismatch` in §5.

### 4.8 `Sources/MferenceApp/Core/Inference/AppInferenceClient.swift`

Add one narrow refinement, matching the existing family idiom at `:10-35`:

```swift
public protocol AppInferenceDeltaStreaming: AppInferenceClient {
    func generate(_ request: AppGenerationRequest,
                  onTextDelta: @escaping @Sendable (String) -> Void)
        -> AsyncThrowingStream<AppInferenceEvent, Error>
}
```

### 4.9 `Sources/MferenceApp/Core/State/AppModel.swift`

- `init` (`:79-83`) gains `arbiter: AppInferenceArbiter? = nil`; stored as `arbiter ?? AppInferenceArbiter(client: client)`. **The default is mandatory** or `AppModelTests`, `AppModelLiveMetricsTests`, `AppModelServiceReportingTests` all break at once.
- `launchGeneration` (`:1067-1080`) calls `arbiter.stream(request, owner: .chat)` instead of `client.generate(request)` — same element type, the `for try await` body is untouched.
- `cancel()` (`:1082-1089`) calls `arbiter.cancel(owner: .chat)` instead of `client.cancel()`.
- `beginLoad` (`:409-448`), `cancelLoad` (`:454-467`), `unloadModel` (`:470-483`) and the implicit unload in `setModelURL` (`:378-386`) wrap the `lifecycle` call in `arbiter.withExclusiveSession(reason:)`, then `arbiter.publishLoadedSession(...)`.
- A new observed `inferenceActivity` (display only, fed from `arbiter.activityStream` via `Task { @MainActor in }`) drives the "waiting for an API request" HUD state. **The capability gates `canRun`/`canLoadModel`/`canUnloadModel` are NOT changed** — `withExclusiveSession` makes the drain correct regardless of what the button was showing a moment ago, which avoids a TOCTOU mirror of actor state.

### 4.10 `Sources/MferenceApp/Core/State/AppModelLoadState.swift`

`AppGenerationPhase` (`:45-50`) gains `case queued` for "waiting behind an API request".

### 4.11 `Sources/MferenceApp/Core/State/AppPresentationState.swift`

`AppPresentationSnapshot` gains `queuedBehindAPIRequest: Bool`; `resolve(_:)` (`:81`) returns "Waiting for an API request" / `.active` / `showsActivity: true` for it, so the string is unit-tested outside SwiftUI.

### 4.12 `Sources/MferenceApp/Mac/App/MferenceMacApp.swift`

`init` (`:34-37`) hoists the client and the arbiter into locals and hands the **same instances** to both owners:

```swift
let client = DecodeServiceInferenceClient()
let arbiter = AppInferenceArbiter(client: client)
_model  = State(initialValue: AppModel(client: client, arbiter: arbiter,
                                       settingsPersistenceEnabled: true))
_server = State(initialValue: APIServerModel(arbiter: arbiter))
```

This is the entire "no second model copy" mechanism. `AppModel.client` is `private let` with no accessor (`:57`), so sharing at construction is the only clean route.

Add `CommandMenu("Server")` to `.commands` (`:50-83`) — Start / Stop / Copy Base URL, plus ⌘1 / ⌘2 pane switching — each `.disabled(...)` off `APIServerModel` flags the way the existing four menus key off `AppModel`.

### 4.13 `Sources/MferenceApp/Mac/App/RootView.swift`

`primaryContent` (`:75-91`) — today a single boolean — becomes the install gate wrapping a switch:

```swift
@AppStorage(AppPrimaryPane.storageKey) private var primaryPaneRawValue = AppPrimaryPane.chat.rawValue

if model.requiresModelInstallation { ModelInstallView(model: model) }
else {
    switch AppPrimaryPane.resolve(primaryPaneRawValue) {
    case .chat:      conversationView
    case .apiServer: APIServerView(model: model, server: server)
    }
}
```

`StatusHUDView` stays attached as the same `.safeAreaInset(edge: .top)` over both branches. Add `.animation(.smooth(duration: 0.22), value: primaryPaneRawValue)` to the stack at `:59-62`. The log list inherits the `.transaction { if model.isRunning { $0.animation = nil } }` suppression at `:63-67` — correct: rows should pop in, not animate, while decoding.

### 4.14 `Sources/MferenceApp/Mac/Generation/ChatSidebarView.swift`

A segmented `Picker` over `AppPrimaryPane.allCases` in the footer beside the existing appearance picker.

### 4.15 `Sources/MferenceApp/Mac/Diagnostics/StatusHUDView.swift`

Take the active pane in and swap the chat-specific middle region (`selectedChatTitle` + tok/s) for the listening address on the API Server pane. **Also: a small accent `network` badge visible from the Chat pane whenever the server is listening**, with a click target that jumps to the pane. An open unauthenticated port must never be invisible because the user happens to be on the other tab.

---

## 5. The `ServerInferenceBackend` conformance, gap by gap

`ServerInferenceBackend` has exactly two requirements (`ServerInference.swift:42-50`), plus three implicit ones from `HTTPServer`: everything that can reject must be in `prepare` (`:43-46`); `prepare` runs **outside** the coordinator gate so it must be safe during another generation (`ServerCoordinator.run` calls `render` before `acquire`, `:81-95`; `HTTPServer.swift:249-252`); and `onEvent` is a synchronous `@Sendable` callback receiving incremental content.

### `prepare(_:) async throws -> PreparedGeneration`

Runs entirely app-side with **no IPC**. This is mandatory, not a preference: touching the pipe here collides with a concurrent generation. It needs only `MFTokenizer`, which the app already loads (`DecodeServiceInferenceClient.swift:39-40`) and which is cached per source by a process-wide actor (`Tokenizer.swift:928-934`), so it is pure CPU after the first call.

1. `guard let session = await arbiter.loadedSession else { throw ServerRequestError.unavailable("no model is loaded") }` → **503**.
2. `ArbiterRequestTranslation.translate(request, session:, policy:)` — the pure function. Every rejection is a `ServerRequestError` so it carries a real code; anything else would flatten to the opaque 500 at `HTTPServer.swift:459-462`.
3. `AppGenerationContextWindow.prepareWithReport(translated, tokenizer:)` (`AppGenerationContextWindow.swift:106-124`) for the fit and the exact prompt-token count. **If `removedMessages` is non-empty → 400 `context_length_exceeded`, not silent truncation.** This is a deliberate divergence from the chat path, which trims quietly (`RealInferenceClient.swift:318-320`); an OpenAI client expects a status code, not a shortened conversation. Surface the divergence in the pane's caption.
4. `PreparedGeneration(request: request, payload: plan)` where `plan` carries the `AppGenerationRequest`, prompt token count, stop strings and substitution list.

### `generate(_:onEvent:) async throws -> ServerCompletion`

1. Recover `plan` from `prepared.payload`.
2. `let ticket = try await arbiter.acquire(.http(id), timeout: busyTimeout)`. On timeout throw **`ServerRequestError.queueFull` → 429** (`HTTPServer.swift:456-457`) — an existing public case with exactly the right semantics, no new status mapping.
3. **Re-validate the session pin after acquiring**, since `prepare` ran outside the gate and the model may have been unloaded or reloaded in between → `.unavailable` → 503.
4. Consume `arbiter.stream(plan.request, owner: .http(id), onTextDelta:)`. Each delta goes through a `StreamingStopMatcher` (`StreamingStopMatcher.swift:3`, `push` at `:12`, `finish` at `:28`) built from `request.generationConfig.stopStrings`; emit only the safe prefix via `onEvent(.content(...))`. Calling `onEvent` from the reader thread is fine — the HTTP layer hops to the event loop itself (`HTTPServer.swift:636-642`).
5. On a stop-string match, `arbiter.cancel(owner: .http(id))`. Safe **only** because the cancel is lease-scoped.
6. Terminal event → `ServerCompletion`.
7. `defer { arbiter.release(ticket) }`.

### Gap resolution table

| # | Gap | Resolution |
|---|---|---|
| G1 | **Prompt rendering / `promptIDs`** — the server normally renders and passes token IDs; the wire has no field for them and the service re-renders (`RealInferenceClient.swift:321-323`). | App-side render for *measurement only*; `promptIDs` left empty (nothing in `HTTPServer.swift` reads it). Prompt is rendered twice — accepted cost. |
| G2 | **Roles** — validator emits system/developer/user/assistant/tool (`Tokenizer.swift:370`); the wire has system/user/assistant only (`AppGenerationMessage.Role`, `AppGenerationRequest.swift:4-8`). | `.developer` → folded to `system` for **every** dialect (the validator folds only for non-gemma). Multiple/non-leading system messages merged into one leading message (`validate()` requires system at index 0, `AppGenerationRequest.swift:92-95`). `.tool` role or assistant `toolCalls` → **400 `tool_messages_not_supported`**. |
| G3 | **Tool calls** — token-ID-driven parsing (`StructuredAssistantDecoder.swift:58-80`); service passes `allowedTools: []` and throws (`RealInferenceClient.swift:341-345`). | **400 `tools_not_supported`** in `prepare` when `!request.tools.isEmpty`. Empty `tools: []` is accepted (many clients send it by default). **Deferred to phase 2 — requires a wire change.** |
| G4 | **Usage** | `promptTokens` from the terminal frame's `promptTokenCount` (authoritative — measured after the service's own trim), falling back to `prepare`'s count when nil; `completionTokens` from `tokenCount`. **`cachedTokens` always 0**, honestly. |
| G5 | **`finish_reason`** — `AppStopReason` has `.cancelled`, OpenAI has no slot (`AppDiagnostics.swift:4-12`). | `.maxTokens` → `"length"`; `.eos`/`.endOfTurn`/`.stopString` → `"stop"`; `.toolCalls` → `"stop"` (unreachable); `.failed` → rethrow. `.cancelled` → `"stop"` **only** when we cancelled for a stop string; otherwise throw `ServerRequestError.unavailable` → 503, so a truncated answer is never dressed up as a success. |
| G6 | **Stop strings** — no wire field, no per-request early stop. | Applied app-side by truncation, so the *response* is correct; early termination via the lease-scoped cancel, so tokens are not wasted either. Note: the terminal frame is then `.cancelled` with partial diagnostics, so counts/timings on a stop-string completion are approximate. |
| G7 | **Streaming granularity** | Fixed by the `onTextDelta` hook (§4.6). Cadence is the service's ~100 ms outbox coalescing (`DecodeServiceOutbox.swift:63-108`) — roughly **10 SSE chunks/sec**. Honest, not token-by-token. **Not fixable app-side.** |
| G8 | **Concurrency / single reader** | The arbiter is the sole caller of the client — generate, `ensureLoaded`, `unload`. Structural, not conventional. |
| G9 | **`prepare` during `generate`** | Pure tokenizer, no IPC. Enforced by a test that runs `prepare` concurrently with a fake in-flight generation and fails if the client is touched. |
| G10 | **Cancellation** — `case cancel` has no payload (`DecodeProtocol.swift:161`). | Global *is* scoped, because the arbiter guarantees exactly one runner. A queued owner is dequeued without touching the wire. Plus the `onTermination` fix (§4.6). |
| G11 | **Session key / sampling mode** | Primary: relax `forceLogitsHead` to compatibility (§4.7). Residual (session loaded greedy-only, request sampled): **400 `sampling_mode_mismatch`** naming the loaded mode, by default. `pinToSession` is an **opt-in** toggle that rewrites temperature/top_p/top_k and logs one substitution row per request. `maxContextTokens` and `runtimeOptions` are always taken from the loaded session, never the request — `Entry.swift:62-68` fails the generation outright when options differ. |
| G12 | **Prompt cache** | Not achievable. Ship with `.off` semantics; say so in the pane. |
| G13 | **Activity log** | Injectable sink in `MferenceServerCore` (§7). |
| G14 | **Package graph** | Bridge target (§4.1). |
| — | **`seed`** — in `GenerationConfig` (`OpenAIModels.swift:325`), no wire field. | **400 `unsupported_parameter`**. Better than lying. |
| — | **`repetition_penalty`** — server accepts `> 0` (`OpenAIModels.swift:293-296`), app requires `>= 1` (`AppGenerationRequest.swift:124). | `< 1` → **400** in `prepare`, instead of a deep opaque 500. |
| — | **Assistant prefill** — `validate()` requires `messages.last?.role == .user` (`:89`). | **400 `assistant_prefill_not_supported`**. |
| — | Already handled upstream: `n>1`, `logprobs`, presence/frequency penalties, `tool_choice: required` (`OpenAIModels.swift:261-277`). | No work. |

---

## 6. Concurrency policy — decisive

**One generation at a time, always. The arbiter owns the pipe. Chat has priority admission. Nothing preempts a request that is already decoding.**

**HTTP request arrives while the chat UI is generating.** It waits on the lease inside `generate`. It is not left on a silent socket: `ServerCoordinator` fires `onQueued` → `startStream` writes the SSE head and schedules the `": ping"` heartbeat every 5 s (`HTTPServer.swift:225-238`, `:432-437`). After `busyTimeout` (default 120 s) it throws `queueFull` → **429**. Other HTTP requests queue behind it in `ServerCoordinator` up to `queueLimit`, then 429. Every wait and every 429 is a log row.

**Chat generation starts while an HTTP request is in flight.** `AppModel.run()` → `arbiter.stream(request, owner: .chat)`. Chat is enqueued **at the head**, ahead of any queued HTTP, but does **not** interrupt the decoding request. The HUD shows `AppGenerationPhase.queued` → "Waiting for an API request to finish", and the API Server pane offers an explicit, confirmed, logged **Cancel active request**. Rationale: preempting on a committed SSE stream can only be reported in-band (`HTTPServer.swift:465-485`), which many OpenAI clients ignore — a truncated answer would be silently reported as a success. Killing a remote client's work should be a deliberate act, not a side effect of pressing Send.

**⌘. while chat is queued** cancels only chat's queued lease — it does not reach the wire and does not touch the HTTP request. That is precisely what the lease buys: today `AppModel.cancel()` calls the global `client.cancel()` (`:1088`) and would kill the remote client's generation.

**HTTP client disconnects mid-stream.** `channelInactive` cancels `activeTask` (`HTTPServer.swift:171-175`) → the backend sees cancellation → `arbiter.cancel(owner: .http(id))`. Queued → dequeued silently; decoding → the global cancel, correctly targeted.

**User unloads / reloads the model.** `arbiter.withExclusiveSession`: stop admitting (new `prepare` → 503) → cancel the active lease → **await its terminal frame** (mandatory: `unload` reads the same handle at `DecodeServiceInferenceClient.swift:66`, and a stolen frame is a hard sequence error at `:104-108`) → fail queued waiters with 503 → run `unload` → publish `loadedSession = nil`. **The server keeps listening.** `/health` stays 200, `/v1/models` still lists the pinned id, `/v1/chat/completions` returns 503. The pane reads "Listening — no model loaded". This matches LM Studio.

**Stop button.** Same drain **first**, then `server.shutdown()`. This ordering is load-bearing: `ChildChannelRegistry.closeAll()` cancels *and awaits* every in-flight task (`HTTPServer.swift:614-628`), and those tasks sit inside the uninterruptible `handle.read(upToCount:)` of `DecodeFrameCodec.readExactly` (`DecodeProtocol.swift:302-311`). Calling `shutdown()` first would block for the remainder of a generation.

**Start/stop serialization.** `MferenceHTTPServer` is **single-use** — `shutdown()` calls `group.shutdownGracefully()` (`:90`) on the group its own default init created (`:26`) — and `start` is unguarded, overwriting `self.channel` (`:61`). `APIServerModel` serializes both operations behind one `Task` chain and always constructs a **fresh** server per start. The button is disabled while `.starting`/`.stopping`.

**Model-directory change while listening** → auto-stop with a logged reason: `modelID` and `chatDialect` are frozen into `MferenceHTTPServer.init` (`:23-24`) and read at `:186-192`. A *reload of the same directory* does **not** stop the server.

---

## 7. Activity log

**Why a `MferenceServerCore` change is unavoidable.** A backend wrapper never sees: the 413 oversize path (`HTTPServer.swift:162`), `GET /health` (`:184`), `GET /v1/models` (`:186`), the 415 content-type rejection (`:197`), 405 (`:204`), 404 (`:208`), the JSON-decode / validation 400-404s (`:294-301`), or the 429 from `ServerCoordinator.claim` (`ServerInference.swift:112-113`) — every one completes without calling the backend, and method/path never reach it at all. And `ServerLog` is an internal enum whose only output is `FileHandle.standardError` (`ServerLog.swift:10`, `:51-54`), with `init` accepting no logger (`:21-33`). The change is small and additive.

**What is recorded, per row:** timestamp · method · path · HTTP status · requested model (captured *before* validate throws) · streaming flag · prompt / cached / completion tokens · duration · `finish_reason` · detail · severity. Plus bridge-authored rows only the backend knows: *queued behind chat*, *temperature pinned*, *stop string matched*, *model not loaded*, *cancelled by unload*, *preempted*. Both go through the **same sink**, so ordering is preserved on one channel.

**Where it lives:** `APIServerLogSink` in the bridge — a `Sendable` final class over `Mutex<Deque<APIServerLogEntry>>`.

**How it reaches SwiftUI:** `record` does a mutex append and, if no flush is pending, schedules **exactly one** `Task { @MainActor in server.drainLog() }` — the hop-back idiom already used at `AppModel.swift:432-436`. Coalescing is a **shipping requirement, not a polish item**: `record` runs on the NIO event loop at several rejection sites, and a misconfigured client hammering 404s would otherwise produce hundreds of `@Observable` MainActor mutations per second and visibly stutter the UI. Batch on a ~100 ms tick; never format or block inside `record`.

**Retention:** 500 entries, drop-oldest, with a `droppedBefore` counter so the list renders "… 37 entries dropped" rather than silently lying. Clear and Copy All in the toolbar; a filter toggle to hide `/health` and `/v1/models` polling noise (many clients poll `/v1/models` on every request and would otherwise make the log unreadable within a minute).

**Failed rows expand** to show the underlying `detail`. This is the only place the real cause of a 500 is visible — the wire body is a fixed generic envelope (`HTTPServer.swift:459-462`) — and it is the whole point of having a log pane. It matters doubly here because the decode service is launched with `process.standardError = FileHandle.nullDevice` (`DecodeServiceInferenceClient.swift:225`).

---

## 8. Security defaults

- **Bind default: `.loopback` → `127.0.0.1`** (`ServerArguments.swift:112`). Unchanged from the CLI.
- **There is no authentication. None.** The route table (`HTTPServer.swift:178-211`) has no API key, no bearer token, no CORS, and no TLS. The only defences are the loopback default and `ServerBindMode`'s refusal to widen: `.tailnet` accepts exactly one address inside `100.64.0.0/10` and fails rather than guessing (`ServerArguments.swift:120-153`). There is no wildcard and no LAN option, by design.
- **`.tailnet` requires an explicit in-pane confirmation the first time**, and shows a persistent amber warning row while selected: *"Anyone on your tailnet can use this model. There is no authentication."*
- **No auto-start on launch** by default. Binding an unauthenticated endpoint should be a deliberate per-session act. The "Start when the model loads" toggle exists but ships **off**.
- **The listening state is visible from both panes** (§4.15), and Stop is reachable from the `Server` command menu.
- Request bodies are capped at 1 MiB (`HTTPServer.swift:9`, enforced `:152-157` → 413) and per-client concurrency is bounded only by `queueLimit`. Both are surfaced in the pane's limits disclosure.
- `bindMode.host()` is resolved **off the MainActor** — `.tailnet` shells out to `tailscale ip -4` (`ServerArguments.swift:159-181`, no shell, nothing interpolated). Its thrown `ServerArgumentError.description` ("tailscale reported 2 IPv4 addresses; refusing to guess which to bind") becomes the pane's inline error verbatim.
- **Bind failures must read like English.** Catch the NIO bind error and render "port 8080 is already in use" — plausible in practice, since the standalone `MferenceServer` executable ships (`Package.swift:19`) and defaults to 8080.
- Persistence: `@AppStorage("Mference.apiServer.port" / ".bindMode" / ".queueLimit" / ".autoStart" / ".samplingPolicy")`, matching the sidebar and appearance prefs (`RootView.swift:9-12`; `MferenceMacTheme.swift:9`). **Not `MacAppSettings`** — it is stored in the model directory's *parent* (`MacAppSettings.swift:28-32`) so the port would follow the model folder; `isValid()` demands an exact version match and any failure **deletes the file and resets every user's temperature/topK/context** (`:18`, `:41-45`); and `persistSettings()` is called from exactly one place, `run()` (`AppModel.swift:963`), so a server-only session that never generates from the UI would never flush.

---

## 9. Staged build order

Each stage is independently testable and independently mergeable.

**Stage 0 — fd 1 hygiene in the decode service (precursor, independent of everything).**
`Sources/Mference/Runtime/Inference/RealForwardRunner.swift` contains **12 live `print()` calls** (`:2645`, `:2649`, `:2654`, `:3373`, `:3498`, `:3499`, `:3500`, `:3969`, `:4069`, `:4869`, `:5540`, `:5547`), several on an ungated Metal command-buffer error path. In the decode service, **stdout *is* the length-prefixed frame stream**, and the app-side reader throws on a sequence gap rather than resyncing (`DecodeServiceInferenceClient.swift:104-108`). At service startup, `dup` fd 1 to a private descriptor for `DecodeFrameCodec` and reopen fd 1 onto stderr or `/dev/null`. Latent bug today; a long-running server multiplies the exposure enormously. *Test: a frame round-trip with an interleaved `print()`.*

**Stage 1 — cancellation regression net, then the `onTermination` narrowing (§4.6).**
Write `DecodeServiceCancelRegressionTests` **first** — today ⌘. works via two overlapping paths (`runTask?.cancel()` tearing down into `onTermination`, *and* an explicit `client.cancel()` at `AppModel.swift:1088`), so narrowing `onTermination` is exactly the change that silently breaks Cancel. Land the test, then the fix.

**Stage 2 — `AppInferenceArbiter` + `AppModel` migration.** No server, no NIO. All tests run in the existing `MferenceAppCoreTests` target against the existing `MockInferenceClient` / `FakeInferenceClient`. Ship this alone and verify the chat UI is byte-identical in behaviour.

**Stage 3 — the session-key relaxation (§4.7), gated on `GreedyHeadEquivalenceTests`.** Independently valuable: it fixes `temperature: 0` for the *chat* UI too.

**Stage 4 — `AppInferenceDeltaStreaming` (§4.6, §4.8).** ~6 lines. Test: a fake service emitting N snapshots yields N deltas un-throttled, while the existing `AppInferenceEvent` throttling is unchanged.

**Stage 5 — `MferenceServerCore` edits: `ServerLogSink`, `ServerRequestError.unavailable`, `PreparedGeneration.payload`, public counters.** All additive. `Tests/MferenceServer/ServerLogSinkTests.swift` proves every previously-silent site now emits, and that `main.swift`'s stderr output is unchanged.

**Stage 6 — the bridge target: `ArbiterRequestTranslation` (pure, table-driven) then `ArbiterInferenceBackend` driven through a real `MferenceHTTPServer` on `port: 0` with a fake client.** At the end of this stage the feature works headlessly, with no UI at all — `curl` against a test-spun server returns real SSE.

**Stage 7 — `APIServerModel` + log sink + presentation state.** Start/stop lifecycle tests, ring-buffer tests, presentation tests.

**Stage 8 — SwiftUI: `AppPrimaryPane`, `RootView` switch, `APIServerView`, log list, HUD badge, `CommandMenu("Server")`.**

---

## 10. Explicitly out of scope for the first cut

1. **Tool calling** (`tools`, `tool_choice`, `tool` role, assistant `tool_calls`) — 400. Impossible without a wire change; requires the phase-2 topology.
2. **Prompt caching / `cached_tokens`** — always 0, every request re-prefills. Impossible without moving the server into the decode service.
3. **`seed` / deterministic replay** — 400. No wire field.
4. **Token-by-token SSE** — capped at ~10 chunks/sec by the service's 100 ms coalescing.
5. **Parallel decoding.** One generation at a time, forever, on this transport. Even in phase 2 it would need N KV caches at gigabytes each.
6. **Any authentication, API key, TLS, or CORS.** Not added; the risk is managed by loopback-default and the tailnet confirmation.
7. **`/v1/completions`, `/v1/embeddings`, model load/unload endpoints, model switching over HTTP.** Only `/health`, `/v1/models`, `/v1/chat/completions` exist (`HTTPServer.swift:178-211`), and `model` in the body can be validated but never switched.
8. **Multimodal content parts, `logit_bias`, `min_p`, `response_format` / JSON mode, `echo`, `n > 1`, `logprobs`, presence/frequency penalties.**
9. **Assistant prefill / continuation** (trailing assistant message) — 400.
10. **`reasoning_content`.** Thinking text never crosses the wire; `StructuredAssistantDecoder` strips it inside the service.
11. **Reload-on-demand to satisfy a mismatched request.** A request that genuinely cannot be served by the loaded session gets a 400 explaining what to reload — the server never reloads a 20 GB model on an HTTP request's behalf.
12. **Multi-window / multiple concurrent servers.** Single `Window` scene (`MferenceMacApp.swift:41`), one server.

---

### Phase 2, recorded so nobody retrofits it the wrong way

Tools and the prompt cache require moving `MferenceHTTPServer` **into `MferenceDecodeService`**, because tool blocks are discriminated by special token ID (`StructuredAssistantDecoder.swift:58-80`) and the cache's resume hooks (`RawCompletionStart.resume`, `kvPosition`, `kvBackedTokenIDs`) never cross the pipe. That migration additionally needs: an outbound frame enum, an app-side demux reader, a mutex-serialized fd-1 writer, and a `MferenceDecodeServiceCore` split so any of it can be tested. Do not attempt these app-side.