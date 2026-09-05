import Foundation
import Testing
import MferenceAppCore
import MferenceServerBridge
import MferenceServerCore

@Suite("API server presentation state")
struct APIServerPresentationStateTests {
    private static let listening = APIServerRunState.listening(host: "127.0.0.1", port: 8080)

    @Test func stoppedOffersStartOnlyWithAModel() {
        let ready = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: .stopped, hasLoadedSession: true))
        #expect(ready.label == "Stopped")
        #expect(ready.primaryAction == .start)
        #expect(ready.severity == .neutral)
        #expect(!ready.showsActivity)

        let noModel = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: .stopped, hasLoadedSession: false))
        #expect(noModel.label == "Stopped")
        #expect(noModel.detail == "Load a model to serve it.")
        #expect(noModel.primaryAction == nil)
    }

    @Test func transitionsShowActivityAndNoActions() {
        for state in [APIServerRunState.starting, .stopping] {
            let resolved = APIServerPresentationState.resolve(
                APIServerSnapshot(runState: state, hasLoadedSession: true))
            #expect(resolved.severity == .active, "\(state)")
            #expect(resolved.showsActivity, "\(state)")
            #expect(resolved.primaryAction == nil, "\(state)")
            #expect(resolved.secondaryAction == nil, "\(state)")
        }
        #expect(APIServerPresentationState.resolve(
            APIServerSnapshot(runState: .starting, hasLoadedSession: true)).label == "Starting")
        #expect(APIServerPresentationState.resolve(
            APIServerSnapshot(runState: .stopping, hasLoadedSession: true)).label == "Stopping")
    }

    @Test func failureCarriesTheMessageAndRetries() {
        let failed = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: .failed("Port 8080 is already in use on 127.0.0.1."),
                              hasLoadedSession: true))
        #expect(failed.label == "Could not start")
        #expect(failed.detail == "Port 8080 is already in use on 127.0.0.1.")
        #expect(failed.severity == .error)
        #expect(failed.primaryAction == .start)

        let noModel = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: .failed("Load a model before starting the server."),
                              hasLoadedSession: false))
        #expect(noModel.primaryAction == nil)
    }

    @Test func listeningIdleIsSuccessWithStop() {
        let idle = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: Self.listening, hasLoadedSession: true))
        #expect(idle.label == "Listening on 127.0.0.1:8080")
        #expect(idle.detail == "http://127.0.0.1:8080/v1")
        #expect(idle.severity == .success)
        #expect(idle.primaryAction == .stop)
        #expect(idle.secondaryAction == nil)
        #expect(!idle.showsActivity)
    }

    @Test func listeningWithoutAModelWarns() {
        let resolved = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: Self.listening, hasLoadedSession: false))
        #expect(resolved.label == "Listening · no model loaded")
        #expect(resolved.detail?.contains("503") == true)
        #expect(resolved.severity == .warning)
        #expect(resolved.primaryAction == .stop)
    }

    @Test func servingARequestOffersCancel() {
        let serving = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: Self.listening, hasLoadedSession: true,
                              activeOwner: .http(UUID()), queuedHTTPCount: 2))
        #expect(serving.label == "Serving a request · 2 waiting")
        #expect(serving.severity == .active)
        #expect(serving.showsActivity)
        #expect(serving.primaryAction == .stop)
        #expect(serving.secondaryAction == .cancelActiveRequest)

        let single = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: Self.listening, hasLoadedSession: true,
                              activeOwner: .http(UUID())))
        #expect(single.label == "Serving a request")
    }

    @Test func chatGeneratingIsReportedWithoutCancel() {
        let resolved = APIServerPresentationState.resolve(
            APIServerSnapshot(runState: Self.listening, hasLoadedSession: true,
                              activeOwner: .chat, queuedHTTPCount: 1))
        #expect(resolved.label == "Listening · chat is generating · 1 waiting")
        #expect(resolved.severity == .active)
        #expect(resolved.primaryAction == .stop)
        #expect(resolved.secondaryAction == nil)
    }

    @Test func runStateHelpers() {
        #expect(Self.listening.isListening)
        #expect(!APIServerRunState.stopped.isListening)
        #expect(APIServerRunState.starting.isTransitioning)
        #expect(APIServerRunState.stopping.isTransitioning)
        #expect(!APIServerRunState.failed("x").isTransitioning)
        #expect(!Self.listening.isTransitioning)
    }

    @Test func bindOptionsNameBothModesAndOnlyTailnetIsExposed() {
        #expect(APIServerBindOption.all.map(\.mode) == [.loopback, .tailnet])
        #expect(!APIServerBindOption.option(for: .loopback).isExposedBeyondThisMac)
        #expect(APIServerBindOption.option(for: .tailnet).isExposedBeyondThisMac)
        #expect(APIServerBindOption.exposureWarning.contains("no authentication"))
        #expect(Set(APIServerBindOption.all.map(\.id)).count == 2)
    }

    @Test func modelIdentityTableMatchesTheStandaloneServer() {
        #expect(APIServerModelIdentity.modelID(for: .gemma4) == "gemma-4-26b-a4b-it")
        #expect(APIServerModelIdentity.modelID(for: .qwen36) == "qwen3.6-35b-a3b")
        #expect(APIServerModelIdentity.modelID(for: .qwen38) == "qwen3.8-27b-4bit")
        #expect(APIServerModelIdentity.modelID(for: .deepseekV4Flash) == "deepseek-v4-flash-2bit-dq")
        #expect(APIServerModelIdentity.modelID(for: .inklingSmall) == "inkling-small-4bit")
        #expect(APIServerModelIdentity.modelID(for: .maple) == "maple-preview-2bit-mlx")
    }
}
