/*
 Vergence - Depth model cache and lazy pre-warming

 Loading a Core ML depth model is the slow part of starting AI 3D: ~60 ms when
 the system has the compiled model hot, seconds the first time, and the first
 inference after a load is slower again while the Neural Engine specializes.
 Every provider used to load its own copy and throw it away with itself, so a
 gallery that released its provider after 20 idle seconds paid it all again on
 the next picture.

 Two pieces:

 - `LoadedDepthModel` / `DepthModelCache`: one loaded `MLModel` per compiled
   model file, shared by every `CoreMLDepthProvider` (stills, real-time video,
   the offline converter). The cache holds models weakly, so a model lives
   exactly as long as a provider or the warmer uses it.
 - `DepthModelWarmer`: keeps a model resident, and warm, while something has
   *intent* to use it. A host takes a `lease` for a role when the user is
   somewhere AI 3D is the default (a tab, an open slideshow) and drops it when
   they leave. The warmer loads lazily and tolerates people flicking between
   places: a load starts only after the lease has been held for `dwell`, and a
   model stays resident for `grace` after the last lease goes, so tab hopping
   neither triggers loads that are never used nor drops and reloads one that is.
   Memory pressure drops what nothing is using.

 Nothing here changes what a provider computes; it only decides when the model
 is already in memory.
 */

import CoreML
import CoreVideo
import Foundation
import Vision

/// A loaded Core ML depth model, shared by the providers made from it.
final class LoadedDepthModel: @unchecked Sendable {
    let url: URL
    let mlModel: MLModel
    let vnModel: VNCoreMLModel
    /// The model's declared image input size, if any.
    let inputSize: (width: Int, height: Int)?
    private let lock = NSLock()
    private var warmedUp = false

    init?(compiledURL: URL) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let mlModel = try? MLModel(contentsOf: compiledURL, configuration: config),
              let vnModel = try? VNCoreMLModel(for: mlModel) else { return nil }
        self.url = compiledURL
        self.mlModel = mlModel
        self.vnModel = vnModel
        self.inputSize = mlModel.modelDescription.inputDescriptionsByName.values
            .compactMap(\.imageConstraint).first
            .map { (width: $0.pixelsWide, height: $0.pixelsHigh) }
    }

    /// One throwaway prediction, so the first real frame doesn't pay for the
    /// Neural Engine's specialization. Once per loaded model; failures are
    /// ignored (the real inference will report its own).
    func warmUp() {
        lock.lock()
        let already = warmedUp
        warmedUp = true
        lock.unlock()
        guard !already,
              let (name, constraint) = mlModel.modelDescription.inputDescriptionsByName
                .first(where: { $0.value.imageConstraint != nil }),
              let image = constraint.imageConstraint else { return }
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, image.pixelsWide, image.pixelsHigh, image.pixelFormatType, nil, &buffer)
        guard let buffer,
              let input = try? MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(pixelBuffer: buffer)]) else { return }
        _ = try? mlModel.prediction(from: input)
    }
}

/// Loaded models by compiled file, held weakly.
final class DepthModelCache: @unchecked Sendable {
    static let shared = DepthModelCache()

    private final class Weak { weak var model: LoadedDepthModel?; init(_ model: LoadedDepthModel) { self.model = model } }
    private let lock = NSLock()
    private var entries: [URL: Weak] = [:]

    /// The model for `compiledURL`, loading it if nothing holds it. Loads are
    /// serialized, so a provider asking while the warmer loads the same model
    /// waits for that load instead of starting a second one.
    func model(for compiledURL: URL) -> LoadedDepthModel? {
        lock.lock()
        defer { lock.unlock() }
        if let model = entries[compiledURL]?.model { return model }
        guard let model = LoadedDepthModel(compiledURL: compiledURL) else { return nil }
        entries = entries.filter { $0.value.model != nil }
        entries[compiledURL] = Weak(model)
        return model
    }
}

/// Keeps the depth model for a role loaded and warm while something intends
/// to use it. See the file comment.
public final class DepthModelWarmer: @unchecked Sendable {
    public static let shared = DepthModelWarmer()

    /// How long a lease must be held before the load starts.
    static let dwell: TimeInterval = 0.8
    /// How long a model stays resident after its last lease is dropped.
    static let grace: TimeInterval = 45

    /// Proof of intent. Dropping it (or calling `release()`) withdraws the
    /// intent; the model then stays resident for `grace`.
    public final class Lease: @unchecked Sendable {
        fileprivate let id = UUID()
        fileprivate let role: DepthModelRole
        fileprivate weak var warmer: DepthModelWarmer?
        fileprivate init(role: DepthModelRole, warmer: DepthModelWarmer) {
            self.role = role
            self.warmer = warmer
        }
        public func release() {
            warmer?.drop(id)
            warmer = nil
        }
        deinit { release() }
    }

    private let queue = DispatchQueue(label: "com.illixion.vergence.depth-warmer", qos: .utility)
    // All below: `queue` only.
    private var leases: [UUID: DepthModelRole] = [:]
    private var resident: [DepthModelRole: LoadedDepthModel] = [:]
    /// The model file (before compiling) each resident model came from.
    private var residentSource: [DepthModelRole: URL] = [:]
    /// A model file that failed to load, not retried until the choice changes.
    private var failedSource: [DepthModelRole: URL] = [:]
    /// No new loads until then, after memory pressure.
    private var pressureHoldUntil = Date.distantPast
    private var loading: Set<DepthModelRole> = []
    private var dwellTimers: [DepthModelRole: DispatchWorkItem] = [:]
    private var graceTimers: [DepthModelRole: DispatchWorkItem] = [:]
    private var pressureSource: DispatchSourceMemoryPressure?
    private var defaultsObserver: NSObjectProtocol?

    private init() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: queue)
        source.setEventHandler { [weak self] in self?.handleMemoryPressure() }
        source.resume()
        pressureSource = source
        // A model chosen in Settings while a lease is held: warm the new one,
        // let the old one go.
        defaultsObserver = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification, object: nil, queue: nil
        ) { [weak self] _ in self?.queue.async { self?.reconcile() } }
    }

    /// Declare that AI 3D for `role` is likely to be needed soon. Cheap to
    /// call and to drop; take it while the user is somewhere that opens AI 3D
    /// by default, hold it for as long as they are there.
    public func lease(role: DepthModelRole) -> Lease {
        let lease = Lease(role: role, warmer: self)
        queue.async {
            self.leases[lease.id] = role
            self.reconcile()
        }
        return lease
    }

    /// Roles currently resident (diagnostics and tests).
    public var residentRoles: Set<DepthModelRole> {
        queue.sync { Set(resident.keys) }
    }

    fileprivate func drop(_ id: UUID) {
        queue.async {
            self.leases[id] = nil
            self.reconcile()
        }
    }

    private func reconcile() {
        let wanted = Set(leases.values)
        for role in [DepthModelRole.realtime, .preprocess] {
            if wanted.contains(role) {
                graceTimers.removeValue(forKey: role)?.cancel()
                // Resident but for a model the Settings choice has moved off.
                if resident[role] != nil, residentSource[role] != CoreMLDepthProvider.modelURL(role: role) {
                    resident[role] = nil
                    residentSource[role] = nil
                }
                // No model installed (or one that won't load): nothing to warm,
                // and nothing to poll for; Settings changes re-run this.
                guard let source = CoreMLDepthProvider.modelURL(role: role), failedSource[role] != source,
                      resident[role] == nil, !loading.contains(role), dwellTimers[role] == nil,
                      Date() >= pressureHoldUntil else { continue }
                let work = DispatchWorkItem { [weak self] in self?.startLoad(role) }
                dwellTimers[role] = work
                queue.asyncAfter(deadline: .now() + Self.dwell, execute: work)
            } else {
                // Intent withdrawn before the dwell ran out: no load at all.
                dwellTimers.removeValue(forKey: role)?.cancel()
                guard resident[role] != nil, graceTimers[role] == nil else { continue }
                let work = DispatchWorkItem { [weak self] in
                    guard let self else { return }
                    self.graceTimers[role] = nil
                    guard !Set(self.leases.values).contains(role) else { return }
                    self.resident[role] = nil
                    self.residentSource[role] = nil
                    VergenceLog.pipeline.debug("Depth model released after grace (\(role.rawValue, privacy: .public))")
                }
                graceTimers[role] = work
                queue.asyncAfter(deadline: .now() + Self.grace, execute: work)
            }
        }
    }

    private func startLoad(_ role: DepthModelRole) {
        dwellTimers[role] = nil
        guard Set(leases.values).contains(role), resident[role] == nil, !loading.contains(role) else { return }
        loading.insert(role)
        let source = CoreMLDepthProvider.modelURL(role: role)
        // The load blocks (seconds the first time), so it runs off the state
        // queue; leases arriving meanwhile still reconcile.
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let began = Date()
            var model: LoadedDepthModel?
            if let source, let url = CoreMLDepthProvider.compiledModelURL(for: source) {
                model = DepthModelCache.shared.model(for: url)
                model?.warmUp()
            }
            _ = VergenceMetal.shared
            let ms = Int(Date().timeIntervalSince(began) * 1000)
            self?.queue.async {
                guard let self else { return }
                self.loading.remove(role)
                // Intent can be gone by the time a long load finishes; the
                // grace timer in `reconcile` then decides how long it stays.
                self.resident[role] = model
                self.residentSource[role] = model == nil ? nil : source
                self.failedSource[role] = model == nil ? source : nil
                if model != nil {
                    VergenceLog.pipeline.info("Depth model warmed (\(role.rawValue, privacy: .public)) in \(ms, privacy: .public) ms")
                }
                self.reconcile()
            }
        }
    }

    private func handleMemoryPressure() {
        // Providers in use keep their model alive through the cache; this only
        // gives up what nothing is using, and stays down until the next lease
        // change or model switch asks again.
        dwellTimers.values.forEach { $0.cancel() }
        graceTimers.values.forEach { $0.cancel() }
        dwellTimers = [:]
        graceTimers = [:]
        if !resident.isEmpty {
            VergenceLog.pipeline.info("Memory pressure: dropping warmed depth models")
        }
        resident = [:]
        residentSource = [:]
        pressureHoldUntil = Date().addingTimeInterval(60)
    }
}
