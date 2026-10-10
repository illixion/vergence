// swift-tools-version: 6.2
import PackageDescription

// Layers (see README → Layout):
// - VergenceCore: stereo layouts, warp settings and depth maths (anchoring,
//   depth budget, histograms). Foundation only, no dependencies.
// - VergenceDisc: Blu-ray reading and MVC decoding, Foundation only, so it
//   builds wherever Swift does (Apple platforms, Linux, Android, Windows).
// - VergenceApple: the Apple-framework half (CoreVideo pixel buffers,
//   AudioToolbox AC-3). Its sources are wrapped in `#if canImport(...)`, so
//   the target exists and builds empty elsewhere and consumers need no
//   per-platform dependency lists.
// The tests check decoded pictures against ffmpeg's MD5. Apple platforms
// have CryptoKit; elsewhere swift-crypto supplies the same API, fetched only
// there (this is the manifest's host OS, the Oneiros pattern).
var dependencies: [Package.Dependency] = []
var testCrypto: [Target.Dependency] = []
#if !canImport(Darwin)
dependencies.append(.package(url: "https://github.com/apple/swift-crypto.git", "3.0.0" ..< "5.0.0"))
testCrypto.append(.product(name: "Crypto", package: "swift-crypto"))
#endif

let package = Package(
    name: "Vergence",
    platforms: [.macOS(.v14), .iOS(.v26), .visionOS(.v26), .tvOS(.v26)],
    products: [
        .library(name: "VergenceCore", targets: ["VergenceCore"]),
        .library(name: "VergenceDisc", targets: ["VergenceDisc"]),
        .library(name: "VergenceApple", targets: ["VergenceApple"]),
        .executable(name: "vergence", targets: ["VergenceCLI"]),
    ],
    dependencies: dependencies,
    targets: [
        // edge264 (BSD-3), vendored with the patches listed in Vendor/README.md.
        // It is a unity build: edge264.c includes every other source file.
        .target(
            name: "CEdge264",
            exclude: [
                "LICENSE",
                "src/edge264_bitstream.c", "src/edge264_deblock.c", "src/edge264_headers.c",
                "src/edge264_inter.c", "src/edge264_intra.c", "src/edge264_mvpred.c",
                "src/edge264_residual.c", "src/edge264_sei.c", "src/edge264_slice.c",
            ],
            sources: ["src/edge264.c"],
            cSettings: [
                // edge264's asserts abort the process; keep them for debugging only.
                .define("NDEBUG", .when(configuration: .release)),
            ]
        ),
        .target(name: "VergenceCore"),
        .target(name: "VergenceDisc", dependencies: ["CEdge264"]),
        .target(name: "VergenceApple", dependencies: ["VergenceCore", "VergenceDisc"]),
        // The command line's option parsing and output names, apart from the
        // executable so its tests can import them. Foundation only.
        .target(name: "VergenceCommandLine", dependencies: ["VergenceCore", "VergenceDisc"]),
        .executableTarget(name: "VergenceCLI", dependencies: ["VergenceDisc", "VergenceApple", "VergenceCommandLine"]),
        .testTarget(name: "VergenceCoreTests", dependencies: ["VergenceCore"]),
        .testTarget(name: "VergenceDiscTests", dependencies: ["VergenceDisc"] + testCrypto, resources: [.copy("Fixtures")]),
        // Synthetic clips shared by the Apple and command line tests (macOS only).
        .target(name: "VergenceTestSupport", dependencies: ["VergenceApple"], path: "Tests/VergenceTestSupport"),
        .testTarget(name: "VergenceAppleTests", dependencies: ["VergenceApple", "VergenceTestSupport"], resources: [.copy("Fixtures")]),
        // Depends on the executable so `swift test` builds the binary it runs.
        .testTarget(name: "VergenceCLITests", dependencies: ["VergenceCommandLine", "VergenceCLI", "VergenceTestSupport"]),
    ],
    cLanguageStandard: .gnu11
)
