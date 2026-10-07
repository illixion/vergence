// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "mvc3d",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1), .tvOS(.v17)],
    products: [
        .library(name: "MVC3D", targets: ["MVC3D"]),
        .executable(name: "mvc3d", targets: ["mvc3d-cli"]),
    ],
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
        .target(name: "MVC3D", dependencies: ["CEdge264"]),
        .executableTarget(name: "mvc3d-cli", dependencies: ["MVC3D"]),
        .testTarget(name: "MVC3DTests", dependencies: ["MVC3D"], resources: [.copy("Fixtures")]),
    ],
    cLanguageStandard: .gnu11
)
