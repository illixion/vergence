// swift-tools-version: 6.2
// The Swift half of the interop check (`companion/interop/run.sh`): RAVESDK's
// own RAVECompanion code against the Rust port. Not part of any build; it
// expects RAVESDK beside vergence (`~/Projects/RAVESDK`).
import PackageDescription

let package = Package(
    name: "CompanionInterop",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "../../../../RAVESDK")],
    targets: [
        .executableTarget(
            name: "CompanionInterop",
            dependencies: [.product(name: "RAVECompanion", package: "RAVESDK")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
