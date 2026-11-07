/*
 Vergence - which depth models' weights restrict their use

 Vergence is MIT and ships no weights. Depth Anything V2 Small, the model
 everything here is built and tuned around, is Apache-2.0. The larger
 relatives are not: Depth Anything V2 Base and Large and Video Depth Anything
 Base and Large are CC-BY-NC-4.0 (non-commercial use only), and Depth Pro's
 weights come under Apple's machine-learning research licence. A tool may
 take one only when it was built to (the command line's
 `VERGENCE_NONCOMMERCIAL_MODELS` condition), and says the licence when it does.

 It lives with the command line, not in VergenceApple, so an app that links
 Vergence carries none of these names unless it chooses to.

 Recognised by the model's file name (conversions keep the family and size
 in it) and by a licence its Core ML metadata states; a conversion's metadata
 alone can't be trusted, since a converter may copy the Small model's.
 */

import Foundation

public enum DepthModelLicence {
    public struct Restriction: Sendable, Equatable {
        public let name: String
        public let url: String
        public let note: String
    }

    static let ccByNC = Restriction(
        name: "CC-BY-NC-4.0", url: "https://creativecommons.org/licenses/by-nc/4.0/",
        note: "non-commercial use only; credit Depth Anything V2 (Yang et al., 2024)")
    static let appleResearch = Restriction(
        name: "Apple ML Research Model licence", url: "https://huggingface.co/apple/DepthPro",
        note: "research and non-commercial use only")

    /// The restriction a model's weights carry, or nil when none is known
    /// (Depth Anything V2 Small, Apache-2.0).
    /// - Parameter metadataLicence: the licence its metadata states, if read.
    public static func restriction(modelName: String, metadataLicence: String? = nil) -> Restriction? {
        let name = modelName.lowercased().replacingOccurrences(of: "-", with: "").replacingOccurrences(of: "_", with: "")
        if name.contains("depthpro") { return appleResearch }
        let family = name.contains("depthanything") || name.contains("dav2") || name.contains("vda")
        if family, ["base", "large", "giant", "vitb", "vitl", "vitg"].contains(where: name.contains) { return ccByNC }
        if let licence = metadataLicence?.lowercased(),
           licence.contains("nc") && licence.contains("cc") || licence.contains("non-commercial")
            || licence.contains("noncommercial") || licence.contains("research") {
            return Restriction(name: metadataLicence ?? "", url: "", note: "non-commercial or research use only")
        }
        return nil
    }

    /// The same for a model on disk (`.mlpackage` or `.mlmodelc`), reading a
    /// compiled model's metadata when it has one.
    public static func restriction(forModelAt url: URL) -> Restriction? {
        var metadataLicence: String?
        let metadata = url.appendingPathComponent("metadata.json")
        if let data = try? Data(contentsOf: metadata),
           let entries = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            metadataLicence = entries.first?["license"] as? String
        }
        return restriction(modelName: url.deletingPathExtension().lastPathComponent, metadataLicence: metadataLicence)
    }
}
