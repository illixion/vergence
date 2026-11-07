import Foundation
import Testing
@testable import VergenceCore

/// A checkpoint torch.save wrote (torch 2.7): {"params_ema": OrderedDict(body.0.weight: float32 [2,3],
/// body.0.alpha: float16 [4], view: a transposed view of base, base: float32 [2,3]), "iteration": 7}.
private let fixture = Data(base64Encoded: [
        "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAKFgKAAAA",
        "cGFyYW1zX2VtYXEBY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnECKVJxAyhYDQAAAGJvZHkuMC53ZWlnaHRxBGN0b3JjaC5fdXRp",
        "bHMKX3JlYnVpbGRfdGVuc29yX3YyCnEFKChYBwAAAHN0b3JhZ2VxBmN0b3JjaApGbG9hdFN0b3JhZ2UKcQdYAQAAADBxCFgDAAAA",
        "Y3B1cQlLBnRxClFLAEsCSwOGcQtLA0sBhnEMiWgCKVJxDXRxDlJxD1gMAAAAYm9keS4wLmFscGhhcRBoBSgoaAZjdG9yY2gKSGFs",
        "ZlN0b3JhZ2UKcRFYAQAAADFxEmgJSwR0cRNRSwBLBIVxFEsBhXEViWgCKVJxFnRxF1JxGFgEAAAAdmlld3EZaAUoKGgGaAdYAQAA",
        "ADJxGmgJSwZ0cRtRSwBLA0sChnEcSwFLA4ZxHYloAilScR50cR9ScSBYBAAAAGJhc2VxIWgFKChoBmgHaBpoCUsGdHEiUUsASwJL",
        "A4ZxI0sDSwGGcSSJaAIpUnEldHEmUnEndVgJAAAAaXRlcmF0aW9ucShLB3UuUEsHCNPU8qSvAQAArwEAAFBLAwQAAAgIAAAAAAAA",
        "AAAAAAAAAAAAAAAAFwAMAGFyY2hpdmUvLmZvcm1hdF92ZXJzaW9uRkIIAFpaWlpaWlpaMVBLBwi379yDAQAAAAEAAABQSwMEAAAI",
        "CAAAAAAAAAAAAAAAAAAAAAAAABoANwBhcmNoaXZlLy5zdG9yYWdlX2FsaWdubWVudEZCMwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpa",
        "WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo2NFBLBwg/d3HpAgAAAAIAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABEA",
        "PwBhcmNoaXZlL2J5dGVvcmRlckZCOwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa",
        "WlpaWlpaWmxpdHRsZVBLBwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APgBhcmNoaXZlL2RhdGEvMEZC",
        "OgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAAAAAgD4AAAA/AABA",
        "PwAAgD8AAKA/UEsHCFLl5x8YAAAAGAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgAsAGFyY2hpdmUvZGF0YS8xRkIoAFpa",
        "WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAPgDAAAD/e1BLBwh7vJggCAAAAAgAAABQSwMEAAAICAAAAAAA",
        "AAAAAAAAAAAAAAAAAA4APABhcmNoaXZlL2RhdGEvMkZCOABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa",
        "WlpaWlpaWlpaWlpaWlpaWgAAAAAAAIA/AAAAQAAAQEAAAIBAAACgQFBLBwgXkOeRGAAAABgAAABQSwMEAAAICAAAAAAAAAAAAAAA",
        "AAAAAAAAAA8AKwBhcmNoaXZlL3ZlcnNpb25GQicAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMwpQSwcI",
        "0Z5nVQIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADIAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkRkIu",
        "AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlowODA2NDAxMzIwOTM0MjAxMTE3NzA1MTAxNzYy",
        "MzkzMjIzNzk1MzE0UEsHCNUEH3AoAAAAKAAAAFBLAQIAAAAACAgAAAAAAADT1PKkrwEAAK8BAAAQAAAAAAAAAAAAAAAAAAAAAABh",
        "cmNoaXZlL2RhdGEucGtsUEsBAgAAAAAICAAAAAAAALfv3IMBAAAAAQAAABcAAAAAAAAAAAAAAAAA/wEAAGFyY2hpdmUvLmZvcm1h",
        "dF92ZXJzaW9uUEsBAgAAAAAICAAAAAAAAD93cekCAAAAAgAAABoAAAAAAAAAAAAAAAAAUQIAAGFyY2hpdmUvLnN0b3JhZ2VfYWxp",
        "Z25tZW50UEsBAgAAAAAICAAAAAAAAIU94xkGAAAABgAAABEAAAAAAAAAAAAAAAAA0gIAAGFyY2hpdmUvYnl0ZW9yZGVyUEsBAgAA",
        "AAAICAAAAAAAAFLl5x8YAAAAGAAAAA4AAAAAAAAAAAAAAAAAVgMAAGFyY2hpdmUvZGF0YS8wUEsBAgAAAAAICAAAAAAAAHu8mCAI",
        "AAAACAAAAA4AAAAAAAAAAAAAAAAA6AMAAGFyY2hpdmUvZGF0YS8xUEsBAgAAAAAICAAAAAAAABeQ55EYAAAAGAAAAA4AAAAAAAAA",
        "AAAAAAAAWAQAAGFyY2hpdmUvZGF0YS8yUEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAA8AAAAAAAAAAAAAAAAA6AQAAGFyY2hp",
        "dmUvdmVyc2lvblBLAQIAAAAACAgAAAAAAADVBB9wKAAAACgAAAAeAAAAAAAAAAAAAAAAAFIFAABhcmNoaXZlLy5kYXRhL3Nlcmlh",
        "bGl6YXRpb25faWRQSwYGLAAAAAAAAAAeAy0AAAAAAAAAAAAJAAAAAAAAAAkAAAAAAAAARwIAAAAAAAD4BQAAAAAAAFBLBgcAAAAA",
        "PwgAAAAAAAABAAAAUEsFBgAAAAAJAAkARwIAAPgFAAAAAA==",
].joined())!

@Suite struct TorchCheckpointTests {
    @Test func readsTheStateDictUnderItsWrapper() throws {
        let checkpoint = try TorchCheckpoint(data: fixture)
        let tensors = checkpoint.tensors()
        #expect(Set(tensors.keys) == ["body.0.weight", "body.0.alpha", "view", "base"])
        #expect(tensors["body.0.weight"] == .init(shape: [2, 3], values: [0, 0.25, 0.5, 0.75, 1, 1.25]))
        #expect(tensors["body.0.alpha"] == .init(shape: [4], values: [1.5, -2, 0, 65504]))
        // A strided view reads in its own order, from the storage it shares.
        #expect(tensors["view"] == .init(shape: [3, 2], values: [0, 3, 1, 4, 2, 5]))
        #expect(tensors["base"]?.values == [0, 1, 2, 3, 4, 5])
        #expect(checkpoint.tensors(under: ["params_ema"], prefix: "body.0.").keys.sorted() == ["alpha", "weight"])
    }

    @Test func refusesWhatIsntACheckpoint() {
        #expect(throws: TorchCheckpoint.ReadError.self) { try TorchCheckpoint(data: Data("not a zip".utf8)) }
        var broken = fixture
        broken.replaceSubrange(0 ..< 4, with: [0, 0, 0, 0])
        #expect(throws: (any Error).self) { try TorchCheckpoint(data: broken.prefix(600)) }
    }
}
