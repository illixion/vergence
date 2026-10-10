/*
 Vergence - big-endian reading with bounds checks

 Disc structures (MPLS, CLPI, UDF) are read from untrusted files: every read
 is checked and a short or malformed structure throws instead of trapping.
 */

import Foundation

public struct FormatError: Error, CustomStringConvertible, Equatable {
    public var description: String
    public init(_ description: String) { self.description = description }
}

struct BinaryReader {
    let data: Data
    /// Offset from `data.startIndex`.
    var offset: Int

    init(_ data: Data, offset: Int = 0) {
        self.data = data
        self.offset = offset
    }

    var remaining: Int { data.count - offset }

    mutating func skip(_ n: Int) throws {
        guard n >= 0, n <= remaining else { throw FormatError("structure ends early at \(offset)") }
        offset += n
    }

    mutating func bytes(_ n: Int) throws -> Data {
        guard n >= 0, n <= remaining else { throw FormatError("structure ends early at \(offset)") }
        let start = data.startIndex + offset
        offset += n
        return data[start ..< start + n]
    }

    mutating func u8() throws -> UInt8 { try bytes(1).first! }

    mutating func u16() throws -> UInt16 {
        let b = try bytes(2)
        return UInt16(b[b.startIndex]) << 8 | UInt16(b[b.startIndex + 1])
    }

    mutating func u32() throws -> UInt32 {
        let b = try bytes(4)
        return b.reduce(0) { $0 << 8 | UInt32($1) }
    }

    mutating func ascii(_ n: Int) throws -> String {
        String(decoding: try bytes(n), as: UTF8.self)
    }

    /// Little-endian, for UDF.
    mutating func u16le() throws -> UInt16 {
        let b = try bytes(2)
        return UInt16(b[b.startIndex]) | UInt16(b[b.startIndex + 1]) << 8
    }

    mutating func u32le() throws -> UInt32 {
        let b = try bytes(4)
        return b.reversed().reduce(0) { $0 << 8 | UInt32($1) }
    }

    mutating func u64le() throws -> UInt64 {
        let b = try bytes(8)
        return b.reversed().reduce(0) { $0 << 8 | UInt64($1) }
    }

    /// A reader over `length` bytes from here, advancing past them.
    mutating func sub(_ length: Int) throws -> BinaryReader {
        BinaryReader(try bytes(length))
    }
}
