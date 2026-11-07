/*
 Vergence - PyTorch checkpoints, read without Python

 A `.pth` file as `torch.save` writes it since PyTorch 1.6: a zip archive
 (entries stored, not compressed) holding `archive/data.pkl`, a pickle of the
 state dict, and one raw little-endian file per tensor storage
 (`archive/data/<key>`). This reads the tensors out by name, which is all the
 upscaler needs to build its network from the weights its authors published:
 Vergence ships no weights and runs no downloaded code. The pickle is read by
 a small interpreter that knows only what a state dict uses; anything else (a
 class it doesn't know is reduced to an opaque value) never runs.
 */

import Foundation

public struct TorchCheckpoint: Sendable {
    public struct Tensor: Sendable, Equatable {
        public var shape: [Int]
        /// Row-major (contiguous) values, whatever the stored type was.
        public var values: [Float]
        public var count: Int { shape.reduce(1, *) }
    }

    public struct ReadError: Error, CustomStringConvertible, Equatable {
        public let description: String
        init(_ text: String) { description = text }
    }

    /// Every tensor of the state dict, by its name ("body.0.weight"). A
    /// checkpoint that wraps its state dict ({"params_ema": {...}}) has its
    /// names unprefixed: `tensors(under:)` picks a wrapper.
    public let wrapped: [String: [String: Tensor]]

    public init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url, options: .mappedIfSafe))
    }

    public init(data: Data) throws {
        let archive = try ZipEntries(data)
        guard let pickleName = archive.names.first(where: { $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }) else {
            throw ReadError("not a PyTorch checkpoint (no data.pkl in the archive)")
        }
        let prefix = String(pickleName.dropLast("data.pkl".count))
        var machine = Unpickler(try archive.contents(of: pickleName))
        let root = try machine.run()
        func storage(_ key: String) throws -> Data { try archive.contents(of: prefix + "data/" + key) }
        func tensors(of value: Unpickler.Value) throws -> [String: Tensor]? {
            guard case .dict(let pairs) = value else { return nil }
            var out: [String: Tensor] = [:]
            for (key, item) in pairs {
                guard case .string(let name) = key, case .tensor(let t) = item else { continue }
                out[name] = try t.materialize(storage)
            }
            return out.isEmpty ? nil : out
        }
        var wrapped: [String: [String: Tensor]] = [:]
        if let direct = try tensors(of: root) { wrapped[""] = direct }
        if case .dict(let pairs) = root {
            for (key, value) in pairs {
                if case .string(let name) = key, let inner = try tensors(of: value) { wrapped[name] = inner }
            }
        }
        guard !wrapped.isEmpty else { throw ReadError("the checkpoint holds no tensors") }
        self.wrapped = wrapped
    }

    /// The tensors under the first of `keys` the checkpoint has ("" is the top
    /// level), stripping `prefix` from the names that have it.
    public func tensors(under keys: [String] = ["params_ema", "params", "state_dict", ""], prefix: String = "") -> [String: Tensor] {
        for key in keys {
            guard let found = wrapped[key] else { continue }
            if prefix.isEmpty { return found }
            var out: [String: Tensor] = [:]
            for (name, t) in found where name.hasPrefix(prefix) { out[String(name.dropFirst(prefix.count))] = t }
            return out
        }
        return [:]
    }
}

// MARK: - Zip (stored entries, as torch.save writes them)

struct ZipEntries {
    private let data: Data
    private var offsets: [String: (offset: Int, size: Int, method: UInt16)] = [:]
    var names: [String] { Array(offsets.keys) }

    init(_ data: Data) throws {
        self.data = data
        let bytes = [UInt8](data.suffix(min(data.count, 65_557)))
        let base = data.count - bytes.count
        // End of central directory: the last "PK\5\6".
        guard let eocd = stride(from: bytes.count - 22, through: 0, by: -1).first(where: {
            bytes[$0] == 0x50 && bytes[$0 + 1] == 0x4B && bytes[$0 + 2] == 5 && bytes[$0 + 3] == 6
        }) else { throw TorchCheckpoint.ReadError("not a zip archive") }
        func u16(_ at: Int) -> Int { Int(data[at]) | Int(data[at + 1]) << 8 }
        func u32(_ at: Int) -> Int { u16(at) | u16(at + 2) << 16 }
        func u64(_ at: Int) -> Int { u32(at) | u32(at + 4) << 32 }
        let e = base + eocd
        var count = u16(e + 10)
        var directory = u32(e + 16)
        // ZIP64 (torch writes it for large files): the locator just before.
        if directory == 0xFFFF_FFFF || count == 0xFFFF, e >= 20, data[e - 20] == 0x50, data[e - 19] == 0x4B,
           data[e - 18] == 6, data[e - 17] == 7 {
            let record = u64(e - 20 + 8)
            count = u64(record + 32)
            directory = u64(record + 48)
        }
        var at = directory
        for _ in 0 ..< count {
            guard at + 46 <= data.count, u32(at) == 0x0201_4B50 else { throw TorchCheckpoint.ReadError("a broken zip directory") }
            let method = UInt16(u16(at + 10))
            var size = u32(at + 20)
            let nameLength = u16(at + 28), extraLength = u16(at + 30), commentLength = u16(at + 32)
            var local = u32(at + 42)
            let name = String(decoding: data[(at + 46) ..< (at + 46 + nameLength)], as: UTF8.self)
            // ZIP64 extra field: the sizes and offset that didn't fit.
            var extra = at + 46 + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let id = u16(extra), length = u16(extra + 2)
                if id == 1 {
                    var field = extra + 4
                    if u32(at + 24) == 0xFFFF_FFFF { field += 8 } // uncompressed size
                    if size == 0xFFFF_FFFF { size = u64(field); field += 8 }
                    if local == 0xFFFF_FFFF { local = u64(field) }
                }
                extra += 4 + length
            }
            guard local + 30 <= data.count else { throw TorchCheckpoint.ReadError("a broken zip entry") }
            let start = local + 30 + u16(local + 26) + u16(local + 28)
            offsets[name] = (start, size, method)
            at = extraEnd + commentLength
        }
    }

    func contents(of name: String) throws -> Data {
        guard let entry = offsets[name] else { throw TorchCheckpoint.ReadError("no \(name) in the archive") }
        guard entry.method == 0 else { throw TorchCheckpoint.ReadError("\(name) is compressed; torch.save stores entries") }
        guard entry.offset + entry.size <= data.count else { throw TorchCheckpoint.ReadError("\(name) runs past the end") }
        return data.subdata(in: (data.startIndex + entry.offset) ..< (data.startIndex + entry.offset + entry.size))
    }
}

// MARK: - The pickle subset a state dict uses

struct Unpickler {
    struct TensorRef: Sendable {
        var storageKey: String
        var dtype: String
        var offset: Int
        var shape: [Int]
        var stride: [Int]

        func materialize(_ storage: (String) throws -> Data) throws -> TorchCheckpoint.Tensor {
            let raw = try storage(storageKey)
            let size: Int
            switch dtype {
            case "FloatStorage": size = 4
            case "HalfStorage", "BFloat16Storage": size = 2
            case "DoubleStorage": size = 8
            default: throw TorchCheckpoint.ReadError("tensors of \(dtype) aren't read")
            }
            let count = shape.reduce(1, *)
            var values = [Float](repeating: 0, count: count)
            // Walk the strides, so a non-contiguous view reads right too.
            var index = [Int](repeating: 0, count: shape.count)
            try raw.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
                for n in 0 ..< count {
                    var element = offset
                    for d in 0 ..< shape.count { element += index[d] * stride[d] }
                    let byte = element * size
                    guard byte + size <= buffer.count else { throw TorchCheckpoint.ReadError("a tensor runs past its storage") }
                    switch size {
                    case 4: values[n] = Float(bitPattern: buffer.loadUnaligned(fromByteOffset: byte, as: UInt32.self).littleEndian)
                    case 8: values[n] = Float(Double(bitPattern: buffer.loadUnaligned(fromByteOffset: byte, as: UInt64.self).littleEndian))
                    default:
                        let bits = buffer.loadUnaligned(fromByteOffset: byte, as: UInt16.self).littleEndian
                        values[n] = dtype == "HalfStorage" ? HalfFloat.value(from: bits) : Float(bitPattern: UInt32(bits) << 16)
                    }
                    var d = shape.count - 1
                    while d >= 0 {
                        index[d] += 1
                        if index[d] < shape[d] { break }
                        index[d] = 0
                        d -= 1
                    }
                }
            }
            return TorchCheckpoint.Tensor(shape: shape, values: values)
        }
    }

    indirect enum Value: Sendable {
        case none, bool(Bool), int(Int), float(Double), string(String), bytes(Data)
        case tuple([Value]), list([Value]), dict([(Value, Value)])
        case global(String, String)
        case storage(key: String, dtype: String)
        case tensor(TensorRef)
        case object
        case mark
    }

    private let bytes: [UInt8]
    private var at = 0
    private var stack: [Value] = []
    private var memo: [Int: Value] = [:]

    init(_ data: Data) { bytes = [UInt8](data) }

    private mutating func byte() throws -> UInt8 {
        guard at < bytes.count else { throw TorchCheckpoint.ReadError("the pickle ends early") }
        defer { at += 1 }
        return bytes[at]
    }

    private mutating func take(_ n: Int) throws -> ArraySlice<UInt8> {
        guard n >= 0, at + n <= bytes.count else { throw TorchCheckpoint.ReadError("the pickle ends early") }
        defer { at += n }
        return bytes[at ..< at + n]
    }

    private mutating func uint(_ n: Int) throws -> Int {
        try take(n).reversed().reduce(0) { $0 << 8 | Int($1) }
    }

    private mutating func line() throws -> String {
        guard let end = bytes[at...].firstIndex(of: 0x0A) else { throw TorchCheckpoint.ReadError("the pickle ends early") }
        defer { at = end + 1 }
        return String(decoding: bytes[at ..< end], as: UTF8.self)
    }

    private mutating func pop() throws -> Value {
        guard let v = stack.popLast() else { throw TorchCheckpoint.ReadError("the pickle's stack runs out") }
        return v
    }

    /// Everything since the last mark, removed with the mark.
    private mutating func popMark() throws -> [Value] {
        guard let i = stack.lastIndex(where: { if case .mark = $0 { true } else { false } }) else {
            throw TorchCheckpoint.ReadError("the pickle has no mark")
        }
        let items = Array(stack[(i + 1)...])
        stack.removeSubrange(i...)
        return items
    }

    private static func setItems(_ target: Value, _ items: [Value]) -> Value {
        guard case .dict(var pairs) = target else { return target }
        var i = 0
        while i + 1 < items.count {
            pairs.append((items[i], items[i + 1]))
            i += 2
        }
        return .dict(pairs)
    }

    private static func reduce(_ callable: Value, _ arguments: Value) -> Value {
        guard case .global(let module, let name) = callable, case .tuple(let args) = arguments else { return .object }
        switch (module, name) {
        case ("collections", "OrderedDict"), ("builtins", "dict"), ("__builtin__", "dict"):
            return .dict([])
        case ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"):
            guard args.count >= 4, case .storage(let key, let dtype) = args[0], case .int(let offset) = args[1],
                  case .tuple(let size) = args[2], case .tuple(let stride) = args[3] else { return .object }
            let shape = size.compactMap { if case .int(let v) = $0 { v } else { nil } }
            let strides = stride.compactMap { if case .int(let v) = $0 { v } else { nil } }
            guard shape.count == size.count, strides.count == shape.count else { return .object }
            return .tensor(TensorRef(storageKey: key, dtype: dtype, offset: offset, shape: shape, stride: strides))
        default:
            return .object
        }
    }

    mutating func run() throws -> Value {
        while true {
            let op = try byte()
            switch op {
            case 0x80: _ = try byte() // PROTO
            case 0x95: _ = try take(8) // FRAME
            case 0x2E: return try pop() // STOP
            case 0x28: stack.append(.mark) // MARK
            case 0x7D: stack.append(.dict([])) // EMPTY_DICT
            case 0x29: stack.append(.tuple([])) // EMPTY_TUPLE
            case 0x5D: stack.append(.list([])) // EMPTY_LIST
            case 0x4E: stack.append(.none) // NONE
            case 0x88: stack.append(.bool(true)) // NEWTRUE
            case 0x89: stack.append(.bool(false)) // NEWFALSE
            case 0x4B: stack.append(.int(Int(try byte()))) // BININT1
            case 0x4D: stack.append(.int(try uint(2))) // BININT2
            case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4))))) // BININT
            case 0x8A: // LONG1
                let n = Int(try byte())
                let raw = try take(n)
                var v = 0
                for (i, b) in raw.enumerated() where i < 8 { v |= Int(b) << (8 * i) }
                if n > 0, n < 8, raw.last! & 0x80 != 0 { v -= 1 << (8 * n) }
                stack.append(.int(v))
            case 0x47: // BINFLOAT, big-endian
                stack.append(.float(Double(bitPattern: UInt64(try take(8).reduce(0) { $0 << 8 | Int($1) }))))
            case 0x58: stack.append(.string(String(decoding: try take(try uint(4)), as: UTF8.self))) // BINUNICODE
            case 0x8C: stack.append(.string(String(decoding: try take(Int(try byte())), as: UTF8.self))) // SHORT_BINUNICODE
            case 0x8D: stack.append(.string(String(decoding: try take(try uint(8)), as: UTF8.self))) // BINUNICODE8
            case 0x43: stack.append(.bytes(Data(try take(Int(try byte()))))) // SHORT_BINBYTES
            case 0x42: stack.append(.bytes(Data(try take(try uint(4))))) // BINBYTES
            case 0x71: memo[Int(try byte())] = stack.last // BINPUT
            case 0x72: memo[try uint(4)] = stack.last // LONG_BINPUT
            case 0x94: memo[memo.count] = stack.last // MEMOIZE
            case 0x68: stack.append(memo[Int(try byte())] ?? .none) // BINGET
            case 0x6A: stack.append(memo[try uint(4)] ?? .none) // LONG_BINGET
            case 0x63: // GLOBAL
                let module = try line()
                stack.append(.global(module, try line()))
            case 0x93: // STACK_GLOBAL
                let name = try pop(), module = try pop()
                guard case .string(let m) = module, case .string(let n) = name else { throw TorchCheckpoint.ReadError("a bad global") }
                stack.append(.global(m, n))
            case 0x74: stack.append(.tuple(try popMark())) // TUPLE
            case 0x85: stack.append(.tuple([try pop()])) // TUPLE1
            case 0x86: let b = try pop(), a = try pop(); stack.append(.tuple([a, b])) // TUPLE2
            case 0x87: let c = try pop(), b = try pop(), a = try pop(); stack.append(.tuple([a, b, c])) // TUPLE3
            case 0x6C: stack.append(.list(try popMark())) // LIST
            case 0x61: // APPEND
                let item = try pop()
                if case .list(var items) = try pop() { items.append(item); stack.append(.list(items)) } else { stack.append(.object) }
            case 0x65: // APPENDS
                let items = try popMark()
                if case .list(var list) = try pop() { list += items; stack.append(.list(list)) } else { stack.append(.object) }
            case 0x73: // SETITEM
                let value = try pop(), key = try pop()
                stack.append(Self.setItems(try pop(), [key, value]))
            case 0x75: // SETITEMS
                let items = try popMark()
                stack.append(Self.setItems(try pop(), items))
            case 0x52: // REDUCE
                let args = try pop()
                stack.append(Self.reduce(try pop(), args))
            case 0x81: // NEWOBJ
                _ = try pop(); _ = try pop(); stack.append(.object)
            case 0x62: _ = try pop() // BUILD: state for the object below (a state dict's _metadata); dropped
            case 0x51: // BINPERSID: ("storage", StorageType, key, location, count)
                guard case .tuple(let parts) = try pop(), parts.count >= 3, case .global(_, let dtype) = parts[1],
                      case .string(let key) = parts[2] else { throw TorchCheckpoint.ReadError("a storage reference it can't read") }
                stack.append(.storage(key: key, dtype: dtype))
            case 0x30: _ = try pop() // POP
            case 0x31: _ = try popMark() // POP_MARK
            case 0x32: stack.append(stack.last ?? .none) // DUP
            default:
                throw TorchCheckpoint.ReadError(String(format: "pickle opcode 0x%02X isn't read", op))
            }
        }
    }
}
