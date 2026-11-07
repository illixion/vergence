#!/usr/bin/env python3
"""Compares analysis tracks record by record: a reference (one whole track,
e.g. from `vergence analyze`) against one or more others (e.g. a coordinator's
chunks, which together cover the same title). Exits 1 when a record is
missing on either side or a value differs by more than the tolerance.

  compare-analysis-tracks.py REFERENCE.vgat OTHER.vgat... [--tolerance PX]

Payloads are inflated (raw DEFLATE, row deltas) and decoded the way
VergenceCore's AnalysisTrack.swift does. Standard library only.
"""
import json
import struct
import sys
import zlib

KINDS = {1: "flowPair", 2: "disparity", 3: "depth", 4: "gainMap"}


def records(path):
    data = open(path, "rb").read()
    if data[:4] != b"VGAT":
        sys.exit(f"{path}: not an analysis track")
    length = struct.unpack_from("<I", data, 5)[0]
    header = json.loads(data[9:9 + length])
    position = 9 + length
    out = {}
    while position < len(data):
        size = struct.unpack_from("<I", data, position)[0]
        body = data[position + 4:position + 4 + size]
        kind, eye, encoding, channels = body[0], body[1], body[2], body[3]
        width, height = struct.unpack_from("<HH", body, 4)
        time, end = struct.unpack_from("<qq", body, 8)
        timescale = struct.unpack_from("<i", body, 24)[0]
        scale, offset = struct.unpack_from("<ff", body, 28)
        payload = body[36:]
        base = encoding & 0x7F
        width_bytes = 1 if base == 3 else 2
        if encoding & 0x80:
            payload = zlib.decompress(payload, -15)
        raw = list(struct.unpack(f"<{len(payload) // width_bytes}{'B' if width_bytes == 1 else 'H'}", payload))
        if encoding & 0x80:
            modulus = 1 << (8 * width_bytes)
            for row in range(channels * height):
                previous = 0
                for x in range(width):
                    i = row * width + x
                    previous = (raw[i] + previous) % modulus
                    raw[i] = previous
        if base == 1:
            raw = list(struct.unpack(f"<{len(raw)}e", struct.pack(f"<{len(raw)}H", *raw)))
        elif base == 2:
            raw = [v - 65536 if v >= 32768 else v for v in raw]
        out[(kind, eye, time / timescale, end / timescale)] = [v * scale + offset for v in raw]
        position += 4 + size
    return header, out


def main():
    args = sys.argv[1:]
    tolerance = 0.5
    if "--tolerance" in args:
        i = args.index("--tolerance")
        tolerance = float(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    _, reference = records(args[0])
    other = {}
    for path in args[1:]:
        other.update(records(path)[1])
    missing, extra = set(reference) - set(other), set(other) - set(reference)
    worst = {}
    for key in set(reference) & set(other):
        difference = max((abs(a - b) for a, b in zip(reference[key], other[key])), default=0)
        worst[KINDS.get(key[0], key[0])] = max(worst.get(KINDS.get(key[0], key[0]), 0), difference)
    print(f"{len(reference)} records in the reference, {len(other)} in the others, "
          f"{len(missing)} missing, {len(extra)} extra")
    print("largest difference by kind (in the record's own units):",
          ", ".join(f"{k} {v:.3f}" for k, v in sorted(worst.items())))
    if missing or extra or any(v > tolerance for v in worst.values()):
        print("DIFFERENT")
        sys.exit(1)
    print("same within", tolerance)


main()
