/*
 Vergence - the sphere a VR picture is drawn on

 A VR video or photo is drawn from the inside of a sphere around the
 viewer: each vertex is a direction, and its texture coordinate is where
 that direction lands in one eye's picture (`VRProjection.eyeCoordinate`).
 Doing the projection here, per vertex, means the renderer needs nothing
 but a texture lookup per eye, and a fisheye needs no remapping pass: the
 mapping is smooth, so a dense enough mesh interpolates it closely (the
 error is checked in the tests).

 Only the part of the sphere the projection covers is built: a cap for a
 fisheye, a band of longitude for an equirectangular picture. Triangles
 face inward (counter-clockwise seen from the centre), the radius is 1,
 and the viewer looks down -Z with +Y up.

 Texture coordinates are the eye's own (0...1, origin top left, v down);
 the renderer moves them into the packed frame per eye with
 `VRFormat.eyeRegion` and flips v if its textures are bottom-up.
 */

import Foundation

public struct VRSphereMesh: Sendable {
    public var positions: [SIMD3<Float>]
    public var eyeCoordinates: [SIMD2<Float>]
    public var indices: [UInt32]

    /// The mesh for `projection`, nil for a flat picture. `detail` is the
    /// number of segments across 180°.
    public static func make(for projection: VRProjection, detail: Int = 96) -> VRSphereMesh? {
        let detail = max(8, detail)
        switch projection {
        case .flat:
            return nil
        case .equirectangular:
            return band(projection, detail: detail)
        case .fisheye:
            return cap(projection, detail: detail)
        }
    }

    /// Longitude ±half by latitude ±90°, as a grid.
    private static func band(_ projection: VRProjection, detail: Int) -> VRSphereMesh {
        let half = projection.halfWidthDegrees * .pi / 180
        let columns = max(8, Int((Double(detail) * (2 * half) / .pi).rounded()))
        let rows = detail
        var mesh = VRSphereMesh(positions: [], eyeCoordinates: [], indices: [])
        mesh.positions.reserveCapacity((columns + 1) * (rows + 1))
        for row in 0 ... rows {
            let v = Double(row) / Double(rows)
            let latitude = (0.5 - v) * .pi
            for column in 0 ... columns {
                let u = Double(column) / Double(columns)
                let longitude = (u * 2 - 1) * half
                let direction = SIMD3(cos(latitude) * sin(longitude), sin(latitude), -cos(latitude) * cos(longitude))
                mesh.positions.append(SIMD3<Float>(direction))
                // Exactly the grid's own u, v (also at the 360° seam, where
                // atan2 would wrap the last column back to 0).
                mesh.eyeCoordinates.append(SIMD2(Float(u), Float(v)))
            }
        }
        let stride = UInt32(columns + 1)
        for row in 0 ..< UInt32(rows) {
            for column in 0 ..< UInt32(columns) {
                let a = row * stride + column, b = a + 1, c = a + stride, d = c + 1
                // Seen from inside: a (top left) → c (bottom left) → b.
                mesh.indices += [a, c, b, b, c, d]
            }
        }
        return mesh
    }

    /// A cap of angular radius half around -Z, as rings around the centre.
    private static func cap(_ projection: VRProjection, detail: Int) -> VRSphereMesh {
        let half = projection.halfWidthDegrees * .pi / 180
        let rings = max(4, Int((Double(detail) * half / .pi).rounded()))
        let sectors = detail * 2
        var mesh = VRSphereMesh(positions: [], eyeCoordinates: [], indices: [])
        func add(theta: Double, psi: Double) {
            let direction = SIMD3(sin(theta) * cos(psi), sin(theta) * sin(psi), -cos(theta))
            mesh.positions.append(SIMD3<Float>(direction))
            let uv = projection.eyeCoordinate(of: direction) ?? SIMD2(0.5, 0.5)
            mesh.eyeCoordinates.append(SIMD2<Float>(uv))
        }
        add(theta: 0, psi: 0) // the centre
        for ring in 1 ... rings {
            let theta = half * Double(ring) / Double(rings)
            for sector in 0 ... sectors {
                add(theta: theta, psi: 2 * .pi * Double(sector) / Double(sectors))
            }
        }
        let stride = UInt32(sectors + 1)
        func vertex(_ ring: Int, _ sector: Int) -> UInt32 { 1 + UInt32(ring - 1) * stride + UInt32(sector) }
        // psi grows counter-clockwise as the viewer sees it (+X right, +Y
        // up), so a ring step outward then along psi is counter-clockwise.
        for sector in 0 ..< sectors {
            mesh.indices += [0, vertex(1, sector), vertex(1, sector + 1)]
        }
        if rings > 1 {
            for ring in 1 ..< rings {
                for sector in 0 ..< sectors {
                    let a = vertex(ring, sector), b = vertex(ring, sector + 1)
                    let c = vertex(ring + 1, sector), d = vertex(ring + 1, sector + 1)
                    mesh.indices += [a, c, b, b, c, d]
                }
            }
        }
        return mesh
    }
}
