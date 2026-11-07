// RAVECompanion's Swift code on one side, the Rust port on the other.
//
//   CompanionInterop device <pair-port> <tls-port>   pair with a Rust host, then
//                                                    open its TLS-PSK channel and identify
//   CompanionInterop host <pair-port> <tls-port>     the Swift host a Rust device pairs with
//
// Prints `CODE <six digits>` when the code is known, `OK <detail>` on success,
// and exits non-zero on failure. Uses a namespace of its own so nothing real
// answers it.

import Foundation
import Network
import RAVECompanion

enum Grant: String, Codable, Hashable, Sendable { case play, work, admin }

let args = CommandLine.arguments
let pairPort = args.count == 4 ? UInt16(args[2]) ?? 0 : 0
let tlsPort = args.count == 4 ? UInt16(args[3]) ?? 0 : 0
if pairPort == 0 || tlsPort == 0 {
    FileHandle.standardError.write(Data("usage: CompanionInterop device|host <pair-port> <tls-port>\n".utf8))
    exit(2)
}

let namespace = RAVECompanionNamespace(serviceStem: "vergence-interop", labelPrefix: "VergenceInterop",
                                       urlScheme: "vergence-interop", defaultPairPort: pairPort,
                                       hostAppName: "Rust interop host")
let channel = RAVECompanionChannel(identityBase: "VergenceInterop/v1", salt: "Vergence-Interop-PSK-v1")
let exporterLabel = "EXPORTER-VergenceInterop-Work-Device-v1"

func say(_ line: String) {
    print(line)
    fflush(stdout)
}

func fail(_ reason: String) -> Never {
    say("FAIL \(reason)")
    exit(1)
}

func frame(_ body: Data) -> Data {
    var length = UInt32(body.count).littleEndian
    var out = Data(bytes: &length, count: 4)
    out.append(body)
    return out
}

/// Reads one length-prefixed frame.
func readFrame(_ connection: NWConnection, _ done: @escaping @Sendable (Data?) -> Void) {
    connection.receive(minimumIncompleteLength: 4, maximumLength: 4) { header, _, _, _ in
        guard let header, header.count == 4 else { return done(nil) }
        let length = header.enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        connection.receive(minimumIncompleteLength: length, maximumLength: length) { body, _, _, _ in
            done(body)
        }
    }
}

DispatchQueue.main.asyncAfter(deadline: .now() + 60) { fail("timed out") }

@MainActor
func runDevice() {
    let locator = RAVECompanionLocator(namespace: namespace, deviceID: "swift-device", deviceName: { "Swift harness" })
    let link = RAVECompanionPairLink(namespace, macID: "", name: "Rust", addresses: ["127.0.0.1"])
    let client = RAVEPairingClient<Grant>(locator: locator, link: link, deviceName: "Swift harness",
                                          requested: [.play, .work])
    client.start()
    var printedCode = false
    Timer.scheduledTimer(withTimeInterval: 0.02, repeats: true) { timer in
        MainActor.assumeIsolated {
            switch client.state {
            case .confirming(let code) where !printedCode:
                printedCode = true
                say("CODE \(code)")
            case .paired(let grant):
                timer.invalidate()
                say("GRANT host=\(grant.macID) grants=\((grant.grants ?? []).joined(separator: ","))")
                connectChannel(token: grant.token)
            case .failed(let reason):
                fail(reason)
            default:
                break
            }
        }
    }
}

@MainActor
func connectChannel(token: String) {
    let credential = RAVECompanionCredential(token: token, deviceID: "swift-device")
    let connection = NWConnection(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: tlsPort)!,
                                  using: channel.tcpParameters(credentials: [credential]))
    connection.stateUpdateHandler = { state in
        switch state {
        case .ready:
            guard let identify = RAVECompanionIdentify.make(namespace, credential: credential,
                                                            exporterLabel: exporterLabel, connection: connection) else {
                fail("no exporter value")
            }
            connection.send(content: frame(identify.encoded()), completion: .contentProcessed { _ in })
            readFrame(connection) { reply in
                guard let reply else { fail("no reply on the channel") }
                say("OK channel reply: \(String(decoding: reply, as: UTF8.self))")
                exit(0)
            }
        case .failed(let error), .waiting(let error):
            fail("channel: \(error)")
        default:
            break
        }
    }
    connection.start(queue: .main)
}

@MainActor
final class HostSide {
    let listener = RAVEDirectPairingListener(namespace: namespace)
    var server: RAVEPairingServer<Grant>?
    var enrolled: RAVECompanionCredential?
    var keyring = RAVECompanionKeyring<Grant>(namespace)
    var tls: NWListener?

    func start() {
        listener.onConnection = { [self] connection in
            let server = RAVEPairingServer<Grant>(context: .init(
                namespace: namespace, deviceID: nil, deviceName: "", hostID: "SWIFT-HOST", hostName: "Swift Host",
                defaultGrants: [.play], addresses: { ["127.0.0.1"] },
                enroll: { [self] id, _, grants, _ in
                    let token = try! RAVECompanionCredential.generateToken()
                    let credential = RAVECompanionCredential(token: token, deviceID: id)
                    enrolled = credential
                    keyring.replace([.init(credential: credential, grants: grants)])
                    return token
                }))
            server.approve = { request in
                say("CODE \(request.code)")
                return .pair(request.requested)
            }
            server.onFinish = { [self] outcome, name in
                Task { @MainActor in
                    guard case .paired = outcome else { fail("pairing ended \(outcome) for \(name)") }
                    say("PAIRED \(name)")
                    listener.setListening(false)
                    openChannel()
                }
            }
            self.server = server
            server.start(accepting: connection)
        }
        listener.setListening(true)
        say("LISTENING")
    }

    func openChannel() {
        guard let enrolled else { fail("nothing enrolled") }
        let parameters = channel.tcpParameters(credentials: [enrolled])
        parameters.allowLocalEndpointReuse = true
        let tls = try! NWListener(using: parameters, on: NWEndpoint.Port(rawValue: tlsPort)!)
        tls.newConnectionHandler = { [keyring] connection in
            connection.stateUpdateHandler = { state in
                guard case .ready = state else {
                    if case .failed(let error) = state { fail("channel: \(error)") }
                    return
                }
                readFrame(connection) { body in
                    guard let body, let identify = RAVECompanionIdentify.decode(body),
                          let binding = RAVECompanionIdentify.binding(connection, exporterLabel: exporterLabel) else {
                        fail("no identify")
                    }
                    let entry = keyring.verify(identify, binding: binding)
                    let reply = entry.map { "ok \($0.credential.deviceID ?? "shared")" } ?? "refused"
                    connection.send(content: frame(Data(reply.utf8)), completion: .contentProcessed { _ in
                        say(entry == nil ? "FAIL identify refused" : "OK verified \(identify.deviceID ?? "shared")")
                        exit(entry == nil ? 1 : 0)
                    })
                }
            }
            connection.start(queue: .main)
        }
        tls.stateUpdateHandler = { state in
            if case .ready = state { say("CHANNEL") }
            if case .failed(let error) = state { fail("channel listener: \(error)") }
        }
        tls.start(queue: .main)
        self.tls = tls
    }
}

var host: HostSide?
MainActor.assumeIsolated {
    switch args[1] {
    case "device": runDevice()
    case "host":
        host = HostSide()
        host?.start()
    default: fail("unknown mode \(args[1])")
    }
}
// A run loop, not dispatchMain(): the state poll is a Timer.
RunLoop.main.run()
