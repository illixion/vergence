# Third-party notices

## edge264

`Sources/CEdge264` is [edge264](https://github.com/tvlabs/edge264) by Thibault Raffaillac and
Celticom / TVLabs, the H.264 decoder (MVC included) behind Vergence's disc decoding. It is
distributed under the BSD 3-Clause license in `Sources/CEdge264/LICENSE`, and carries the local
patches listed in `Vendor/README.md`.

## OpenSSL

The Windows build of the desktop app (`desktop/`) builds [OpenSSL](https://www.openssl.org/) into
its executable (the `openssl-src` crate) for the paired channel's TLS-PSK. OpenSSL 3 is distributed
under the Apache License 2.0. The macOS and Linux builds link the system's OpenSSL instead.


## Real-ESRGAN

`convert --upscale` builds the compact network of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
by Xintao Wang et al. (SRVGGNetCompact, BSD 3-Clause license) from the authors' published weights
(`realesr-general-x4v3.pth`). Neither its code nor its weights are part of Vergence: the layers are
written out from the weight file's own names and shapes (`Sources/VergenceApple/Upscale/UpscaleNetwork.swift`,
`convert/src/upscale.rs`), and the weights are downloaded by whoever runs `--download-model`.
