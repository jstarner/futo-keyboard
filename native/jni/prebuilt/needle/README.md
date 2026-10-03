# Cactus Needle engine (prebuilt)

Prebuilt static library for the Cactus engine that runs Whistle `.cact` speech models
(https://github.com/cactus-compute/needle). The engine source is not public; only these
binaries are distributed, under the Apache-2.0 license in `LICENSE`.

Source: https://huggingface.co/Cactus-Compute/needle3 at commit `c7c415a3d1b3d929014bc6e866d51ebb971f7089`

| File | Upstream path | sha256 |
|---|---|---|
| `arm64-v8a/libneedle.a` | `android-arm64/libneedle.a` | `16752fb75adea7af89bbc435a97d1cda7e71bc74d04578d551d0d3bbd5f9e633` |
| `armeabi-v7a/libneedle.a` | `android-armv7/libneedle.a` | `8cb5af3c2a7e6ad55a1fe94cb9b6c012b40ae921aae2981594362389872fa7c0` |
| `include/needle.h` | `android-arm64/needle.h` | `90f347f9dca1199de79967ab199a56e0588bd473fe306051f8d80d146976a324` |
| `LICENSE` | `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

There are no x86/x86_64 builds; on those ABIs `CMakeLists.txt` leaves the engine out and
`org_futo_voiceinput_WhistleModel.cpp` compiles stubs.

The static library imports no networking or `getenv` symbols; the telemetry mentioned in the
upstream README lives only in the `needle` CLI binary, which is not included here.

Compatible model: https://huggingface.co/Cactus-Compute/whistle (`whistle.cact`).
