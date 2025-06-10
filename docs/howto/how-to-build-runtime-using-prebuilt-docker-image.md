# How to Build Using Prebuilt Docker Image

You can pull a prebuilt image from the `DockerHub` and use it to build your project.

We are supporting docker images for the following environments:
- Ubuntu 20.04 LTS (x86_64, arm64): `nnfw/one-devtools:focal`
- Ubuntu 22.04 LTS (x86_64, arm64): `nnfw/one-devtools:jammy`
- Ubuntu 24.04 LTS (x86_64, arm64): `nnfw/one-devtools:noble` (not tested yet)

We are supporting docker images for cross building as well:
- Ubuntu 20.04 LTS (x86_64 host, arm32/arm64 target): `nnfw/one-devtools:focal-cross`
- Ubuntu 22.04 LTS (x86_64 host, arm32/arm64 target): `nnfw/one-devtools:jammy-cross`
- Ubuntu 24.04 LTS (x86_64 host, arm32/arm64 target): `nnfw/one-devtools:noble-cross` (not tested yet)

We are supporting docker images for android building as well:
- Ubuntu host, android target: `nnfw/one-devtools:android-sdk`
