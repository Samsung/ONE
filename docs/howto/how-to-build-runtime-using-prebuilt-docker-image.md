# How to Build Using Prebuilt Docker Image

You can pull a prebuilt image from the `DockerHub` and use it to build your project.

We are supporting docker images for the following environments:
- Ubuntu 22.04 LTS (x86_64, arm64): `nnfw/one-devtools:jammy`
- Ubuntu 24.04 LTS (x86_64, arm64): `nnfw/one-devtools:noble`
  - Built arm64 runtime on Ubuntu 24.04 is not tested yet

We are supporting docker images for cross building as well:
- Ubuntu 22.04 LTS (x86_64 host, arm32/arm64 target): `nnfw/one-devtools:jammy`
- Ubuntu 24.04 LTS (x86_64 host, arm32/arm64 target): `nnfw/one-devtools:noble`
  - Built arm64 runtime on Ubuntu 24.04 is not tested yet

We are supporting docker images for android building as well:
- Ubuntu host, android target: `nnfw/one-devtools:android-sdk`
