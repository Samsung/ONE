# vconone

_vconone_ provides version number and strings for one-* commands and command
line tools

# Revise version number

To revise version number, update `VCONONE_VERSION` in `CmakeLists.txt`
or give `-DVCONONE_VERSION=0x0000000100080001` at cmake configure step.

Number given is four numbers `build`, `patch`, `minor` and `major` in order for
each 16bit integers. `build` is not used for now.

`0x0000000100080001` version is interpretered as `1.8.1`
