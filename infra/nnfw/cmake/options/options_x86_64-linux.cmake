#
# x86_64 linux cmake options
#
option(BUILD_ARMCOMPUTE "Build ARM Compute from the downloaded source" OFF)
option(BUILD_XNNPACK "Build XNNPACK" OFF)
option(DOWNLOAD_ARMCOMPUTE "Download ARM Compute source" OFF)

option(BUILD_NPUD "Build NPU daemon" ON)
option(ENVVAR_NPUD_CONFIG "Use environment variable for npud configuration" ON)
