# Check cross compiling
# 1. pkg_check_modules is not working as expected on cross-compiling environment.
# 2. TRIXEngine will not be used on cross-compiling environment
if(CMAKE_CROSSCOMPILING)
  set(TRIXEngine_FOUND FALSE)
  return()
endif(CMAKE_CROSSCOMPILING)

# Check if target already exists. If so, do nothing.
if(TARGET trix-engine)
  return()
endif(TARGET trix-engine)

find_package(PkgConfig REQUIRED)
# TRIXEngine version is required to higher than 2.5.0
pkg_check_modules(TRIXEngine QUIET IMPORTED_TARGET npu-engine>2.5.0)
if(NOT TRIXEngine_FOUND)
  return()
endif(NOT TRIXEngine_FOUND)

add_library(trix-engine INTERFACE)
target_link_libraries(trix-engine INTERFACE PkgConfig::TRIXEngine)
