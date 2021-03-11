# assume for now that we build fox x86 linux
set(Halide_11_0_1_URL "https://github.com/halide/Halide/releases/download/v11.0.1/Halide-11.0.1-x86-64-linux-85c1b91c47ce15aab0d9502d955e48615f3bcee0.tar.gz")

ExternalSource_Download(Halide DIRNAME Halide-11.0.1 "${Halide_11_0_1_URL}")

find_library(Halide_LIB NAMES libHalide.so PATHS "${Halide_SOURCE_DIR}/lib")
set(Halide_INCLUDE_DIR "${Halide_SOURCE_DIR}/include")

add_library(Halide::Halide SHARED IMPORTED)
set_property(TARGET Halide::Halide PROPERTY IMPORTED_LOCATION "${Halide_LIB}")
set_property(TARGET Halide::Halide PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Halide_INCLUDE_DIR}")
