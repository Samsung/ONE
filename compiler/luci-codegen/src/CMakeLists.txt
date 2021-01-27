set(SOURCES LuciCodegen.cpp
            CodegenKernelBuilder.cpp
            CodegenKernelBuilder.h
            SubgraphContext.cpp
            SubgraphContext.h
            Utilities.cpp
            Utilities.h
            )
set(HEADERS ${LUCI_CODEGEN_INCLUDE_DIR}/LuciCodegen.h)

add_library(luci_codegen SHARED ${SOURCES} ${HEADERS})
target_include_directories(luci_codegen PUBLIC "${LUCI_CODEGEN_INCLUDE_DIR}")
target_include_directories(luci_codegen PRIVATE "${LUCI_CODEGEN_SOURCE_DIR}")

target_link_libraries(luci_codegen
        PUBLIC luci_lang
        PRIVATE nncc_common Halide::Halide flatbuffers)

install(TARGETS luci_codegen DESTINATION lib)

set(DRIVER_SOURCE Driver.cpp)

if(ENABLE_TEST)
    nnas_find_package(GTest REQUIRED)
    GTest_AddTest(luci_codegen_kernel_builder_test CodegenKernelBuilder.test.cpp)
    target_link_libraries(luci_codegen_kernel_builder_test luci_codegen luci_import luci_export Halide::Halide)
endif(ENABLE_TEST)


add_executable(circle_codegen ${DRIVER_SOURCE})
target_link_libraries(circle_codegen PRIVATE luci_codegen luci_import luci_export)
