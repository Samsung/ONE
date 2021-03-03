# luci_codegen LIBRARY

set(SOURCES Codegen.cpp
        KernelBuilder.cpp
        KernelBuilder.h
        Scheduler.cpp
        Scheduler.h
        SubgraphContext.cpp
        SubgraphContext.h
        Utilities.cpp
        Utilities.h
        )

set(WRAPPER_SOURCES ${LUCI_CODEGEN_SUPPORT_DIR}/WrapperLib.h)
set(WRAPPER_LIB ${CMAKE_CURRENT_BINARY_DIR}/GeneratedWrapperLib.h)

set(HEADERS ${LUCI_CODEGEN_INCLUDE_DIR}/Codegen.h ${LUCI_CODEGEN_INCLUDE_DIR}/CodegenOptions.h)

add_library(luci_codegen SHARED ${SOURCES} ${HEADERS} ${WRAPPER_LIB})

target_include_directories(luci_codegen PUBLIC "${LUCI_CODEGEN_INCLUDE_DIR}")
target_include_directories(luci_codegen PRIVATE "${LUCI_CODEGEN_SOURCE_DIR}")
target_include_directories(luci_codegen PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(luci_codegen PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(luci_codegen PUBLIC luci_lang)
target_link_libraries(luci_codegen PRIVATE nncc_common)
target_link_libraries(luci_codegen PRIVATE Halide::Halide)
target_link_libraries(luci_codegen PRIVATE flatbuffers)
target_link_libraries(luci_codegen PRIVATE ${Boost_LIBRARIES})

add_custom_command(OUTPUT ${WRAPPER_LIB}
        COMMAND python3 ${LUCI_CODEGEN_SUPPORT_DIR}/embed.py ${LUCI_CODEGEN_SUPPORT_DIR}/WrapperLib.h ${CMAKE_CURRENT_BINARY_DIR}/GeneratedWrapperLib.h wrapper_lib_source
        DEPENDS ${WRAPPER_SOURCES}
        VERBATIM)

install(TARGETS luci_codegen DESTINATION lib)

# native_compiler EXECUTABLE

set(DRIVER_SOURCE Driver.cpp)

add_executable(native_compiler ${DRIVER_SOURCE})

target_include_directories(native_compiler PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(native_compiler PRIVATE luci_codegen)
target_link_libraries(native_compiler PRIVATE luci_import)
target_link_libraries(native_compiler PRIVATE luci_export)
target_link_libraries(native_compiler PRIVATE ${Boost_LIBRARIES})

# tests

if(ENABLE_TEST)
    nnas_find_package(GTest REQUIRED)
    set(TEST_SOURCES KernelBuilder.test.cpp Codegen.test.cpp)
    GTest_AddTest(luci_codegen_test ${TEST_SOURCES})
    target_link_libraries(luci_codegen_test luci_codegen luci_import luci_export Halide::Halide)
endif(ENABLE_TEST)

add_executable(simple_runner simple_runner.cpp)
target_link_libraries(simple_runner PUBLIC Halide::Halide)
