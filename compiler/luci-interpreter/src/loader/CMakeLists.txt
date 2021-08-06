if (NOT LUCI_MICRO)
  set(KERNEL_REGISTER_FILE "${LUCI_INTERPRETER_SOURCE_DIR}/KernelsToBuild.lst")
else ()
  set(KERNEL_REGISTER_FILE "${LUCI_INTERPRETER_SOURCE_DIR}/micro/KernelsToBuild.lst")
endif ()

set(SOURCES
    GraphLoader.h
    GraphLoader.cpp
    KernelBuilderHelper.h
    KernelBuilderHelper.cpp
    KernelBuilder.h
    KernelBuilder.cpp
    ModuleLoader.h
    ModuleLoader.cpp
    RuntimeToIR.h
    ${KERNEL_REGISTER_FILE})

add_library(${LUCI_INTERPRETER_LOADER} STATIC ${SOURCES})
set_target_properties(${LUCI_INTERPRETER_LOADER} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${LUCI_INTERPRETER_LOADER} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_LOADER}
    PUBLIC luci_lang ${LUCI_INTERPRETER_CORE}
    PRIVATE ${LUCI_INTERPRETER_KERNELS} nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCES KernelBuilder.test.cpp)

GTest_AddTest(${LUCI_INTERPRETER_LOADER}_test ${TEST_SOURCES})
target_link_libraries(${LUCI_INTERPRETER_LOADER}_test ${LUCI_INTERPRETER_LOADER})
