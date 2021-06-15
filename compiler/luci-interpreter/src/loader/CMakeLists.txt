set(SOURCES
    GraphLoader.h
    GraphLoader.cpp
    KernelBuilderHelper.h
    KernelBuilderHelper.cpp
    KernelBuilder.h
    KernelBuilder.cpp
    ModuleLoader.h
    ModuleLoader.cpp
    RuntimeToIR.h)

add_library(luci_interpreter_loader STATIC ${SOURCES})
set_target_properties(luci_interpreter_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luci_interpreter_loader PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(luci_interpreter_loader
    PUBLIC luci_lang luci_interpreter_core
    PRIVATE luci_interpreter_kernels nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCES KernelBuilder.test.cpp)

GTest_AddTest(luci_interpreter_loader_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_loader_test luci_interpreter_loader)
