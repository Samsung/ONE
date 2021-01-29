nnas_find_package(TensorFlowSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowEigenSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowRuySource EXACT 2.3.0 QUIET)

if (NOT TensorFlowSource_FOUND)
  message(STATUS "Skipping luci-interpreter: TensorFlow not found")
  return()
endif ()

if (NOT TensorFlowGEMMLowpSource_FOUND)
  message(STATUS "Skipping luci-interpreter: gemmlowp not found")
  return()
endif ()

if (NOT TensorFlowEigenSource_FOUND)
  message(STATUS "Skipping luci-interpreter: Eigen not found")
  return()
endif ()

if (NOT TensorFlowRuySource_FOUND)
  message(STATUS "Skipping luci-interpreter: Ruy not found")
  return()
endif ()

add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(loader)

set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/Interpreter.h"
    Interpreter.cpp)

add_library(luci_interpreter SHARED ${SOURCES})
target_include_directories(luci_interpreter PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(luci_interpreter PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(luci_interpreter
    PUBLIC luci_lang luci_interpreter_loader luci_interpreter_core
    PRIVATE nncc_common)

install(TARGETS luci_interpreter DESTINATION lib)

add_executable(luci_runner Runner.cpp)
target_link_libraries(luci_runner PRIVATE luci_interpreter)
target_link_libraries(luci_runner PRIVATE oops)
target_link_libraries(luci_runner PRIVATE loco)
target_link_libraries(luci_runner PRIVATE luci_import)
target_link_libraries(luci_runner PRIVATE luci_export)
target_link_libraries(luci_runner PRIVATE luci_lang)
