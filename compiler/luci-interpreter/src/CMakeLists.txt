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

set(LUCI_INTERPRETER_BINARY "luci_interpreter${LUCI_INTERPRETER_SUFFIX}")

if (NOT LUCI_INTERPRETER_STATIC)
  add_library(${LUCI_INTERPRETER_BINARY} SHARED ${SOURCES})
else ()
  add_library(${LUCI_INTERPRETER_BINARY} STATIC ${SOURCES})
endif ()

target_include_directories(${LUCI_INTERPRETER_BINARY} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_BINARY} PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_BINARY}
    PUBLIC luci_lang luci_interpreter_loader luci_interpreter_core
    PRIVATE nncc_common)

install(TARGETS luci_interpreter DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
