include("${LUCI_INTERPRETER_PAL_DIR}/pal.cmake")

initialize_pal()

if (NOT PAL_INITIALIZED)
  message("PAL Failed to initialize, skip luci-interpreter")
  return()
endif()

message(STATUS "LUCI INTERPRETER BEGIN")

set(LUCI_INTERPRETER_BINARY "luci_interpreter_micro${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_MEMORY_MANAGER "luci_interpreter_micro_memory_manager${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_CORE "luci_interpreter_core_micro${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_KERNELS "luci_interpreter_kernels_micro${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_LOADER "luci_interpreter_loader_micro${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_IMPORT "luci_interpreter_import_micro${LUCI_INTERPRETER_SUFFIX}")

add_subdirectory(memory_managers)
message(STATUS "LUCI INTERPRETER MEMORY MANAGER")
add_subdirectory(core)
message(STATUS "LUCI INTERPRETER CORE")
add_subdirectory(kernels)
message(STATUS "LUCI INTERPRETER KERNELS")
add_subdirectory(loader)
message(STATUS "LUCI INTERPRETER LOADER")

target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC ${LUCI_INTERPRETER_KERNELS})
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC ${LUCI_INTERPRETER_KERNELS})

message(STATUS "LUCI INTERPTER INITALIZED")

set(SOURCES
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/Interpreter.h" Interpreter.cpp TrainingOnertMicro.cpp)

add_library(${LUCI_INTERPRETER_BINARY} STATIC ${SOURCES})

target_include_directories(${LUCI_INTERPRETER_BINARY} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_BINARY} PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_BINARY}
        PUBLIC ${LUCI_INTERPRETER_MEMORY_MANAGER} ${LUCI_INTERPRETER_LOADER} ${LUCI_INTERPRETER_CORE})

install(TARGETS ${LUCI_INTERPRETER_BINARY} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
