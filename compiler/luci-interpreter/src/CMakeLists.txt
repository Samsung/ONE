include("${LUCI_INTERPRETER_PAL_DIR}/pal.cmake")

initialize_pal()

if (NOT PAL_INITIALIZED)
  message("PAL Failed to initialize, skip luci-interpreter")
  return()
endif()

message(STATUS "LUCI INTERPRETER BEGIN")

set(LUCI_INTERPRETER_BINARY "luci_interpreter${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_CORE "luci_interpreter_core${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_KERNELS "luci_interpreter_kernels${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_LOADER "luci_interpreter_loader${LUCI_INTERPRETER_SUFFIX}")
set(LUCI_INTERPRETER_IMPORT "luci_interpreter_import${LUCI_INTERPRETER_SUFFIX}")

add_subdirectory(core)
message(STATUS "LUCI INTERPRETER CORE")
add_subdirectory(kernels)
message(STATUS "LUCI INTERPRETER KERNELS")
add_subdirectory(loader)
message(STATUS "LUCI INTERPRETER LOADER")
add_subdirectory(import)
message(STATUS "LUCI INTERPRETER IMPORT")

message(STATUS "LUCI INTERPTER INITALIZED")

set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/Interpreter.h"
    Interpreter.cpp "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/SimpleMemoryManager.h" SimpleMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/TestMemoryManager.h" TestMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/BuddyMemoryManager.h" BuddyMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/StaticMemoryManager.h" StaticMemoryManager.cpp)

if (NOT LUCI_INTERPRETER_STATIC)
  add_library(${LUCI_INTERPRETER_BINARY} SHARED ${SOURCES})
else ()
  add_library(${LUCI_INTERPRETER_BINARY} STATIC ${SOURCES})
endif ()

set(TEST_SOURCES SimpleMemoryManager.test.cpp BuddyMemoryManager.test.cpp)

target_include_directories(${LUCI_INTERPRETER_BINARY} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_BINARY} PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_BINARY}
    PUBLIC luci_lang ${LUCI_INTERPRETER_LOADER} ${LUCI_INTERPRETER_CORE}
    PRIVATE nncc_common)

install(TARGETS ${LUCI_INTERPRETER_BINARY} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_interpreter_memory_manager_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_memory_manager_test ${LUCI_INTERPRETER_BINARY})
