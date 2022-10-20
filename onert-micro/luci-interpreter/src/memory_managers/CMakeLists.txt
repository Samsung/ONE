set(SOURCES
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/memory_managers/SimpleMemoryManager.h" SimpleMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/memory_managers/TestMemoryManager.h" TestMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/memory_managers/BuddyMemoryManager.h" BuddyMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/memory_managers/StaticMemoryManager.h" StaticMemoryManager.cpp)

add_library(${LUCI_INTERPRETER_MEMORY_MANAGER} STATIC ${SOURCES})
target_include_directories(${LUCI_INTERPRETER_MEMORY_MANAGER} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")

if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

set(TEST_SOURCES BuddyMemoryManager.test.cpp)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(buddy_manager_test_micro ${TEST_SOURCES})
target_link_libraries(buddy_manager_test_micro ${LUCI_INTERPRETER_BINARY})
