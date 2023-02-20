set(SOURCES
        "SimpleMemoryManager.h" SimpleMemoryManager.cpp
        "TestMemoryManager.h" TestMemoryManager.cpp
        "BuddyMemoryManager.h" BuddyMemoryManager.cpp
        "StaticMemoryManager.h" StaticMemoryManager.cpp)

add_library(${LUCI_INTERPRETER_MEMORY_MANAGER} STATIC ${SOURCES})
target_include_directories(${LUCI_INTERPRETER_MEMORY_MANAGER} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_MEMORY_MANAGER} PUBLIC "luci_micro_circle_reader${READER_SUFFIX}")
target_link_libraries(${LUCI_INTERPRETER_MEMORY_MANAGER} PUBLIC luci_micro_circle_schema)

if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

set(TEST_SOURCES BuddyMemoryManager.test.cpp)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(buddy_manager_test_micro ${TEST_SOURCES})
target_link_libraries(buddy_manager_test_micro ${LUCI_INTERPRETER_BINARY})
