file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

if (NOT LUCI_LIBRARY_TYPE)
    set(LUCI_LIBRARY_TYPE "SHARED")
endif(NOT LUCI_LIBRARY_TYPE)

add_library(luci_plan ${LUCI_LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_plan PRIVATE src)
target_include_directories(luci_plan PUBLIC include)
target_link_libraries(luci_plan PUBLIC loco)
target_link_libraries(luci_plan PUBLIC luci_lang)

install(TARGETS luci_plan DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_plan_test ${TESTS})
target_link_libraries(luci_plan_test luci_plan)
