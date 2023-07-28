if (NOT VCONONE_VERSION)
  set(VCONONE_VERSION 0x0000000000190001)
  # NOTE order is [build patch minor major]
  # if VCONONE_VERSION is set with -D option, it will be cached
  # you may have to remove cache file if you remove -D option
endif()

configure_file(version_cfg.h.in version_cfg.h @ONLY)

set(DRIVER "driver/driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(vconone STATIC ${SOURCES})
target_include_directories(vconone PUBLIC include)
target_include_directories(vconone PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_executable(one-version ${DRIVER})
target_link_libraries(one-version vconone)
install(TARGETS one-version DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(vconone_test ${TESTS})
target_link_libraries(vconone_test vconone)
