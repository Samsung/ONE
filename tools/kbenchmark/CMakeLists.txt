if(NOT BUILD_KBENCHMARK)
  return()
endif(NOT BUILD_KBENCHMARK)

nnfw_find_package(Nonius QUIET)

if(NOT Nonius_FOUND)
  message(STATUS "kbenchmark: Cannot found Nonius")
  return()
endif(NOT Nonius_FOUND)

nnfw_find_package(Boost REQUIRED program_options)

if(NOT Boost_FOUND)
  return()
endif(NOT Boost_FOUND)

# driver
file(GLOB_RECURSE SOURCES "*.cc")

add_executable(kbenchmark ${SOURCES})
target_compile_options(kbenchmark PRIVATE -Wno-psabi)
target_include_directories(kbenchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kbenchmark PUBLIC nonius)
target_link_libraries(kbenchmark PUBLIC dl)
target_link_libraries(kbenchmark PUBLIC pthread boost_program_options)
install(TARGETS kbenchmark DESTINATION bin)

# kernel libraries
add_subdirectory(kernels)
