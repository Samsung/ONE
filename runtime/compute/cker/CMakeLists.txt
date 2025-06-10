nnfw_find_package(Eigen REQUIRED)
nnfw_find_package(GEMMLowp REQUIRED)
nnfw_find_package(Ruy REQUIRED)

add_library(nnfw_lib_cker INTERFACE)
target_link_libraries(nnfw_lib_cker INTERFACE eigen)
target_link_libraries(nnfw_lib_cker INTERFACE gemmlowp)
target_link_libraries(nnfw_lib_cker INTERFACE ruy)
target_link_libraries(nnfw_lib_cker INTERFACE ruy_instrumentation)
target_compile_definitions(nnfw_lib_cker INTERFACE USE_RUY_GEMV)
if(PROFILE_RUY)
  target_link_libraries(nnfw_lib_cker INTERFACE ruy_profiler)
endif(PROFILE_RUY)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  target_compile_definitions(nnfw_lib_cker INTERFACE CKER_X86_PLATFORM)
endif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")

target_include_directories(nnfw_lib_cker INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Workaround to avoid warning
# TODO Resolve warning
target_compile_options(nnfw_lib_cker INTERFACE -Wno-attributes)

# Set EIGEN_NEON_GEBP_NR for arm 64bit
target_compile_definitions(eigen INTERFACE EIGEN_NEON_GEBP_NR=4)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

set(TEST_CKER test_cker)

file(GLOB_RECURSE TESTS "src/*.test.cc")

add_executable(${TEST_CKER} ${TESTS})

target_link_libraries(${TEST_CKER} nnfw_lib_cker)
target_link_libraries(${TEST_CKER} nnfw_coverage)
target_link_libraries(${TEST_CKER} gtest gtest_main ${LIB_PTHREAD})

add_test(${TEST_CKER} ${TEST_CKER})
install(TARGETS ${TEST_CKER} DESTINATION unittest)
