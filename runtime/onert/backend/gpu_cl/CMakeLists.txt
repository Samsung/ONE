set(LIB_ONERT_BACKEND_GPU_CL onert_backend_gpu_cl)

# TODO
# Change find_package OPENCL header
nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

nnas_find_package(Farmhash QUIET)
if(NOT Farmhash_FOUND)
  return()
endif(NOT Farmhash_FOUND)

file(GLOB_RECURSE SOURCES "*.cc")

find_package(absl)

add_library(${LIB_ONERT_BACKEND_GPU_CL} SHARED ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE
  absl::algorithm
  absl::base
  absl::debugging
  absl::flat_hash_map
  absl::flags
  absl::memory
  absl::meta
  absl::numeric
  absl::random_random
  absl::strings
  absl::status
  absl::synchronization
  absl::time
  absl::utility
)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE dl)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE farmhash)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_GPU_CL} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_GPU_CL} PROPERTIES OUTPUT_NAME backend_gpu_cl)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_GPU_CL} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_GPU_CL}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_GPU_CL} DESTINATION lib)
