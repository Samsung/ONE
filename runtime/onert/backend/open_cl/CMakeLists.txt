# Unsupported architecture

# TODO Modify to OpenCL-header 
nnfw_find_package(ARMCompute QUIET)
if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

nnas_find_package(Abseil REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_OPEN_CL} STATIC ${SOURCES})

target_include_directories(${LIB_ONERT_BACKEND_OPEN_CL} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_ONERT_BACKEND_OPEN_CL} PUBLIC onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_OPEN_CL} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_OPEN_CL} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_OPEN_CL} PRIVATE abseil)
target_link_libraries(${LIB_ONERT_BACKEND_OPEN_CL} PRIVATE farmhash)

set_target_properties(${LIB_ONERT_BACKEND_OPEN_CL} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_ONERT_BACKEND_OPEN_CL} PROPERTIES OUTPUT_NAME backend_open_cl)
