nnas_find_package(ARMCompute QUIET)

if(NOT ARMCompute_FOUND)
  message(STATUS "Check ARM Compute library extension build: need ARM Compute library")
  return()
else(NOT ARMCompute_FOUND)
  message(STATUS "Check ARM Compute library extension build: OK")
endif(NOT ARMCompute_FOUND)

set(ACL_EX_BASE ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE ACL_EX_SRCS "${ACL_EX_BASE}/*.cpp")

# generate embeded cl_kernel
execute_process (
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND bash -c "python resolve_includes.py"
)

add_library(arm_compute_ex SHARED ${ACL_EX_SRCS})
target_include_directories(arm_compute_ex PUBLIC ${ACL_EX_BASE})
target_link_libraries(arm_compute_ex PRIVATE arm_compute)
target_link_libraries(arm_compute_ex PRIVATE nnfw_common)
target_link_libraries(arm_compute_ex PRIVATE nnfw_coverage)
# Defines to enable validate check in debug build
target_compile_definitions(arm_compute_ex PRIVATE EMBEDDED_KERNELS
                                                  $<$<CONFIG:Debug>:ARM_COMPUTE_DEBUG_ENABLED ARM_COMPUTE_ASSERTS_ENABLED
                                                                    ARM_COMPUTE_LOGGING_ENABLED>)
# Validate check functions are not used on release build
# Some parameter are used for validate check function call, and these parameter may not used on release build
target_compile_options(arm_compute_ex PRIVATE $<$<NOT:$<CONFIG:Debug>>:-Wno-unused-parameter -Wno-unused-function>)
install(TARGETS arm_compute_ex DESTINATION lib)
