nnfw_find_package(Ruy REQUIRED)

add_library(nnfw_lib_ruy INTERFACE)
target_link_libraries(nnfw_lib_ruy INTERFACE ruy)
target_link_libraries(nnfw_lib_ruy INTERFACE ruy_instrumentation)
target_compile_definitions(nnfw_lib_ruy INTERFACE USE_RUY_GEMV)
if(PROFILE_RUY)
  target_link_libraries(nnfw_lib_ruy INTERFACE ruy_profiler)
endif(PROFILE_RUY)

target_include_directories(nnfw_lib_ruy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
