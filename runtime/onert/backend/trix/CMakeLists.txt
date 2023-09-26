set(LIB_ONERT_BACKEND_TRIX onert_backend_trix)

nnfw_find_package(TRIXEngine QUIET 2.5.0)
if(NOT TRIXEngine_FOUND)
  message(STATUS "ONERT backend: Failed to find TRIXEngine")
  return()
endif(NOT TRIXEngine_FOUND)
message(STATUS "ONERT backend: Found TRIXEngine")

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_TRIX} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE trix_engine)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_TRIX} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_TRIX} PROPERTIES OUTPUT_NAME backend_trix)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_TRIX} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_TRIX}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_TRIX} DESTINATION lib)
