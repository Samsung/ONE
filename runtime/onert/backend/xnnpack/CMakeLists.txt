set(LIB_ONERT_BACKEND_XNNPACK onert_backend_xnnpack)

nnfw_find_package(Xnnpack REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_XNNPACK} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE pthreadpool)
target_link_libraries(${LIB_ONERT_BACKEND_XNNPACK} PRIVATE XNNPACK)

set_target_properties(${LIB_ONERT_BACKEND_XNNPACK} PROPERTIES OUTPUT_NAME backend_xnnpack)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_XNNPACK} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_XNNPACK}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_XNNPACK} DESTINATION lib)
