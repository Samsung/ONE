if (NOT BUILD_TRIX_LOADER)
  return()
endif ()

set(SOURCES src/trix_loader_base.cc)

nnfw_find_package(TRIXEngine EXACT 2.5.0 QUIET)
if(TRIXEngine_FOUND)
  message(WARNING "TRIXEngine found")
  list(APPEND SOURCES src/trix_loader.cc)
else()
  message(WARNING "TRIXEngine NOT found, trix loader is dummy.")
  list(APPEND SOURCES src/trix_loader_dummy.cc)
endif(TRIXEngine_FOUND)

add_library(trix_loader STATIC ${SOURCES})
set_target_properties(trix_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(trix_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(trix_loader PRIVATE onert_core)
target_link_libraries(trix_loader PRIVATE nnfw_common nnfw_coverage)

if(TRIXEngine_FOUND)
  target_include_directories(trix_loader PUBLIC ${TRIXEngine_INCLUDE_DIR})
  target_link_libraries(trix_loader PRIVATE trix_engine) # Rename npu_engine to trix_engine
endif(TRIXEngine_FOUND)
