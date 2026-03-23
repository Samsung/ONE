nnas_find_package(Eigen QUIET)

if(NOT Eigen_FOUND)
  return()
endif(NOT Eigen_FOUND)

nnas_find_package(GEMMLowp QUIET)

if(NOT GEMMLowp_FOUND)
  return()
endif(NOT GEMMLowp_FOUND)

nnas_include(TargetRequire)

TargetRequire_Assert(ann_api eigen gemmlowp)

file(GLOB_RECURSE SOURCES "src/*.cpp")

function(ann_ref_configure TARGET)
  target_include_directories(${TARGET} PRIVATE src)
  target_link_libraries(${TARGET} PRIVATE ann_api)
  target_link_libraries(${TARGET} PRIVATE eigen)
  target_link_libraries(${TARGET} PRIVATE gemmlowp)
endfunction(ann_ref_configure)

add_library(ann_ref_static STATIC ${SOURCES})
set_target_properties(ann_ref_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
ann_ref_configure(ann_ref_static)

add_library(ann_ref_shared SHARED ${SOURCES})
set_target_properties(ann_ref_shared PROPERTIES OUTPUT_NAME neuralnetworks)
ann_ref_configure(ann_ref_shared)
