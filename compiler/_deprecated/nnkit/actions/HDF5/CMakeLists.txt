nnas_find_package(HDF5 QUIET)

if(NOT HDF5_FOUND)
  return()
endif(NOT HDF5_FOUND)

add_library(nnkit_HDF5_common STATIC Common.cpp)
set_target_properties(nnkit_HDF5_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nnkit_HDF5_common ${HDF5_CXX_LIBRARIES})

add_library(nnkit_HDF5_export_action SHARED Export.cpp)
target_include_directories(nnkit_HDF5_export_action PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(nnkit_HDF5_export_action nnkit_intf_action)
target_link_libraries(nnkit_HDF5_export_action nnkit_HDF5_common)

add_library(nnkit_HDF5_import_action SHARED Import.cpp)
target_include_directories(nnkit_HDF5_import_action PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(nnkit_HDF5_import_action nnkit_intf_action)
target_link_libraries(nnkit_HDF5_import_action nnkit_HDF5_common)
