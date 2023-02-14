function(_Eigen_import)
  nnas_find_package(EigenSource QUIET)

  if(NOT EigenSource_FOUND)
    set(Eigen_FOUND
        FALSE
        PARENT_SCOPE)
    return()
  endif(NOT EigenSource_FOUND)

  if(NOT TARGET eigen)
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE "${EigenSource_DIR}")
    # Add EIGEN_MPL2_ONLY to remove license issue posibility
    target_compile_definitions(eigen INTERFACE EIGEN_MPL2_ONLY)
  endif(NOT TARGET eigen)

  set(EigenSource_FOUND
      TRUE
      PARENT_SCOPE)
endfunction(_Eigen_import)

_Eigen_import()
