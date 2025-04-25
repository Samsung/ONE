function(_Eigen_import)
  nnfw_find_package(EigenSource QUIET)

  if(NOT EigenSource_FOUND)
    set(Eigen_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT EigenSource_FOUND)

  if(NOT TARGET eigen)
    add_library(eigen INTERFACE)
    target_include_directories(eigen SYSTEM INTERFACE "${EigenSource_DIR}")
    # Add EIGEN_MPL2_ONLY to remove license issue posibility
    target_compile_definitions(eigen INTERFACE EIGEN_MPL2_ONLY)
    # Some used Eigen functions makes deprecated declarations warning
    target_compile_options(eigen INTERFACE -Wno-deprecated-declarations)
  endif(NOT TARGET eigen)

  set(Eigen_FOUND TRUE PARENT_SCOPE)
endfunction(_Eigen_import)

_Eigen_import()
