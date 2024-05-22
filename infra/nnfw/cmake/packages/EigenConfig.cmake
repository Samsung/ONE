function(_Eigen_import)
  nnas_find_package(TensorFlowEigenSource EXACT 2.16.1 QUIET)

  if(NOT TensorFlowEigenSource_FOUND)
    set(Eigen_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT TensorFlowEigenSource_FOUND)

  if(NOT TARGET eigen)
    add_library(eigen INTERFACE)
    target_include_directories(eigen SYSTEM INTERFACE "${TensorFlowEigenSource_DIR}")
    # Add EIGEN_MPL2_ONLY to remove license issue posibility
    target_compile_definitions(eigen INTERFACE EIGEN_MPL2_ONLY)
    # Some used Eigen functions makes deprecated declarations warning
    target_compile_options(eigen INTERFACE -Wno-deprecated-declarations)
  endif(NOT TARGET eigen)

  set(Eigen_FOUND TRUE PARENT_SCOPE)
endfunction(_Eigen_import)

_Eigen_import()
