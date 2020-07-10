function(_Eigen_import)
  nnas_find_package(EigenSource QUIET)

  if(NOT EigenSource_FOUND)
    set(TensorFlowEigen_1_13_1_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT EigenSource_FOUND)

  if(NOT TARGET eigen-tf-1.13.1)
    add_library(eigen-tf-1.13.1 INTERFACE)
    target_include_directories(eigen-tf-1.13.1 SYSTEM INTERFACE "${EigenSource_DIR}")
    # Add EIGEN_MPL2_ONLY to remove license issue posibility
    target_compile_definitions(eigen-tf-1.13.1 INTERFACE EIGEN_MPL2_ONLY)
  endif(NOT TARGET eigen-tf-1.13.1)

  set(TensorFlowEigen_1_13_1_FOUND TRUE PARENT_SCOPE)
endfunction(_Eigen_import)

_Eigen_import()
