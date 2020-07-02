# NOTE TensorFlow 1.12 uses eigen commit ID fd6845384b86

# find_package rejects version with commit number. Commit ID is appended to the package name
# as a workaround.
#
# TODO Find a better way
function(_Eigen_import)
  nnas_find_package(EigenSource-fd6845384b86 QUIET)

  if(NOT EigenSource-fd6845384b86_FOUND)
    set(Eigen-fd6845384b86_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT EigenSource-fd6845384b86_FOUND)

  if(NOT TARGET eigen-fd6845384b86)
    add_library(eigen-fd6845384b86 INTERFACE)
    target_include_directories(eigen-fd6845384b86 INTERFACE "${EigenSource_DIR}")
    # Add EIGEN_MPL2_ONLY to remove license issue posibility
    target_compile_definitions(eigen-fd6845384b86 INTERFACE EIGEN_MPL2_ONLY)
  endif(NOT TARGET eigen-fd6845384b86)

  set(Eigen-fd6845384b86_FOUND TRUE PARENT_SCOPE)
endfunction(_Eigen_import)

_Eigen_import()
