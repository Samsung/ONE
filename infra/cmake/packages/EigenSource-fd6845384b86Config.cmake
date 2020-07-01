# find_package rejects version with commit number. Commit ID is appended to the package name
# as a workaround.
#
# TODO Find a better way
function(_import)
  if(NOT DOWNLOAD_EIGEN)
    set(EigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnas_include(ExternalSourceTools)
  nnas_include(ThirdPartyTools)

  # NOTE TensorFlow 1.12 downloads farmhash from the following URL
  ThirdParty_URL(EIGEN_URL PACKAGE Eigen VERSION fd6845384b86)

  ExternalSource_Download(EIGEN
    DIRNAME EIGEN-fd6845384b86
    CHECKSUM MD5=4c884968ede816a84c70e2cd2c81de8d
    ${EIGEN_URL}
  )

  set(EigenSource_DIR ${EIGEN_SOURCE_DIR} PARENT_SCOPE)
  set(EigenSource-fd6845384b86_FOUND TRUE PARENT_SCOPE)
endfunction(_import)

_import()
