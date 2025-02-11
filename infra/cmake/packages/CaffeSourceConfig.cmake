function(_CaffeSource_import)
  if(NOT DOWNLOAD_CAFFE)
    set(CaffeSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_CAFFE)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(CAFFE_URL ${EXTERNAL_DOWNLOAD_SERVER}/BVLC/caffe/archive/1.0.tar.gz)

  ExternalSource_Download(CAFFE ${CAFFE_URL} PATCH ${CMAKE_CURRENT_LIST_DIR}/CaffeSource.patch)

  set(CaffeSource_DIR ${CAFFE_SOURCE_DIR} PARENT_SCOPE)
  set(CaffeSource_FOUND ${DOWNLOAD_CAFFE} PARENT_SCOPE)
endfunction(_CaffeSource_import)

_CaffeSource_import()
