function(_TensorFlowThreadpoolSource_import)
  if(NOT DOWNLOAD_THREADPOOL)
    set(TensorFlowThreadpoolSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_THREADPOOL)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.12.1.
  # See tensorflow/workspace2.bzl
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_12_1_THREADPOOL_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/pthreadpool/archive/b8374f80e42010941bda6c85b0e3f1a1bd77a1e0.zip)

  ExternalSource_Download(THREADPOOL DIRNAME TENSORFLOW-2.12.1-THREADPOOL ${TENSORFLOW_2_12_1_THREADPOOL_URL})

  set(TensorFlowThreadpoolSource_DIR ${THREADPOOL_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowThreadpoolSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowThreadpoolSource_import)

_TensorFlowThreadpoolSource_import()
