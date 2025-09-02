function(_TensorFlowThreadpoolSource_import)
  if(NOT DOWNLOAD_THREADPOOL)
    set(TensorFlowThreadpoolSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_THREADPOOL)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Exact version used by TensorFlow v2.19.0.
  # See tensorflow/workspace2.bzl
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(TENSORFLOW_2_19_0_THREADPOOL_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/pthreadpool/archive/4fe0e1e183925bf8cfa6aae24237e724a96479b8.zip)

  ExternalSource_Download(THREADPOOL DIRNAME TENSORFLOW-2.19.0-THREADPOOL ${TENSORFLOW_2_19_0_THREADPOOL_URL})

  set(TensorFlowThreadpoolSource_DIR ${THREADPOOL_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowThreadpoolSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowThreadpoolSource_import)

_TensorFlowThreadpoolSource_import()
