function(_Opengl_HeadersSource_import)
  if(NOT DOWNLOAD_OPENGL_HEADERS)
    set(Opengl_HeadersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_OPENGL_HEADERS)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(OPENGL_HEADERS_URL ${EXTERNAL_DOWNLOAD_SERVER}/KhronosGroup/OpenGL-Registry/archive/0cb0880d91581d34f96899c86fc1bf35627b4b81.zip)

  ExternalSource_Download(OPENGL_HEADERS
    DIRNAME OPENGL_HEADERS
    URL ${OPENGL_HEADERS_URL})

  set(Opengl_HeadersSource_DIR ${OPENGL_HEADERS_SOURCE_DIR} PARENT_SCOPE)
  set(Opengl_HeadersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_Opengl_HeadersSource_import)

_Opengl_HeadersSource_import()
