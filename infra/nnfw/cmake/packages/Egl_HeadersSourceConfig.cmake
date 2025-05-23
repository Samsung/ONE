function(_Egl_HeadersSource_import)
  if(NOT DOWNLOAD_EGL_HEADERS)
    set(Egl_HeadersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EGL_HEADERS)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(EGL_HEADERS_URL ${EXTERNAL_DOWNLOAD_SERVER}/KhronosGroup/EGL-Registry/archive/649981109e263b737e7735933c90626c29a306f2.zip)

  ExternalSource_Download(EGL_HEADERS
    DIRNAME EGL_HEADERS
    URL ${EGL_HEADERS_URL})

  set(Egl_HeadersSource_DIR ${EGL_HEADERS_SOURCE_DIR} PARENT_SCOPE)
  set(Egl_HeadersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_Egl_HeadersSource_import)

_Egl_HeadersSource_import()
