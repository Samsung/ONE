function(_OouraFFT_build)
  nnas_find_package(OouraFFTSource QUIET)

  if(NOT OouraFFTSource_FOUND)
    set(OouraFFT_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT OouraFFTSource_FOUND)

  # TFLite requires fft2d_fftsg2d (oourafft::fftsg2d)
  if(NOT TARGET oourafft::fftsg2d)
    add_library(fft2d_fftsg2d STATIC
      ${OouraFFTSource_DIR}/fftsg2d.c
      ${OouraFFTSource_DIR}/fftsg.c
    )
    add_library(oourafft::fftsg2d ALIAS fft2d_fftsg2d)
  endif(NOT TARGET oourafft::fftsg2d)

  set(OouraFFT_FOUND TRUE PARENT_SCOPE)
endfunction(_OouraFFT_build)

_OouraFFT_build()
