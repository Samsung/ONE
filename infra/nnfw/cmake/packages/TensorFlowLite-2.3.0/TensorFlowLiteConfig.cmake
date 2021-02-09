if(BUILD_TENSORFLOW_LITE_2_3_0)
  macro(return_unless VAR)
  if(NOT ${VAR})
    message("TFLiteVanillaRun: ${VAR} NOT TRUE")
    set(TensorFlowLite_2_3_0_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${VAR})
  endmacro(return_unless)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  nnas_find_package(TensorFlowSource EXACT 2.3.0 QUIET)
  return_unless(TensorFlowSource_FOUND)

  # Below urls come from https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/tensorflow/workspace.bzl
  nnas_find_package(AbseilSource QUIET)
  return_unless(AbseilSource_FOUND)
  nnfw_find_package(Eigen QUIET)
  return_unless(Eigen_FOUND)
  nnas_find_package(Farmhash QUIET)
  return_unless(Farmhash_FOUND)
  nnfw_find_package(FlatBuffers QUIET)
  return_unless(FlatBuffers_FOUND)
  nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.3.0 QUIET)
  return_unless(TensorFlowGEMMLowpSource_FOUND)
  nnas_find_package(OouraFFTSource QUIET)
  return_unless(OouraFFTSource_FOUND)
  nnfw_find_package(Ruy QUIET)
  return_unless(Ruy_FOUND)

  # TensorFlow Lite requires FP16 library's header only
  nnas_find_package(Fp16Source QUIET)
  return_unless(Fp16Source_FOUND)

  # Optional packages
  nnas_find_package(NEON2SSESource QUIET)

  nnas_include(ExternalProjectTools)
  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLite" tflite-2.3.0)

  set(TensorFlowLite_2_3_0_FOUND TRUE)
  return()
endif()
