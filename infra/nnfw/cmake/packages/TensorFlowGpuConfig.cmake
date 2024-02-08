# TensorFlowGpuConfig.cmake
macro(return_unless VAR)
if(NOT ${VAR})
  message("TensorFlowGpu: ${VAR} NOT TRUE")
  set(TensorFlowGpu_FOUND FALSE PARENT_SCOPE)
  return()
endif(NOT ${VAR})
endmacro(return_unless)

function(_Build_TfliteGpuDelagate_)
  nnas_find_package(TensorFlowSource EXACT 2.8.0 QUIET)
  return_unless(TensorFlowSource_FOUND)

  nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.8.0 QUIET)
  return_unless(TensorFlowGEMMLowpSource_FOUND)

  nnas_find_package(TensorFlowEigenSource EXACT 2.8.0 QUIET)
  return_unless(TensorFlowEigenSource_FOUND)

  nnas_find_package(Abseil REQUIRED)
  return_unless(Abseil_FOUND)

  nnas_find_package(Farmhash REQUIRED)
  return_unless(Farmhash_FOUND)

  nnas_find_package(Fp16Source REQUIRED)
  return_unless(Fp16Source_FOUND)

  nnas_find_package(VulkanSource QUIET)
  return_unless(VulkanSource_FOUND)

  nnas_find_package(Opencl_HeadersSource QUIET)
  return_unless(Opencl_HeadersSource_FOUND)

  nnas_find_package(Opengl_HeadersSource QUIET)
  return_unless(Opengl_HeadersSource_FOUND)

  nnas_find_package(Egl_HeadersSource QUIET)
  return_unless(Egl_HeadersSource_FOUND)

  nnfw_find_package(FlatBuffers EXACT 2.0 QUIET)
  return_unless(FlatBuffers_FOUND)

  if(NOT TARGET TensorFlowGpu)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLiteGpu" TensorFlowLiteGpu)
  endif()
  set(TensorFlowSource_DIR ${TensorFlowSource_DIR} PARENT_SCOPE)
  set(TensorFlowGpu_DIR ${TensorFlowGpu_DIR} PARENT_SCOPE)
endfunction(_Build_TfliteGpuDelagate_)

if(BUILD_TENSORFLOW_LITE_GPU)
  _Build_TfliteGpuDelagate_()
  set(TensorFlowGpu_FOUND TRUE PARENT_SCOPE)
else(BUILD_TENSORFLOW_LITE_GPU)
  set(TensorFlowGpu_FOUND FALSE PARENT_SCOPE)
endif(BUILD_TENSORFLOW_LITE_GPU)
