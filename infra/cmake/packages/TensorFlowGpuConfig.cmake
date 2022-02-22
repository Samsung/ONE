# TensorFlowGpuConfig.cmake

function(_Build_TfliteGpuDelagate_)
  nnas_find_package(TensorFlowGpuSource REQUIRED)
  nnas_find_package(AbseilSource REQUIRED)
  nnas_find_package(Farmhash REQUIRED)
  nnas_find_package(Fp16Source REQUIRED)

  if(NOT TARGET TensorFlowGpu)
    nnas_include(ExternalProjectTools)
    add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLiteGpu" TensorFlowLiteGpu)
  endif()
  set(TENSORFLOWGPU_SOURCE_DIR ${TENSORFLOWGPU_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowGpu_DIR ${TensorFlowGpu_DIR} PARENT_SCOPE)
endfunction(_Build_TfliteGpuDelagate_)

if(BUILD_TENSORFLOW_LITE_GPU)
  _Build_TfliteGpuDelagate_()
  set(TensorFlowGpu_FOUND TRUE PARENT_SCOPE)
else(BUILD_TENSORFLOW_LITE_GPU)
  set(TensorFlowGpu_FOUND FALSE PARENT_SCOPE)
endif(BUILD_TENSORFLOW_LITE_GPU)
