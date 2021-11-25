#
# Download Tensorflow 2.4.1, use gpu delegate codes only
#

function(_TensorFlowGpuSource_Import)
  SET(PATCH_FILE_CHECK "20211014")
  SET(DATE_STAMP_PATH "${NNAS_EXTERNALS_DIR}/TENSORFLOW_GPU.stamp")

  set(PATCH_DONE FALSE)
  if(EXISTS ${DATE_STAMP_PATH})
    file(STRINGS ${DATE_STAMP_PATH} OBTAINED_CONTENT)
    if(${OBTAINED_CONTENT} STREQUAL "${PATCH_FILE_CHECK}")
      set(PATCH_DONE "TRUE")
    endif()
  endif()
  
  if(${PATCH_DONE} STREQUAL "TRUE")
    message(STATUS "Skip downloading TensorFlowGpuSource")
    set(TENSORFLOWGPU_SOURCE_DIR "${NNAS_EXTERNALS_DIR}/TENSORFLOW_GPU" PARENT_SCOPE)
    set(TensorFlowGpuSource_DIR "${TensorFlowGpuSource_DIR}" PARENT_SCOPE)
    set(TensorFlowGpuSource_FOUND TRUE PARENT_SCOPE)
    return()
  else(${PATCH_DONE} STREQUAL "TRUE")
    # PATCH_DONE FALSE
    message(STATUS "TensorFlowGpuSource patch not found!")
  endif(${PATCH_DONE} STREQUAL "TRUE")

  # Download TFLite Source Code
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)
  envoption(TENSORFLOW_2_4_1_URL https://github.com/tensorflow/tensorflow/archive/v2.4.1.tar.gz)
  ExternalSource_Download(TFLITE_GPU_DELEGATE DIRNAME TENSORFLOW-2.4.1 ${TENSORFLOW_2_4_1_URL})

  # Patch for non used codes on onert backend/gpu_cl
  # ToDo: Do it more simpler
  set(TENSORFLOWGPU_SOURCE_DIR "${NNAS_EXTERNALS_DIR}/TENSORFLOW_GPU")

  # remove & copy gpu delegate source codes only
  if(EXISTS ${TENSORFLOWGPU_SOURCE_DIR})
    file(REMOVE_RECURSE "${TENSORFLOWGPU_SOURCE_DIR}")
  endif()

  file(MAKE_DIRECTORY "${TENSORFLOWGPU_SOURCE_DIR}")
  execute_process(
    WORKING_DIRECTORY "${TFLITE_GPU_DELEGATE_SOURCE_DIR}"
    COMMAND bash -c "cp -r --parents ./tensorflow/lite/delegates/gpu ../TENSORFLOW_GPU"
  )

  # Create Stamp
  set(_remove_path "${TENSORFLOWGPU_SOURCE_DIR}.stamp")
  if(EXISTS ${_remove_path})
    file(REMOVE ${_remove_path})
  endif()
  execute_process(
    WORKING_DIRECTORY "${NNAS_EXTERNALS_DIR}/TENSORFLOW_GPU"
    COMMAND bash -c "patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/TensorFlowGpuSource/patch_for_gpu_cl_build.patch"
  )
  file(WRITE ${DATE_STAMP_PATH} "${PATCH_FILE_CHECK}")
  set(TENSORFLOWGPU_SOURCE_DIR "${TENSORFLOWGPU_SOURCE_DIR}" PARENT_SCOPE)
  set(TensorFlowGpuSource_DIR "${TensorFlowGpuSource_DIR}" PARENT_SCOPE)
  set(TensorFlowGpuSource_FOUND TRUE PARENT_SCOPE)

  execute_process(
    WORKING_DIRECTORY "${NNAS_EXTERNALS_DIR}"
    COMMAND bash -c "rm -rf ${TFLITE_GPU_DELEGATE_SOURCE_DIR}.stamp"
    COMMAND bash -c "rm -rf ${TFLITE_GPU_DELEGATE_SOURCE_DIR}"
  )
endfunction(_TensorFlowGpuSource_Import)

if(NOT TensorFlowGpuSource_FOUND)
   _TensorFlowGpuSource_Import()
else()
  set(TensorFlowGpuSource_FOUND FALSE PARENT_SCOPE)
endif(NOT TensorFlowGpuSource_FOUND)
