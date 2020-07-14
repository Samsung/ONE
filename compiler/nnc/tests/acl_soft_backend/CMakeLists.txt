function(acl_warn MESSAGE)
  message(WARNING "The ACL backend system tests will not be built: ${MESSAGE}")
endfunction(acl_warn)

# Path to the folder where the Odroid root folder is either mounted or copied.
if(NOT DEFINED ENV{ODROID_MIRROR_DIR})
  acl_warn("the ODROID_MIRROR_DIR environment variable was not defined.\n\
    As a cross-build is done, this variable should point to a directory\n\
    which is either mounted to (with e.g. SSHFS) or contains a copy of the target\n\
    (e.g. Odroid XU4) device file system.")
  return()
endif()

# Path to the ACL library root on the Odroid device.
if(NOT DEFINED ENV{ODROID_ACL_DIR})
  acl_warn("the ODROID_ACL_DIR environment variable was not defined.\n\
    As a cross-build is done, this variable should contain the path to the root directory\n\
    of the Arm Compute Library on the target (e.g. Odroid XU4) device.")
  return()
endif()

# Path to the HDF5 library on the Odroid.
# It must most likely be: /usr/lib/arm-linux-gnueabihf/hdf5/serial
if(NOT DEFINED ENV{ODROID_H5_DIR})
  acl_warn("the ODROID_H5_DIR environment variable was not defined.\n\
    As a cross-build is done, this variable should contain the path to the root directory\n\
    of the HDF5 library on the target (e.g. Odroid XU4) device.\n\
    Often it would be the: /usr/lib/arm-linux-gnueabihf/hdf5/serial directory.")
  return()
endif()

# TODO This will be nnas_find_package
find_package(HDF5 COMPONENTS CXX QUIET CONFIG)

if(NOT HDF5_FOUND)
  find_package(HDF5 COMPONENTS CXX REQUIRED MODULE)
endif(NOT HDF5_FOUND)

nnas_find_package(GTest REQUIRED)

# Provide the test suite with the information where to locate executalbes to run etc.
configure_file(BuildInfo.h.in BuildInfo.h)

GTest_AddTest(nnc_acl_soft_backend_system_test AclCppOperations.cpp)
add_dependencies(nnc_acl_soft_backend_system_test tensor_gen caffegen nnkit-run)
target_include_directories(nnc_acl_soft_backend_system_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    ${HDF5_INCLUDE_DIRS})
target_link_libraries(nnc_acl_soft_backend_system_test ${HDF5_CXX_LIBRARIES})

file(GLOB MODELS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "models/*.prototxt")

# Loop over all the existing system test models.
foreach(MODEL IN ITEMS ${MODELS})
  # Set the model-related variables used inside the loop.
  set(MODEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL})
  get_filename_component(MODEL_NAME ${MODEL} NAME_WE)
  set(MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME})
  set(COPIED_MODEL_FILE ${MODEL_DIR}/${MODEL_NAME}.prototxt)
  set(INITIALIZED_MODEL_FILE ${MODEL_DIR}/${MODEL_NAME}.prototxt.weights)
  set(ENCODED_MODEL_FILE ${MODEL_DIR}/${MODEL_NAME}.caffemodel)
  set(INPUT_FILE ${MODEL_DIR}/in_${MODEL_NAME}_caffe.hdf5)
  set(REFERENCE_OUTPUT ${MODEL_DIR}/ref.hdf5)
  set(MODEL_DIR_TARGET nnc_acl_soft_backend_${MODEL_NAME}_dir)
  set(COPIED_MODEL_TARGET nnc_acl_soft_backend_${MODEL_NAME}_prototxt)
  set(INITIALIZED_MODEL_TARGET nnc_acl_soft_backend_${MODEL_NAME}_weights)
  set(ENCODED_MODEL_TARGET nnc_acl_soft_backend_${MODEL_NAME}_caffemodel)
  set(INPUT_TARGET nnc_acl_soft_backend_${MODEL_NAME}_input)
  set(REFERENCE_OUTPUT_TARGET nnc_acl_soft_backend_${MODEL_NAME}_reference_output)
  set(MAIN_FILE_TARGET nnc_acl_soft_backend_${MODEL_NAME}_main)
  set(ARTIFACT_TARGET_CMAKE nnc_acl_soft_backend_${MODEL_NAME}_cmake)
  set(ARTIFACT_TARGET_TOOLCHAIN nnc_acl_soft_backend_${MODEL_NAME}_toolchain)
  set(ARTIFACT_TARGET nnc_acl_soft_backend_${MODEL_NAME}_artifact)
  set(ARTIFACT_TARGET_MAKE nnc_acl_soft_backend_${MODEL_NAME}_artifact_make)
  set(ARTIFACT_TARGET_BINARY nnc_acl_soft_backend_${MODEL_NAME}_artifact_binary)

  # Create a directrory for handling the model in the binary directory.
  add_custom_target(${MODEL_DIR_TARGET} ALL ${CMAKE_COMMAND} -E make_directory ${MODEL_DIR})

  # Copy the model prototxt to the model build directory.
  add_custom_target(${COPIED_MODEL_TARGET} ALL ${CMAKE_COMMAND} -E copy ${MODEL_FILE} ${COPIED_MODEL_FILE})

  # Copy the artifact main and project files into the model subfolder inside the binary directory.
  add_custom_target(${MAIN_FILE_TARGET} ALL ${CMAKE_COMMAND}
      -E copy ${CMAKE_CURRENT_SOURCE_DIR}/artifact_cmake/main.cpp ${MODEL_DIR}/main.cpp)

  # Copy the artifact CMakeLists.txt to the artifact build directory.
  add_custom_target(${ARTIFACT_TARGET_CMAKE} ALL ${CMAKE_COMMAND}
      -E copy ${CMAKE_CURRENT_SOURCE_DIR}/artifact_cmake/CMakeLists.txt ${MODEL_DIR}/CMakeLists.txt)

  # Copy the artifact toolchain file to the artifact build directory.
  add_custom_target(${ARTIFACT_TARGET_TOOLCHAIN} ALL ${CMAKE_COMMAND}
      -E copy ${CMAKE_CURRENT_SOURCE_DIR}/artifact_cmake/odroid.cmake ${MODEL_DIR}/odroid.cmake)

  # Initialize the model with weights.
  add_custom_target(${INITIALIZED_MODEL_TARGET} ALL cat ${COPIED_MODEL_FILE} | GLOG_minloglevel=2
      $<TARGET_FILE:caffegen> init > ${INITIALIZED_MODEL_FILE}
      DEPENDS ${COPIED_MODEL_TARGET})

  # Encode the model.
  add_custom_target(${ENCODED_MODEL_TARGET} ALL cat ${INITIALIZED_MODEL_FILE} | GLOG_minloglevel=2
      $<TARGET_FILE:caffegen> encode > ${ENCODED_MODEL_FILE}
      DEPENDS ${INITIALIZED_MODEL_TARGET})

  add_custom_target(${INPUT_TARGET} ALL $<TARGET_FILE:tensor_gen> data ${MODEL_NAME} 1 3 4 8
      WORKING_DIRECTORY ${MODEL_DIR})

    message("INPUT_FILE = ${INPUT_FILE}")

  # Generate the 'reference' output with NNKIT.
  add_custom_target(${REFERENCE_OUTPUT_TARGET} ALL $<TARGET_FILE:nnkit-run>
      --backend $<TARGET_FILE:nnkit_caffe_backend> --backend-arg ${INITIALIZED_MODEL_FILE}
      --pre $<TARGET_FILE:nnkit_HDF5_import_action> --pre-arg ${INPUT_FILE}
      --post $<TARGET_FILE:nnkit_HDF5_export_action> --post-arg ${REFERENCE_OUTPUT}
      DEPENDS $<TARGET_FILE:nnkit-run> ${INPUT_TARGET} ${INITIALIZED_MODEL_TARGET})

  # Generate an artifact from the model.
  add_custom_target(${ARTIFACT_TARGET} ALL $<TARGET_FILE:nnc>
      --caffe -m ${ENCODED_MODEL_FILE} -o AclArtifact -d ${MODEL_DIR} --target=arm-gpu-c++
      DEPENDS $<TARGET_FILE:nnc> ${ENCODED_MODEL_FILE})

  # Generate a Makefile for the artifact cross-building.
  add_custom_target(${ARTIFACT_TARGET_MAKE} ALL cmake .
      -DCMAKE_TOOLCHAIN_FILE=odroid.cmake
      DEPENDS ${ARTIFACT_TARGET_CMAKE} ${ARTIFACT_TARGET_TOOLCHAIN} ${ARTIFACT_TARGET}
      WORKING_DIRECTORY ${MODEL_DIR})

  # Cross-build the artifact with the generated Makefile.
  add_custom_target(${ARTIFACT_TARGET_BINARY} ALL make
      DEPENDS ${ARTIFACT_TARGET_MAKE}
      WORKING_DIRECTORY ${MODEL_DIR})
endforeach(MODEL)
