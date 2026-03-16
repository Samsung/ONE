option(ENCO_CAFFE_TEST "Enable enco test for caffe" ON)

if(NOT ENCO_CAFFE_TEST)
  return()
endif(NOT ENCO_CAFFE_TEST)

# TODO Use REQUIRED if supported
nncc_find_resource(BVLCCaffeTests)

if(NOT BVLCCaffeTests_FOUND)
  message(FATAL_ERROR "Fail to find BVLCCaffeTests")
endif(NOT BVLCCaffeTests_FOUND)

# TESTCASE_BASE_DIR indicates where all the testcases are located
set(TESTCASE_BASE_DIR "${BVLCCaffeTests_DIR}")

###
### Common function(s)
###
function(get_test_configuration PREFIX)
  set(PROTOTXT_FILE "${PREFIX}.prototxt")
  set(PROTOTXT_FILE "${PROTOTXT_FILE}" PARENT_SCOPE)
  set(PROTOTXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PROTOTXT_FILE}" PARENT_SCOPE)
  set(CAFFEMODEL_FILE "${PREFIX}.caffemodel")
  set(CAFFEMODEL_FILE "${CAFFEMODEL_FILE}" PARENT_SCOPE)
  set(CAFFEMODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CAFFEMODEL_FILE}" PARENT_SCOPE)
  set(SOURCE_FILE ${PREFIX}.cpp)
  set(SOURCE_FILE "${SOURCE_FILE}" PARENT_SCOPE)
  set(SOURCE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE}" PARENT_SCOPE)
  set(ASM_FILE ${PREFIX}.embed.S)
  set(ASM_FILE "${ASM_FILE}" PARENT_SCOPE)
  set(ASM_PATH "${CMAKE_CURRENT_BINARY_DIR}/${ASM_FILE}" PARENT_SCOPE)
  set(BIN_FILE ${PREFIX}.bin)
  set(BIN_FILE "${BIN_FILE}" PARENT_SCOPE)
  set(BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${BIN_FILE}" PARENT_SCOPE)
endfunction(get_test_configuration)

###
### Prepare test(s)
###
if(NOT TARGET caffegen)
  return()
endif(NOT TARGET caffegen)

if(NOT TARGET enco_caffe_frontend)
  return()
endif(NOT TARGET enco_caffe_frontend)

# TODO Use "whitelist" instead
#
# WHY?
#
#   Tests are now shared by multiple frameworks (not private), and thus
#   some tests may be unsupported.
#
file(GLOB MODELS RELATIVE "${TESTCASE_BASE_DIR}" "${TESTCASE_BASE_DIR}/*/test.prototxt")

foreach(MODEL IN ITEMS ${MODELS})
  get_filename_component(PREFIX ${MODEL} DIRECTORY)
  get_test_configuration(${PREFIX})

  set(MODEL_FILE ${TESTCASE_BASE_DIR}/${MODEL})

  # Copy prototxt
  # TODO Fix indentation
  add_custom_command(OUTPUT ${PROTOTXT_PATH}
                     COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_FILE}" "${PROTOTXT_PATH}"
                     DEPENDS "${MODEL_FILE}"
                    COMMENT "Generating ${PROTOTXT_FILE}")

  # Generate caffemodel
  # TODO Fix indentation
  add_custom_command(OUTPUT ${CAFFEMODEL_PATH}
                     COMMAND cat ${PROTOTXT_PATH}
                        | GLOG_minloglevel=2 $<TARGET_FILE:caffegen> init
                        | GLOG_minloglevel=2 $<TARGET_FILE:caffegen> encode
                        > ${CAFFEMODEL_PATH}
                    DEPENDS caffegen ${PROTOTXT_PATH}
                    COMMENT "Generating ${CAFFEMODEL_FILE}")

  # Generate C++ code
  # TODO Fix indentation
  add_custom_command(OUTPUT ${SOURCE_PATH} ${ASM_PATH} ${BIN_PATH}
                     COMMAND $<TARGET_FILE:enco-cli>
                          --frontend $<TARGET_FILE:enco_caffe_frontend>
                          --frontend-arg ${PROTOTXT_FILE}
                          --frontend-arg ${CAFFEMODEL_FILE}
                          --backend-arg ${PREFIX}
                    DEPENDS enco-cli enco_caffe_frontend ${CAFFEMODEL_PATH}
                    COMMENT "Generating ${SOURCE_FILE}")
  set_source_files_properties(${ASM_PATH} PROPERTIES GENERATED TRUE LANGUAGE C)

  list(APPEND CANDIDATES ${PREFIX})
endforeach(MODEL)

###
### Inference test
###
if(NOT TARGET ann_ref_static)
  return()
endif(NOT TARGET ann_ref_static)

find_program(H5DIFF h5diff)

if (NOT H5DIFF)
  return()
endif(NOT H5DIFF)

message(STATUS "Enable enco(caffe) inference test")

foreach(PREFIX IN ITEMS ${CANDIDATES})
  if(NOT EXISTS "${TESTCASE_BASE_DIR}/${PREFIX}/INFERENCE")
    continue()
  endif()

  get_test_configuration(${PREFIX})

  set(BINDER_TARGET enco_caffe_test_${PREFIX}_binder)

  # Compile nnkit binder (from generated C++ code)
  add_library(${BINDER_TARGET} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../binder.cpp ${SOURCE_PATH} ${ASM_PATH})
  target_include_directories(${BINDER_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${BINDER_TARGET} nnkit_intf_backend)
  target_link_libraries(${BINDER_TARGET} ann_api)
  target_link_libraries(${BINDER_TARGET} ann_ref_static)
  set_target_properties(${BINDER_TARGET} PROPERTIES OUTPUT_NAME ${PREFIX})

  list(APPEND TESTS ${PREFIX})
endforeach(PREFIX)

# Run tests
add_test(NAME enco_test_caffe
         COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/runall.sh"
                            $<TARGET_FILE:nnkit-run>
                            $<TARGET_FILE:nnkit_caffe_backend>
                            $<TARGET_FILE:nnkit_randomize_action>
                            $<TARGET_FILE:nnkit_HDF5_export_action>
                            $<TARGET_FILE:nnkit_HDF5_import_action>
                            "${CMAKE_CURRENT_BINARY_DIR}"
                            ${TESTS})
