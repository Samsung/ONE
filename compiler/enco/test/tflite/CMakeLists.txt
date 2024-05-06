option(ENCO_TFLITE_TEST "Enable enco test for TFLite" ON)

if(NOT ENCO_TFLITE_TEST)
  return()
endif(NOT ENCO_TFLITE_TEST)

###
### Common function(s)
###
function(get_test_configuration PREFIX)
  set(RECIPE_FILE "${PREFIX}.recipe" PARENT_SCOPE)
  set(TFLITEMODEL_FILE "${PREFIX}.tflite" PARENT_SCOPE)
  set(SOURCE_FILE ${PREFIX}.cpp PARENT_SCOPE)
  set(ASM_FILE ${PREFIX}.embed.S PARENT_SCOPE)
  set(BIN_FILE ${PREFIX}.bin PARENT_SCOPE)
endfunction(get_test_configuration)

###
### Prepare test(s)
###
if(NOT TARGET nnkit_tflite_backend)
  return()
endif(NOT TARGET nnkit_tflite_backend)

if(NOT TARGET tflchef-file)
  return()
endif(NOT TARGET tflchef-file)

if(NOT TARGET enco_tflite_frontend)
  return()
endif(NOT TARGET enco_tflite_frontend)

file(GLOB MODELS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/test.recipe")

foreach(MODEL IN ITEMS ${MODELS})
  get_filename_component(PREFIX ${MODEL} DIRECTORY)
  get_test_configuration(${PREFIX})

  set(MODEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL})

  # Copy recipe
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}
                     COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_FILE}"
                                                      "${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}"
                     DEPENDS "${MODEL_FILE}"
                     COMMENT "Copying ${RECIPE_FILE}")

  # Generate tflitemodel
  add_custom_command(OUTPUT ${TFLITEMODEL_FILE}
                     COMMAND $<TARGET_FILE:tflchef-file> ${RECIPE_FILE} ${TFLITEMODEL_FILE}
                     DEPENDS tflchef ${CMAKE_CURRENT_BINARY_DIR}/${RECIPE_FILE}
                     COMMENT "Generating ${TFLITEMODEL_FILE}")

  # Generate C++ code
  add_custom_command(OUTPUT ${SOURCE_FILE} ${ASM_FILE} ${BIN_FILE}
                     COMMAND $<TARGET_FILE:enco-cli>
                         --frontend $<TARGET_FILE:enco_tflite_frontend>
                         --frontend-arg ${TFLITEMODEL_FILE}
                         --backend-arg ${PREFIX}
                     DEPENDS enco-cli enco_caffe_frontend ${TFLITEMODEL_FILE}
                     COMMENT "Generating ${SOURCE_FILE}")
  set_source_files_properties(${ASM_FILE} PROPERTIES GENERATED TRUE LANGUAGE C)

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

message(STATUS "Enable enco(tflite) inference test")

foreach(PREFIX IN ITEMS ${CANDIDATES})
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${PREFIX}/INFERENCE")
    continue()
  endif()

  get_test_configuration(${PREFIX})

  set(BINDER_TARGET enco_tflite_test_${PREFIX}_binder)

  # Compile nnkit binder (from generated C++ code)
  add_library(${BINDER_TARGET} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../binder.cpp ${SOURCE_FILE} ${ASM_FILE})
  target_include_directories(${BINDER_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${BINDER_TARGET} nnkit_intf_backend)
  target_link_libraries(${BINDER_TARGET} ann_api)
  target_link_libraries(${BINDER_TARGET} ann_ref_static)
  set_target_properties(${BINDER_TARGET} PROPERTIES OUTPUT_NAME ${PREFIX})

  list(APPEND TESTS ${PREFIX})
endforeach(PREFIX)

# Run tests
add_test(NAME enco_test_tflite
         COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/runall.sh"
                            $<TARGET_FILE:nnkit-run>
                            $<TARGET_FILE:nnkit_tflite_backend>
                            $<TARGET_FILE:nnkit_randomize_action>
                            $<TARGET_FILE:nnkit_HDF5_export_action>
                            $<TARGET_FILE:nnkit_HDF5_import_action>
                            "${CMAKE_CURRENT_BINARY_DIR}"
                            ${TESTS})
