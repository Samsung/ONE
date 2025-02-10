set(FILE_DEPS )

# ConvertUnitModel used in test.lst
set(TEST_MODELS )
macro(ConvertUnitModel MLIR_FNAME)
  # copy to build folder
  set(TEST_MLIR_MODEL_SRC "${CMAKE_SOURCE_DIR}/models/mlir/${MLIR_FNAME}")
  set(TEST_MLIR_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MLIR_FNAME}")
  add_custom_command(
    OUTPUT ${TEST_MLIR_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_MLIR_MODEL_SRC}" "${TEST_MLIR_MODEL_DST}"
    DEPENDS ${TEST_MLIR_MODEL_SRC}
    COMMENT "tools/onnx2circle: prepare mlir/${MLIR_FNAME}"
  )
  list(APPEND TEST_MODELS "${MLIR_FNAME}")
  list(APPEND FILE_DEPS "${TEST_MLIR_MODEL_DST}")
endmacro(ConvertUnitModel)

set(TEST_NEG_MODELS )
macro(ConvertUnitModelNEG MLIR_FNAME)
  # copy to build folder
  set(TEST_MLIR_MODEL_SRC "${CMAKE_SOURCE_DIR}/models/mlir/${MLIR_FNAME}")
  set(TEST_MLIR_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MLIR_FNAME}")
  add_custom_command(
    OUTPUT ${TEST_MLIR_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_MLIR_MODEL_SRC}" "${TEST_MLIR_MODEL_DST}"
    DEPENDS ${TEST_MLIR_MODEL_SRC}
    COMMENT "tools/onnx2circle: prepare mlir/${MLIR_FNAME}"
  )
  list(APPEND TEST_NEG_MODELS "${MLIR_FNAME}")
  list(APPEND FILE_DEPS "${TEST_MLIR_MODEL_DST}")
endmacro(ConvertUnitModelNEG)

set(VALIDATE_SHAPEINF_MODELS)
macro(ValidateShapeInf MLIR_FNAME)
  # copy to build folder
  set(TEST_MLIR_MODEL_SRC "${CMAKE_SOURCE_DIR}/models/mlir/${MLIR_FNAME}")
  set(TEST_MLIR_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MLIR_FNAME}")
  add_custom_command(
    OUTPUT ${TEST_MLIR_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_MLIR_MODEL_SRC}" "${TEST_MLIR_MODEL_DST}"
    DEPENDS ${TEST_MLIR_MODEL_SRC}
    COMMENT "tools/onnx2circle: prepare mlir/${MLIR_FNAME}"
  )
  list(APPEND VALIDATE_SHAPEINF_MODELS "${MLIR_FNAME}")
  list(APPEND FILE_DEPS "${TEST_MLIR_MODEL_DST}")
endmacro(ValidateShapeInf)

# ValidateDynaShapeInf is to test output should have dynamic shape from dynamic shape input
set(VALIDATE_DYNASHAPEINF_MODELS)
macro(ValidateDynaShapeInf MLIR_FNAME)
  # copy to build folder
  set(TEST_MLIR_MODEL_SRC "${CMAKE_SOURCE_DIR}/models/mlir/${MLIR_FNAME}")
  set(TEST_MLIR_MODEL_DST "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MLIR_FNAME}")
  add_custom_command(
    OUTPUT ${TEST_MLIR_MODEL_DST}
    COMMAND ${CMAKE_COMMAND} -E copy "${TEST_MLIR_MODEL_SRC}" "${TEST_MLIR_MODEL_DST}"
    DEPENDS ${TEST_MLIR_MODEL_SRC}
    COMMENT "tools/onnx2circle: prepare mlir/${MLIR_FNAME}"
  )
  list(APPEND VALIDATE_DYNASHAPEINF_MODELS "${MLIR_FNAME}")
  list(APPEND FILE_DEPS "${TEST_MLIR_MODEL_DST}")
endmacro(ValidateDynaShapeInf)

# Read "test.lst"
include("test.lst")

add_custom_target(onnx2circle_deps ALL DEPENDS ${FILE_DEPS})

foreach(MODEL IN ITEMS ${TEST_MODELS})
  set(MLIR_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}")
  set(CIRCLE_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}.circle")
  add_test(
    NAME onnx2circle_test_${MODEL}
    COMMAND "$<TARGET_FILE:onnx2circle>"
      "${MLIR_MODEL_PATH}"
      "${CIRCLE_MODEL_PATH}"
  )
endforeach()

foreach(MODEL IN ITEMS ${TEST_NEG_MODELS})
  set(MLIR_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}")
  set(CIRCLE_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}.circle")
  add_test(
    NAME onnx2circle_neg_test_${MODEL}
    COMMAND "$<TARGET_FILE:onnx2circle>"
      "${MLIR_MODEL_PATH}"
      "${CIRCLE_MODEL_PATH}"
  )
  set_tests_properties(onnx2circle_neg_test_${MODEL} PROPERTIES WILL_FAIL TRUE)
endforeach()

foreach(MODEL IN ITEMS ${VALIDATE_SHAPEINF_MODELS})
  set(MLIR_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}")
  set(CIRCLE_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}.circle")
  add_test(
    NAME onnx2circle_valshapeinf_${MODEL}
    COMMAND "$<TARGET_FILE:onnx2circle>" "--check_shapeinf"
      "${MLIR_MODEL_PATH}"
      "${CIRCLE_MODEL_PATH}"
  )
endforeach()

foreach(MODEL IN ITEMS ${VALIDATE_DYNASHAPEINF_MODELS})
  set(MLIR_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}")
  set(CIRCLE_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/models/mlir/${MODEL}.circle")
  add_test(
    NAME onnx2circle_valdynshapeinf_${MODEL}
    COMMAND "$<TARGET_FILE:onnx2circle>" "--check_dynshapeinf"
      "${MLIR_MODEL_PATH}"
      "${CIRCLE_MODEL_PATH}"
  )
endforeach()
