nncc_find_resource(TensorFlowTests)

if(NOT TARGET nnkit_tf_backend)
  return()
endif(NOT TARGET nnkit_tf_backend)

if(NOT TARGET tfkit)
  return()
endif(NOT TARGET tfkit)

message(STATUS "Build tfts: TRUE")

# NOTE
#
# tfts_check fails on UNIT_Const_001 and UNIT_Shape_000 due to
# the limitation of "nnkit_tf_backend"
#
# "nnkit_tf_backend" supports only TF_FLOAT32 input/output.
#
# TODO Remove nnkit dependency
add_custom_target(tfts_check
  COMMAND
    "${CMAKE_CURRENT_SOURCE_DIR}/check_all.sh"
    "${TensorFlowTests_DIR}"
    $<TARGET_FILE:tfkit>
    $<TARGET_FILE:nnkit-run>
    $<TARGET_FILE:nnkit_tf_backend>
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
