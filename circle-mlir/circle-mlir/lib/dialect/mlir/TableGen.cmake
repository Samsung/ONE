# NOTE create "mlir" folder as mlir_tablegen fails with using folder name mlir/NAME
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mlir)

include_directories("${LLVM_INST_INC}")

set(LLVM_TARGET_DEFINITIONS mlir/CircleOpInterfaces.td)
mlir_tablegen(mlir/CircleOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(mlir/CircleOpInterface.cc.inc -gen-op-interface-defs)
mlir_tablegen(mlir/CircleOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(mlir/CircleOpsDialect.cc.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS mlir/CircleShapeInferenceInterfaces.td)
mlir_tablegen(mlir/CircleShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(mlir/CircleShapeInferenceOpInterfaces.cc.inc -gen-op-interface-defs)

add_public_tablegen_target(circle_mlir_gen_inc)
