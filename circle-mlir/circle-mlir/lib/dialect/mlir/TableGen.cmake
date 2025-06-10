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

set(LLVM_TARGET_DEFINITIONS mlir/CircleOpEnums.td)
mlir_tablegen(mlir/CircleOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(mlir/CircleOpsEnums.cc.inc -gen-enum-defs)
mlir_tablegen(mlir/CircleOpsAttrdefs.h.inc -gen-attrdef-decls)
mlir_tablegen(mlir/CircleOpsAttrdefs.cc.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS mlir/CircleOps.td)
mlir_tablegen(mlir/CircleOps.h.inc -gen-op-decls)
mlir_tablegen(mlir/CircleOps.cc.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS mlir/CircleOps.td)
cir_convertergen(mlir/OperatorConverters.inc --gen-operator-converters)
cir_convertergen(mlir/RuntimeVerifiers.inc --gen-runtime-verifiers)

set(LLVM_TARGET_DEFINITIONS mlir/CircleRewrite.td)
mlir_tablegen(mlir/CircleRewrite.cc.inc -gen-rewriters)

add_public_tablegen_target(circle_mlir_gen_inc)
