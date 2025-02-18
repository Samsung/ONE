# NOTE create "mlir" folder as mlir_tablegen fails with using folder name mlir/NAME
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mlir)

include_directories("${LLVM_INST_INC}")
