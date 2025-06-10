/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// from tensorflow/compiler/mlir/lite/flatbuffer_operator.h

#ifndef __CIRCLE_MLIR_UTILS_FLATBUFFER_OPERATOR_H__
#define __CIRCLE_MLIR_UTILS_FLATBUFFER_OPERATOR_H__

#include <circle_schema/schema_generated.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AssumeBundleQueries.h> // for llvm::MinMax
#include <mlir/IR/Attributes.h>                // from @llvm-project
#include <mlir/IR/Builders.h>                  // from @llvm-project
#include <mlir/IR/Operation.h>                 // from @llvm-project

#include <vector>
#include <stdint.h>

namespace mlir
{

// Returns the builtin op code for the given MLIR operation on success; emits
// error and returns llvm::None on failure.
std::optional<circle::BuiltinOperator> GetBuiltinOpCode(Operation *mlir_op);

// Packs the given MLIR operation into a TFLite FlatBuffer operator object.
// Returns the FlatBuffer offset for the operator on success; emits error and
// returns llvm::None on failure.
std::optional<flatbuffers::Offset<circle::Operator>>
CreateFlatBufferOperator(Operation *mlir_op, uint32_t opcode_index,
                         const std::vector<int32_t> &operands, const std::vector<int32_t> &results,
                         const std::vector<int32_t> &intermediates,
                         flatbuffers::FlatBufferBuilder *fbb);

// Populates the array of mlir::NamedAttributes corresponding to the given
// circle::FlatbufferOptionsUnion.
// We use an out parameter per LLVM convention
void BuiltinOptionsToAttributes(circle::BuiltinOptionsUnion op_union, mlir::Builder builder,
                                // NOLINTNEXTLINE
                                llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);

// While the last several tensors could be optional tensors for an tfl op, the
// number of input operands could vary. This function gets the min/max number of
// operands from tflite op name.
llvm::MinMax OperandNumbersMinMax(llvm::StringRef op_name);

} // namespace mlir

#endif // __CIRCLE_MLIR_UTILS_FLATBUFFER_OPERATOR_H__
