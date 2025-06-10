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

#ifndef __CIRCLE_MLIR_IMPORT_CIRCLE_OPERATOR_H__
#define __CIRCLE_MLIR_IMPORT_CIRCLE_OPERATOR_H__

#include <circle_schema/schema_generated.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/AssumeBundleQueries.h> // MinMax
#include <mlir/IR/Builders.h>

#include <string>
#include <vector>

namespace circle
{

// from tensorflow/lite/schema/schema_utils.h

BuiltinOperator GetBuiltinCode(const OperatorCode *op_code);
BuiltinOperator GetBuiltinCode(const OperatorCodeT *op_code);

} // namespace circle

namespace mlir
{

bool IsStablehloOp(const circle::OperatorCodeT &op_code);

// Returns the MLIR op name for the flatbuffer operator corresponding to `op_code`.
std::string GetMlirOpNameFromOpCode(const circle::OperatorCodeT &op_code);

// Populates the array of mlir::NamedAttributes corresponding to the given
// circle::FlatbufferOptionsUnion.
// We use an out parameter per LLVM convention
void BuiltinOptionsToAttributes(circle::BuiltinOptionsUnion op_union, mlir::Builder builder,
                                // NOLINTNEXTLINE
                                llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);

// Populates the `custom_code` and `custom_options` to attributes.
// `custom_code` is used to identify CustomOp.
// `custom_options` are opaque attribute used to store infomations for this
// custom op.
bool CustomOptionsToAttributes(const std::string &custom_code,
                               const std::vector<uint8_t> &custom_options, mlir::Builder builder,
                               // NOLINTNEXTLINE
                               Location loc,
                               llvm::SmallVectorImpl<mlir::NamedAttribute> *attributes);

} // namespace mlir

#endif // __CIRCLE_MLIR_IMPORT_CIRCLE_OPERATOR_H__
