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

// from tensorflow/compiler/mlir/lite/flatbuffer_import.h

#ifndef __CIRCLE_MLIR_IMPORT_CIRCLE_IMPORT_H__
#define __CIRCLE_MLIR_IMPORT_CIRCLE_IMPORT_H__

#include <string>
#include <vector>

#include <absl/strings/string_view.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>

namespace mlir
{
namespace Circle
{

mlir::OwningOpRef<mlir::ModuleOp>
FlatBufferToMlir(absl::string_view buffer, mlir::MLIRContext *context, mlir::Location base_loc,
                 bool use_external_constant = false,
                 const std::vector<std::string> &ordered_input_arrays = {},
                 const std::vector<std::string> &ordered_output_arrays = {},
                 bool experimental_prune_unreachable_nodes_unconditionally = false);

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_IMPORT_CIRCLE_IMPORT_H__
