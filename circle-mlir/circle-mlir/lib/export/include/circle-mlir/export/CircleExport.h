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

// from tensorflow/compiler/mlir/lite/flatbuffer_export.h

#ifndef __CIRCLE_MLIR_EXPORT_CIRCLE_EXPORT_H__
#define __CIRCLE_MLIR_EXPORT_CIRCLE_EXPORT_H__

#include <mlir/Dialect/Func/IR/FuncOps.h> // from @llvm-project
#include <mlir/IR/BuiltinOps.h>           // from @llvm-project

#include <string>

namespace mlir
{
namespace Circle
{

// Translates the given MLIR `module` into a FlatBuffer and stores the
// serialized flatbuffer into the string.
// Returns true on successful exporting, false otherwise.
bool MlirToFlatBufferTranslateFunction(mlir::ModuleOp module, std::string *serialized_flatbuffer);

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_EXPORT_CIRCLE_EXPORT_H__
