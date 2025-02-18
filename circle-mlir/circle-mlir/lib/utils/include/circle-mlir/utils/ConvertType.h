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

// from tensorflow/compiler/mlir/lite/utils/convert_type.h

#ifndef __CIRCLE_MLIR_UTILS_CONVERT_TYPE_H__
#define __CIRCLE_MLIR_UTILS_CONVERT_TYPE_H__

#include <circle_schema/schema_generated.h>

#include <mlir/IR/BuiltinAttributes.h> // from @llvm-project
#include <mlir/IR/Builders.h>          // from @llvm-project
#include <mlir/IR/Types.h>             // from @llvm-project
#include <mlir/IR/Value.h>             // from @llvm-project

namespace circle
{

// Convert the MLIR type to the corresponding Circle tensor.
circle::TensorType ConvertTypeToTensorType(mlir::Type type);

// Convert the scalar type of a Circle tensor to the corresponding MLIR type.
mlir::Type ConvertElementType(circle::TensorType type, mlir::Builder builder);

} // namespace circle

#endif // __CIRCLE_MLIR_UTILS_CONVERT_TYPE_H__
