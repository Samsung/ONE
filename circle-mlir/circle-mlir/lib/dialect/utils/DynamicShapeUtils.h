/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h

#ifndef __CIRCLE_MLIR_DIALECT_UTILS_DYNAMIC_SHAPE_UTILS_H__
#define __CIRCLE_MLIR_DIALECT_UTILS_DYNAMIC_SHAPE_UTILS_H__

#include <mlir/IR/BuiltinTypes.h> // from @llvm-project

namespace mlir
{
namespace Circle
{

static constexpr int64_t kDynamicSize = -1;

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_UTILS_DYNAMIC_SHAPE_UTILS_H__
