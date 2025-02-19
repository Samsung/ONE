/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/utils/name_utils.h

#ifndef __CIRCLE_MLIR_DIALECT_NAME_UTILS_H__
#define __CIRCLE_MLIR_DIALECT_NAME_UTILS_H__

#include "mlir/IR/Location.h" // from @llvm-project

#include <string>

namespace mlir
{

// Returns the node name associated with a location.
std::string GetNameFromLoc(Location loc);

} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_NAME_UTILS_H__
