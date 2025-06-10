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

// from tensorflow/compiler/mlir/lite/utils/size_utils.cc

#include "circle-mlir/utils/SizeUtils.h"

#include <cstdint>

#include <mlir/IR/BuiltinTypes.h>

namespace mlir
{
namespace Circle
{

int32_t ConvertToCircleSize(int64_t size)
{
  return mlir::ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size);
}

} // namespace Circle
} // namespace mlir
