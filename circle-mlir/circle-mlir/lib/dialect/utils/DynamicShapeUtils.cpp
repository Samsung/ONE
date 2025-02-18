/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.cc

#include "DynamicShapeUtils.h"

#include <llvm/ADT/SmallVector.h>

namespace mlir
{
namespace Circle
{

namespace
{

llvm::SmallVector<int64_t> ConvertShapeToMlir(llvm::ArrayRef<int64_t> shapes)
{
  return llvm::to_vector(llvm::map_range(shapes, [](int64_t shape) {
    return shape == kDynamicSize ? mlir::ShapedType::kDynamic : shape;
  }));
}

} // namespace

mlir::RankedTensorType GetTypeFromTensorShape(llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                                              mlir::Attribute encoding)
{
  return mlir::RankedTensorType::get(ConvertShapeToMlir(shape), elementType, encoding);
}

} // namespace Circle
} // namespace mlir
