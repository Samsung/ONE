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

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc

#ifndef __CIRCLE_MLIR_DIALECT_OPS_GATHER_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_GATHER_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify()
{
  // TODO enable mlir::TF::StringType if necessary
  // GatherOp op = *this;
  // ShapedType params_type = op.getParams().getType().cast<ShapedType>();
  // // TFLite gather kernel supports 1D string input only.
  // if (params_type.getElementType().isa<mlir::TF::StringType>()) {
  //   if (params_type.hasRank() && params_type.getRank() != 1) {
  //     return op.emitOpError(
  //                "expect 1d input when the given type is string, got ")
  //            << params_type;
  //   }
  // }
  return mlir::success();
}

OpFoldResult GatherOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  // TODO support more types
  if (!IsI64ShapedType(getType()))
    return nullptr;

  auto params_type = mlir::cast<ShapedType>(getParams().getType());
  if (!params_type.hasStaticShape())
    return nullptr;

  auto indices_type = mlir::cast<ShapedType>(getIndices().getType());
  if (!indices_type.hasStaticShape())
    return nullptr;

  // NOTE for simplicity, support 1D params_type, indices_type for now
  // TODO support more ranks
  ArrayRef<int64_t> shape = params_type.getShape();
  if (shape.size() != 1)
    return nullptr;

  GatherOp op = *this;
  auto in_params = op.getParams();
  auto in_indices = op.getIndices();

  std::vector<int64_t> params_v;
  std::vector<int64_t> indices_v;
  if (!getAsConstant(in_params, params_v))
    return nullptr;
  if (!getAsConstant(in_indices, indices_v))
    return nullptr;

  // TODO revise this to support 2D or higher ranks.
  SmallVector<int64_t, 4> result;
  for (int64_t idx : indices_v)
  {
    assert(0 <= idx && idx < params_v.size());
    result.push_back(params_v[idx]);
  }
  auto result_type = mlir::cast<ShapedType>(getType());
  return DenseElementsAttr::get<int64_t>(result_type, result);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_GATHER_OP_H__
