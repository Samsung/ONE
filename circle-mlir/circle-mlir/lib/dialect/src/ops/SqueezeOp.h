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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_SQUEEZE_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_SQUEEZE_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

OpFoldResult SqueezeOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  auto input_ty = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  auto result_ty = mlir::dyn_cast<RankedTensorType>(getType());

  if (!input_ty || !result_ty)
    return {};
  if (input_ty == result_ty)
    return getInput();
  return {};
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SQUEEZE_OP_H__
