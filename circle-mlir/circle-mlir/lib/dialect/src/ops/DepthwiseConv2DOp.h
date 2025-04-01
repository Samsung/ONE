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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_DEPTHWISE_CONV2D_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_DEPTHWISE_CONV2D_OP_H__

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// DepthwiseConv2DOp
//===----------------------------------------------------------------------===//

void DepthwiseConv2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context)
{
  // TODO(b/180121750): Enable the pattern after the integration tests are
  // fixed.
  // results.add<RemoveOptionalZeroBias<DepthwiseConv2DOp>>(context);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_DEPTHWISE_CONV2D_OP_H__
