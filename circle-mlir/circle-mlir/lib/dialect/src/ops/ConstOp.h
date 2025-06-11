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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_CONST_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_CONST_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

namespace
{

struct FoldPseudoConstOp : public OpRewritePattern<ConstOp>
{
  using OpRewritePattern<ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOp const_op, PatternRewriter &rewriter) const override
  {
    if (NoValueOp::isBuildableWith(const_op.getValue(), const_op.getType()))
    {
      rewriter.replaceOpWithNewOp<NoValueOp>(const_op, rewriter.getNoneType(),
                                             mlir::cast<UnitAttr>(const_op.getValue()));
      return success();
    }
    return failure();
  }
};

} // namespace

void ConstOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<FoldPseudoConstOp>(context);
}

OpFoldResult ConstOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "constant has no operands");
  // Return the held attribute value.
  return getValue();
}

bool ConstOp::isCompatibleReturnTypes(TypeRange l, TypeRange r)
{
  // Allow the type inferred to not match exactly the inferred type as the
  // inferred type is from the element attribute's type while the op may have
  // gotten constructed from TF const op or be in a partial state of shape
  // refinement, so allow it to only be compatible. The op will be refined
  // during shape inference and casts inserted as needed to satisfy type
  // constraints of consumers.
  return succeeded(verifyCompatibleShapes(l, r));
}

bool ConstOp::isBuildableWith(Attribute value, Type type)
{
  // The value's type must be the same as the provided type.
  auto typedAttr = mlir::dyn_cast<TypedAttr>(value);
  if (!typedAttr || typedAttr.getType() != type)
    return false;
  // Integer values must be signless.
  if (mlir::isa<IntegerType>(type) && !mlir::cast<IntegerType>(type).isSignless())
    return false;
  // Integer, float, and element attributes are buildable.
  return mlir::isa<IntegerAttr, FloatAttr, ElementsAttr>(value);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_CONST_OP_H__
