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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_EXPAND_ONNX_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_EXPAND_ONNX_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ExpandOnnxOp: temporary Op for conversion
//===----------------------------------------------------------------------===//

namespace
{

struct ConvertExpandOnnxOp2MulOp : public OpRewritePattern<ExpandOnnxOp>
{
  using OpRewritePattern<ExpandOnnxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandOnnxOp expand, PatternRewriter &rewriter) const override
  {
    ExpandOnnxOpAdaptor adaptor = ExpandOnnxOpAdaptor(expand);

    mlir::DenseIntElementsAttr shapeAttr;
    if (!matchPattern(expand.getShape(), m_Constant(&shapeAttr)))
      return failure();

    auto input = adaptor.getInput();
    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    auto inshape = intype.getShape();

    mlir::Location opLoc = expand->getLoc();

    std::vector<int64_t> mulShapeVal;
    int64_t numElements = 1;
    for (auto s : shapeAttr)
    {
      mulShapeVal.push_back(s.getSExtValue());
      numElements = numElements * s.getSExtValue();
    }

    // NOTE there can be inshape.size() < shapeAttr.size() when input has single element
    std::vector<int64_t> outShapeVal;
    int32_t inidx = 0;
    int64_t dim = 0;
    for (auto s : shapeAttr)
    {
      if (inidx < inshape.size())
        dim = std::max(s.getSExtValue(), inshape[inidx]);
      else
        dim = s.getSExtValue();
      outShapeVal.push_back(dim);
      inidx++;
    }

    mlir::RankedTensorType outtype =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(expand.getType());

    mlir::Value mulOnes;
    mlir::Type outeletype = outtype.getElementType();
    if (outeletype.isF32())
    {
      llvm::SmallVector<float> cvals;
      for (int64_t c = 0; c < numElements; ++c)
        cvals.push_back(1.0f);
      mlir::Type f32 = rewriter.getF32Type();
      mlir::RankedTensorType ctype = RankedTensorType::get(mulShapeVal, f32);
      auto mulOnes = rewriter.create<ConstOp>(opLoc, mlir::DenseFPElementsAttr::get(ctype, cvals));

      mlir::RankedTensorType otype = RankedTensorType::get(outShapeVal, f32);
      rewriter.replaceOpWithNewOp<MulOp>(expand, otype, expand.getInput(), mulOnes, "NONE");
      return mlir::success();
    }
    else if (outeletype.isSignlessInteger(1))
    {
      // This is for boolean type
      // TODO Reduce duplicated code
      llvm::SmallVector<bool> cvals;
      for (int64_t c = 0; c < numElements; ++c)
        cvals.push_back(true);
      mlir::Type i1 = rewriter.getI1Type();
      mlir::RankedTensorType ctype = RankedTensorType::get(mulShapeVal, i1);
      auto mulOnes = rewriter.create<ConstOp>(opLoc, mlir::DenseIntElementsAttr::get(ctype, cvals));

      mlir::RankedTensorType otype = RankedTensorType::get(outShapeVal, i1);
      rewriter.replaceOpWithNewOp<MulOp>(expand, otype, expand.getInput(), mulOnes, "NONE");
      return mlir::success();
    }
    else if (outeletype.isSignlessInteger(64))
    {
      // TODO Reduce duplicated code
      llvm::SmallVector<int64_t> cvals;
      for (int64_t c = 0; c < numElements; ++c)
        cvals.push_back(1);
      mlir::Type i64 = rewriter.getI64Type();
      mlir::RankedTensorType ctype = RankedTensorType::get(mulShapeVal, i64);
      auto mulOnes = rewriter.create<ConstOp>(opLoc, mlir::DenseIntElementsAttr::get(ctype, cvals));

      mlir::RankedTensorType otype = RankedTensorType::get(outShapeVal, i64);
      rewriter.replaceOpWithNewOp<MulOp>(expand, otype, expand.getInput(), mulOnes, "NONE");
      return mlir::success();
    }

    // TODO support more types
    return mlir::failure();
  }
};

} // namespace

void ExpandOnnxOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<ConvertExpandOnnxOp2MulOp>(context);
}

LogicalResult ExpandOnnxOp::verify()
{
  ExpandOnnxOpAdaptor operandAdaptor = ExpandOnnxOpAdaptor(*this);
  // Get operands.
  auto shape = operandAdaptor.getShape();
  // Check input.
  auto shapeType = mlir::dyn_cast_or_null<ShapedType>(shape.getType());
  if (shapeType && shapeType.hasRank())
  {
    if (shapeType.getRank() != 1)
      return emitOpError("Expand shape rank should be 1");
  }
  return success();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_EXPAND_ONNX_OP_H__
