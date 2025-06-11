/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_RESIZE_ONNX_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_RESIZE_ONNX_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"
#include "circle-mlir/dialect/NameUtils.h"

#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ResizeOnnxOp: temporary Op for conversion
//===----------------------------------------------------------------------===//

namespace
{

// NOTE this rewrite will convert temporary ResizeOnnxOp to ResizeNearestNeighborOp
// or ResizeBilinearOp depending on the mode after input H, W are determined with
// shape inference
struct ConvertResizeOnnxOpToResizeOp : public OpRewritePattern<ResizeOnnxOp>
{
  using OpRewritePattern<ResizeOnnxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ResizeOnnxOp op, PatternRewriter &rewriter) const override
  {
    ResizeOnnxOpAdaptor adaptor = ResizeOnnxOpAdaptor(op);

    auto input = adaptor.getInput();
    auto scales = adaptor.getScales();
    llvm::StringRef mode = adaptor.getMode();
    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());

    auto inshape = intype.getShape();
    if (inshape.size() != 4)
      return failure();
    // permit dynamic shape like [?,H,W,?]
    // NOTE input is in NHWC format
    if (ShapedType::isDynamic(inshape[1]) || ShapedType::isDynamic(inshape[2]))
      return failure();

    DenseElementsAttr scales_const;
    if (!matchPattern(scales, m_Constant(&scales_const)))
    {
      // NOTE this shouldn't happen. break in debug mode to see such case.
      assert(false);
      return failure();
    }

    std::vector<float> scales_float;
    for (const APFloat &scales_value : scales_const.getValues<APFloat>())
    {
      // in NCHW format
      float f = scales_value.convertToFloat();
      scales_float.push_back(f);
    }
    assert(scales_float.size() == 4);

    mlir::Location opLoc = op->getLoc();
    auto op_name = mlir::GetNameFromLoc(opLoc);

    // NOTE input is in NHWC format
    int32_t h = static_cast<int32_t>(inshape[1] * scales_float[2]);
    int32_t w = static_cast<int32_t>(inshape[2] * scales_float[3]);
    assert(h > 0 && w > 0);

    std::vector<int32_t> sizeValue = {h, w};
    auto constLoc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/size"));
    auto ptype = RankedTensorType::get({2}, rewriter.getI32Type());
    auto sizeattr = DenseIntElementsAttr::get(ptype, sizeValue);
    mlir::Value size = rewriter.create<ConstOp>(constLoc, sizeattr);

    mlir::BoolAttr align_corners;
    mlir::BoolAttr half_pixel_centers;
    align_corners = rewriter.getBoolAttr(false);
    half_pixel_centers = rewriter.getBoolAttr(false);

    if (mode == "nearest")
    {
      rewriter.replaceOpWithNewOp<ResizeNearestNeighborOp>(op, op.getType(), op.getInput(), size,
                                                           align_corners, half_pixel_centers);
    }
    else if (mode == "linear")
    {
      rewriter.replaceOpWithNewOp<ResizeBilinearOp>(op, op.getType(), op.getInput(), size,
                                                    align_corners, half_pixel_centers);
    }

    return mlir::success();
  }
};

} // namespace

void ResizeOnnxOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<ConvertResizeOnnxOpToResizeOp>(context);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_RESIZE_ONNX_OP_H__
