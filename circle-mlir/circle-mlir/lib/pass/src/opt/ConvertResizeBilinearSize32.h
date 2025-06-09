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

#ifndef __CIRCLE_MLIR_PASS_OPT_CONVERT_RESIZE_BILINEAR_SIZE_32_H__
#define __CIRCLE_MLIR_PASS_OPT_CONVERT_RESIZE_BILINEAR_SIZE_32_H__

#include "ConvertHelper.h"

#include <cassert>

namespace mlir
{
namespace Circle
{

// Find INT64 Const sizes of ResizeBilinearOp
//    Const(shape/INT64)-ResizeBilinearOp
// Relace Const(shape/INT64) with Const(shape/INT32)
//    Const(shape/INT32)-ResizeBilinearOp
//    ResizeBilinearOp second input size requires 1D, 2 elements for new H,W
//    if input has 4 elements, we assume it is NHWC and only achieve H and W.
struct ConvertResizeBilinearSize32 : public OpRewritePattern<ResizeBilinearOp>
{
  using OpRewritePattern<ResizeBilinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ResizeBilinearOp resize_op,
                                PatternRewriter &rewriter) const override
  {
    mlir::Operation *is_const = resize_op.getOperand(1).getDefiningOp();
    if (!mlir::isa_and_nonnull<ConstOp>(is_const))
      return mlir::failure();

    auto const_op = cast<ConstOp>(is_const);
    auto const_type = mlir::cast<TensorType>(const_op.getType());
    mlir::Value resize_size = const_op; // ExtractConstantValues requries mlir::Value
    std::vector<int32_t> values;
    if (!ExtractConstantValues(resize_size, values))
      return mlir::failure();

    if (const_type.getElementType().isInteger(32))
    {
      assert(values.size() == 2);
      return mlir::failure();
    }
    assert(const_type.getElementType().isInteger(64));

    if (values.size() == 4)
    {
      // values are in NCHW from ONNX
      int32_t H = values[2];
      int32_t W = values[3];
      values.clear();
      values.push_back(H);
      values.push_back(W);
    }
    assert(values.size() == 2);

    mlir::Location opLoc = const_op->getLoc();
    mlir::Type i32 = rewriter.getI32Type();
    mlir::RankedTensorType si32stype = RankedTensorType::get({2}, i32);
    mlir::Value size32 =
      rewriter.create<ConstOp>(opLoc, DenseIntElementsAttr::get(si32stype, values));

    auto &resize_mutable = resize_op.getSizeMutable();
    resize_mutable.assign(size32);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_CONVERT_RESIZE_BILINEAR_SIZE_32_H__
