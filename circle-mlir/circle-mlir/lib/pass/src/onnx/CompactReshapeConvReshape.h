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

#ifndef __CIRCLE_MLIR_PASS_ONNX_COMPACT_RESHAPE_CONV_RESHAPE_H__
#define __CIRCLE_MLIR_PASS_ONNX_COMPACT_RESHAPE_CONV_RESHAPE_H__

#include "ConvertHelper.h"

namespace mlir
{
namespace Circle
{

namespace
{

// check if values are rank 3 1N1
int64_t check_R3_1N1(std::vector<int64_t> &values)
{
  if (values.size() == 3)
  {
    if (values[0] == 1 && values[2] == 1)
      return values[1];
  }
  return 0;
}

// check if values are rank 4 1N11
int64_t check_R4_1N11(std::vector<int64_t> &values)
{
  if (values.size() == 4)
  {
    if (values[0] == 1 && values[2] == 1 && values[3] == 1)
      return values[1];
  }
  return 0;
}

mlir::ArrayAttr duplicate(mlir::PatternRewriter &rewriter, mlir::ArrayAttr input)
{
  mlir::SmallVector<int64_t> temp_v;
  size_t size = input.size();
  for (size_t i = 0; i < size; i++)
  {
    auto val = mlir::dyn_cast<IntegerAttr>(input[i]).getInt();
    temp_v.push_back(val);
    temp_v.push_back(val);
  }
  return rewriter.getI64ArrayAttr(temp_v);
}

} // namespace

// Find sequence with I/O shape of 1N11 -> 1N1 -> 1M1 -> 1M11
//    (1N11)- ONNXReshape -(1N1)- ONNXConv -(1M1)- ONNXReshape -(1M11)
// Relace with with I/O shape of 1N11 -> 1M11
//    (1N11)- ONNXConv -(1M11)
// NOTE
//     ShuffleFaceNet end part has this sequence.
//     onnx-tf does like this.
//     don't know why ShuffleFaceNet creates with torch.nn.Conv1d();
struct CompactReshapeConvReshape : public OpRewritePattern<mlir::ONNXReshapeOp>
{
  using OpRewritePattern<mlir::ONNXReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReshapeOp reshape2_op,
                                      mlir::PatternRewriter &rewriter) const override
  {
    // check Conv-Reshape sequence with shape
    mlir::Operation *is_conv = reshape2_op.getOperand(0).getDefiningOp();
    mlir::Operation *is_const_r2 = reshape2_op.getOperand(1).getDefiningOp();
    bool bis_conv = mlir::isa_and_nonnull<mlir::ONNXConvOp>(is_conv);
    bool bis_const = mlir::isa_and_nonnull<mlir::ONNXConstantOp>(is_const_r2);
    if (!bis_conv || !bis_const)
      return mlir::failure();

    // check if 'shape' value is 1N11
    mlir::Value const_r2_op = cast<mlir::ONNXConstantOp>(is_const_r2);
    std::vector<int64_t> shape_values_r2;
    if (!ExtractConstantValues(const_r2_op, shape_values_r2))
      return mlir::failure();
    if (check_R4_1N11(shape_values_r2) == 0)
      return mlir::failure();

    auto conv_op = cast<mlir::ONNXConvOp>(is_conv);

    // check Reshape-Conv sequence
    mlir::Operation *is_reshape = conv_op.getOperand(0).getDefiningOp();
    bool bis_reshape = mlir::isa_and_nonnull<mlir::ONNXReshapeOp>(is_reshape);
    if (!bis_reshape)
      return mlir::failure();

    // check Reshape shape is constant and 1N1
    auto reshape1_op = cast<mlir::ONNXReshapeOp>(is_reshape);
    mlir::Operation *is_const_r1 = reshape1_op.getOperand(1).getDefiningOp();
    bis_const = mlir::isa_and_nonnull<mlir::ONNXConstantOp>(is_const_r1);
    if (!bis_const)
      return mlir::failure();
    mlir::Value const_r1_op = cast<mlir::ONNXConstantOp>(is_const_r1);
    std::vector<int64_t> shape_values_r1;
    if (!ExtractConstantValues(const_r1_op, shape_values_r1))
      return mlir::failure();
    if (check_R3_1N1(shape_values_r1) == 0)
      return mlir::failure();

    // Get Conv-weight, check shape is OI1, and create new weight with OI11
    mlir::Operation *is_conv_w = conv_op.getOperand(1).getDefiningOp();
    bis_const = mlir::isa_and_nonnull<mlir::ONNXConstantOp>(is_conv_w);
    if (!bis_const)
      return mlir::failure();
    mlir::Value const_w_op = cast<mlir::ONNXConstantOp>(is_conv_w);
    auto w_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(const_w_op.getType());
    if (!w_type.getElementType().isF32())
      return mlir::failure();
    auto w_shape = w_type.getShape();
    if (w_shape.size() != 3)
      return mlir::failure();

    // Now, op sequence and shape values match

    // Create weight with 4D OI11
    std::vector<float> weight_values;
    if (!ExtractConstantValues(const_w_op, weight_values))
      return mlir::failure();

    int64_t w_s_O = w_shape[0];
    int64_t w_s_I = w_shape[1];
    int64_t w_s_2 = w_shape[2];
    auto w_rttype =
      mlir::RankedTensorType::get({w_s_O, w_s_I, w_s_2, w_s_2}, rewriter.getF32Type());
    mlir::Location opLoc = const_w_op.getLoc();
    mlir::Attribute empty_sparse;
    mlir::Attribute attr_value =
      mlir::DenseElementsAttr::get(w_rttype, llvm::ArrayRef(weight_values));
    mlir::Value new_kernel = rewriter.create<mlir::ONNXConstantOp>(opLoc, empty_sparse, attr_value);

    // Get input of first Reshape to be used for input of New Conv
    mlir::Value bias = conv_op.getOperand(2);
    mlir::Value input_r1_op = reshape1_op.getOperand(0);

    // some attributes needs suplicate in size to match 1D -> 2D
    mlir::StringAttr auto_pad = conv_op.getAutoPadAttr();
    mlir::ArrayAttr dilations = duplicate(rewriter, conv_op.getDilationsAttr());
    mlir::IntegerAttr group = conv_op.getGroupAttr();
    mlir::ArrayAttr kernel_shape = duplicate(rewriter, conv_op.getKernelShapeAttr());
    mlir::ArrayAttr pads = duplicate(rewriter, conv_op.getPadsAttr());
    mlir::ArrayAttr strides = duplicate(rewriter, conv_op.getStridesAttr());

    rewriter.replaceOpWithNewOp<mlir::ONNXConvOp>(reshape2_op, reshape2_op.getType(), input_r1_op,
                                                  new_kernel, bias, auto_pad, dilations, group,
                                                  kernel_shape, pads, strides);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_ONNX_COMPACT_RESHAPE_CONV_RESHAPE_H__
