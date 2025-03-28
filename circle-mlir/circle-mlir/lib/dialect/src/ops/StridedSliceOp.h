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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_STRIDED_SLICE_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_STRIDED_SLICE_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// StridedSliceOp
//===----------------------------------------------------------------------===//

namespace
{

struct RewriteZeroEnds : public OpRewritePattern<StridedSliceOp>
{
  using OpRewritePattern<StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StridedSliceOp op, PatternRewriter &rewriter) const override
  {
    // Currently only support all masks being 0.
    if (op.getBeginMask() != 0 || op.getEndMask() != 0 || op.getEllipsisMask() != 0 ||
        op.getNewAxisMask() != 0 || op.getShrinkAxisMask() != 0)
      return failure();

    auto input_type = op.getInput().getType().dyn_cast<RankedTensorType>();
    if (!input_type)
      return failure();
    if (!input_type.hasStaticShape())
      return failure();

    // Begin must be within the input shape dimension.
    mlir::DenseIntElementsAttr begin_dense_elem_attr;
    if (!matchPattern(op.getBegin(), m_Constant(&begin_dense_elem_attr)))
      return failure();
    for (const auto &[i, begin_ele] : llvm::enumerate(begin_dense_elem_attr))
    {
      int64_t val = begin_ele.getSExtValue();
      if (val < 0 || val > input_type.getDimSize(i))
        return failure();
    }

    // End must be within the input shape dimension.
    mlir::DenseIntElementsAttr end_dense_elem_attr;
    if (!matchPattern(op.getEnd(), m_Constant(&end_dense_elem_attr)))
      return failure();
    for (const auto &[i, end_ele] : llvm::enumerate(end_dense_elem_attr))
    {
      int64_t val = end_ele.getSExtValue();
      if (val < 0 || val > input_type.getDimSize(i))
        return failure();
    }

    bool is_changed = false;
    SmallVector<int32_t, 4> new_end;
    for (const auto &[i, end_val] : llvm::enumerate(end_dense_elem_attr))
    {
      if (end_val == 0)
      {
        is_changed = true;
        new_end.push_back(static_cast<int32_t>(input_type.getDimSize(i)));
      }
      else
      {
        new_end.push_back(static_cast<int32_t>(end_val.getSExtValue()));
      }
    }

    if (!is_changed)
      return failure();

    // Replace the zero end to the newly calculated end
    SmallVector<int64_t, 4> shape_type_dim;
    shape_type_dim.push_back(new_end.size());
    auto new_end_type = RankedTensorType::get(shape_type_dim, rewriter.getIntegerType(32));
    auto new_end_dense_attr = mlir::DenseIntElementsAttr::get(new_end_type, new_end);

    auto new_end_op = rewriter.create<ConstOp>(op.getLoc(), new_end_type, new_end_dense_attr);

    // Replace the StridedSliceOp operation itself
    rewriter.replaceOpWithNewOp<StridedSliceOp>(op, op.getType(), op.getInput(), op.getBegin(),
                                                new_end_op, op.getStrides(), op.getBeginMask(),
                                                op.getEndMask(), op.getEllipsisMask(),
                                                op.getNewAxisMask(), op.getShrinkAxisMask());

    return success();
  }
};

} // namespace

void StridedSliceOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<RewriteZeroEnds>(context);
}

LogicalResult StridedSliceOp::verify()
{
  StridedSliceOp op = *this;
  auto ranked_input_type = op.getInput().getType().dyn_cast<RankedTensorType>();

  // If input is unranked, there is nothing else to be verified.
  if (!ranked_input_type)
    return success();
  int num_input_dims = ranked_input_type.getRank();

  if (auto begin_type = op.getBegin().getType().dyn_cast<RankedTensorType>())
  {
    if (begin_type.getRank() != 1)
      return failure();
    if (begin_type.getDimSize(0) > num_input_dims)
      return failure();
  }

  if (auto end_type = op.getEnd().getType().dyn_cast<RankedTensorType>())
  {
    if (end_type.getRank() != 1)
      return failure();
    if (end_type.getDimSize(0) > num_input_dims)
      return failure();
  }

  if (auto strides_type = op.getStrides().getType().dyn_cast<RankedTensorType>())
  {
    if (strides_type.getRank() != 1)
      return failure();
    if (strides_type.getDimSize(0) > num_input_dims)
      return failure();
  }

  // The kernel will reshape the input tensor with new axis, it only supports
  // this reshaped tensor up to 5D.
  uint32_t ellipsis_mask = op.getEllipsisMask();
  uint32_t new_axis_mask = op.getNewAxisMask();
  int num_added_axis = 0;
  for (int i = 0; i < 8; ++i)
  {
    if (!((1 << i) & ellipsis_mask) && ((1 << i) & new_axis_mask))
    {
      num_added_axis++;
    }
  }
  if (num_input_dims + num_added_axis > 5)
    return failure();
  return success();
}

OpFoldResult StridedSliceOp::foldOneDimension()
{
  auto input_type = getInput().getType().dyn_cast_or_null<RankedTensorType>();

  int64_t stride = 1;
  DenseIntElementsAttr stride_dense_elem_attr;
  if (matchPattern(getStrides(), m_Constant(&stride_dense_elem_attr)))
    stride = (stride_dense_elem_attr.getValues<APInt>()[0]).getSExtValue();

  // Begin must be within the input shape dimension.
  DenseIntElementsAttr begin_dense_elem_attr;
  if (!matchPattern(getBegin(), m_Constant(&begin_dense_elem_attr)))
    return {};
  for (const auto &[i, begin_ele] : llvm::enumerate(begin_dense_elem_attr))
  {
    int64_t val = begin_ele.getSExtValue();
    if (stride > 0 && (val < -input_type.getDimSize(i) || val > input_type.getDimSize(i)))
      return {};
    if (stride < 0 && (val < -input_type.getDimSize(i) - 1 || val > input_type.getDimSize(i) + 1))
      return {};
  }

  // End must be within the input shape dimension.
  DenseIntElementsAttr end_dense_elem_attr;
  if (!matchPattern(getEnd(), m_Constant(&end_dense_elem_attr)))
    return {};
  for (const auto &[i, end_ele] : llvm::enumerate(end_dense_elem_attr))
  {
    int64_t val = end_ele.getSExtValue();
    if (stride > 0 && (val < -input_type.getDimSize(i) || val > input_type.getDimSize(i)))
      return {};
    if (stride < 0 && (val < -input_type.getDimSize(i) - 1 || val > input_type.getDimSize(i) + 1))
      return {};
  }

  // Compute the sliced output based on begin and end indices.
  std::vector<int64_t> input_values;
  auto input = getInput();
  if (!getAsConstant(input, input_values))
    return {};

  auto convertNegativeIndex = [&](int64_t index, int64_t size) -> int64_t {
    return index < 0 ? size + index : index;
  };

  int64_t input_size = input_values.size();
  int64_t begin_idx = (begin_dense_elem_attr.getValues<APInt>()[0]).getSExtValue();
  begin_idx = convertNegativeIndex(begin_idx, input_size);
  int64_t end_idx = (end_dense_elem_attr.getValues<APInt>()[0]).getSExtValue();
  end_idx = convertNegativeIndex(end_idx, input_size);

  std::vector<int64_t> sliced_values;
  if (stride > 0)
  {
    sliced_values =
      std::vector<int64_t>(input_values.begin() + begin_idx, input_values.begin() + end_idx);
  }
  else if (stride < 0)
  {
    sliced_values = std::vector<int64_t>(input_values.rbegin() + (input_size - begin_idx - 1),
                                         input_values.rbegin() + (input_size - end_idx - 1));
  }

  auto elementType = input_type.getElementType();
  auto slicedType =
    RankedTensorType::get({static_cast<int64_t>(sliced_values.size())}, elementType);

  return DenseElementsAttr::get<int64_t>(slicedType, sliced_values);
}

OpFoldResult StridedSliceOp::foldReverseInput()
{
  // TODO Currently we only support rank is 2,
  // also need to cover when this rank is greater than 2.
  auto input_type = getInput().getType().dyn_cast_or_null<RankedTensorType>();
  if (input_type.getRank() != 2)
    return {};

  mlir::DenseElementsAttr input_dense_elem_attr;
  if (!matchPattern(getInput(), m_Constant(&input_dense_elem_attr)))
    return {};

  auto input_values = input_dense_elem_attr.getValues<APInt>();

  SmallVector<int64_t, 4> shape(input_type.getShape().begin(), input_type.getShape().end());
  std::vector<int64_t> output_values;

  mlir::DenseIntElementsAttr strides_dense_elem_attr;
  if (!matchPattern(getStrides(), m_Constant(&strides_dense_elem_attr)))
    return {};

  for (int64_t i = 0; i < shape[0]; ++i)
  {
    for (int64_t j = 0; j < shape[1]; ++j)
    {
      int64_t idx0 = i;
      int64_t idx1 = j;

      if (strides_dense_elem_attr.getValues<APInt>()[0].getSExtValue() == -1)
        idx0 = shape[0] - 1 - i;

      if (strides_dense_elem_attr.getValues<APInt>()[1].getSExtValue() == -1)
        idx1 = shape[1] - 1 - j;

      int64_t index = idx0 * shape[1] + idx1;
      output_values.push_back(input_values[index].getSExtValue());
    }
  }

  auto elementType = input_type.getElementType();
  auto output_type = RankedTensorType::get(shape, elementType);
  SmallVector<APInt, 4> apint_output_values;

  for (auto val : output_values)
    apint_output_values.push_back(APInt(elementType.getIntOrFloatBitWidth(), val));

  return DenseElementsAttr::get(output_type, apint_output_values);
}

OpFoldResult StridedSliceOp::fold(FoldAdaptor adaptor)
{
  // Currently only support all masks being 0.
  if (getBeginMask() != 0 || getEndMask() != 0 || getEllipsisMask() != 0 || getNewAxisMask() != 0 ||
      getShrinkAxisMask() != 0)
    return {};

  auto input_type = getInput().getType().dyn_cast_or_null<RankedTensorType>();
  if (!input_type || !input_type.hasStaticShape())
    return {};

  // Strides has to be all 1s or -1s.
  mlir::DenseIntElementsAttr strides_dense_elem_attr;
  if (!matchPattern(getStrides(), m_Constant(&strides_dense_elem_attr)))
    return {};

  for (auto stride_ele : strides_dense_elem_attr)
  {
    if (stride_ele.getSExtValue() != 1 && stride_ele.getSExtValue() != -1)
      return {};
  }

  // Fold if the input tensor is constant and 1-dimensional.
  mlir::DenseElementsAttr input_dense_elem_attr;
  if (matchPattern(getInput(), m_Constant(&input_dense_elem_attr)) && input_type.getRank() == 1)
    return foldOneDimension();

  SmallVector<int64_t, 4> begin_values, end_values;

  mlir::DenseIntElementsAttr begin_dense_elem_attr;
  if (!matchPattern(getBegin(), m_Constant(&begin_dense_elem_attr)))
    return {};

  for (auto begin_ele : llvm::enumerate(begin_dense_elem_attr.getValues<APInt>()))
    begin_values.push_back(begin_ele.value().getSExtValue());

  mlir::DenseIntElementsAttr end_dense_elem_attr;
  if (!matchPattern(getEnd(), m_Constant(&end_dense_elem_attr)))
    return {};

  for (auto end_ele : llvm::enumerate(end_dense_elem_attr.getValues<APInt>()))
    end_values.push_back(end_ele.value().getSExtValue());

  // Check if any stride is -1 and input and output tensor sizes are equal.
  bool needs_reverse = false;
  SmallVector<int64_t, 4> shape(input_type.getShape().begin(), input_type.getShape().end());

  for (auto stride_ele : llvm::enumerate(strides_dense_elem_attr))
  {
    int64_t stride_val = stride_ele.value().getSExtValue();
    int64_t dim = stride_ele.index();

    // This will cover only the case that it slices the whole input tensor,
    // just reversing by specified axis.
    if ((stride_val == 1 && (begin_values[dim] != 0 || end_values[dim] != shape[dim])) ||
        (stride_val == -1 && (begin_values[dim] != -1 || end_values[dim] != -shape[dim] - 1)))
    {
      return {};
    }
    if (stride_val == -1)
      needs_reverse = true;
  }

  if (needs_reverse)
    return foldReverseInput();

  // Begin has to be all 0s, end has to map the input shape.
  for (int i = 0; i < input_type.getRank(); ++i)
  {
    if (begin_values[i] != 0 || end_values[i] != input_type.getDimSize(i))
      return {};
  }

  return getInput();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_STRIDED_SLICE_OP_H__
