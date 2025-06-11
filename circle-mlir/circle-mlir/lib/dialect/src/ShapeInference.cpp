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

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include "circle-mlir/dialect/CircleDialect.h"

#include "utils/DynamicShapeUtils.h"
#include "utils/Padding.h"

#include <mlir/IR/Matchers.h>

namespace mlir
{
namespace Circle
{

// To reuse calculation for shape inference from CircleDialect.cpp
// TODO relocate to some header
LogicalResult ComputeConvWindowedOutputSize(int64_t input_size, int64_t filter_size,
                                            int64_t dilation_rate, int64_t stride,
                                            Circle::Padding padding, int64_t *output_size);

namespace
{

bool extractElements(ConstOp &const_op, std::vector<int64_t> &values)
{
  mlir::DenseElementsAttr dataAttr =
    mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(const_op.getValueAttr());
  if (dataAttr == nullptr)
    return false;
  if (!mlir::isa<mlir::IntegerType>(dataAttr.getElementType()))
    return false;

  for (auto value : dataAttr.getValues<llvm::APInt>())
    values.push_back(value.getSExtValue());
  return true;
}

template <typename OP> void dumpShape(OP op, const llvm::ArrayRef<int64_t> &inferred)
{
  LLVM_DEBUG({
    mlir::Location opLoc = op->getLoc();
    llvm::dbgs() << "-- " << typeid(OP).name() << " " << opLoc << " shape-inf: ";
    for (size_t i = 0; i < inferred.size(); ++i)
    {
      llvm::dbgs() << inferred[i];
      if (i < inferred.size() - 1)
        llvm::dbgs() << ",";
    }
    llvm::dbgs() << "\n";
  });
}

} // namespace

namespace
{

template <typename BINOP> bool inferBinShapes(BINOP &op, SmallVector<int64_t, 4> &inferred)
{
  auto out_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (out_type.hasStaticShape())
    return false;

  auto inp0_op = op.getOperand(0);
  auto inp0_type = mlir::cast<TensorType>(inp0_op.getType());
  auto inp1_op = op.getOperand(1);
  auto inp1_type = mlir::cast<TensorType>(inp1_op.getType());

  if (!OpTrait::util::getBroadcastedShape(inp0_type.getShape(), inp1_type.getShape(), inferred))
    return false;

  dumpShape<BINOP>(op, inferred);

  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::inferShapes()
{
  AddOp op = *this;
  SmallVector<int64_t, 4> inferred;
  if (!inferBinShapes<AddOp>(op, inferred))
    return;

  auto input0_op = getOperand(0);
  auto input0_type = mlir::cast<TensorType>(input0_op.getType());
  RankedTensorType inferred_type = RankedTensorType::get(inferred, input0_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// BatchMatMulOp
//===----------------------------------------------------------------------===//

void BatchMatMulOp::inferShapes(void)
{
  // referenced from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc BatchMatMulOp::verify()
  // and compiler/luci/service/src/Nodes/CircleBatchMatMul.cpp

  BatchMatMulOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // batch size in lhs and rhs must be broadcastable
  mlir::RankedTensorType x_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getX().getType());
  mlir::RankedTensorType y_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getY().getType());
  if (!x_ty || !y_ty)
    return;

  if (not y_ty.hasStaticShape())
    return;
  if (not x_ty.hasStaticShape())
    return;

  mlir::ArrayRef<int64_t> x_shape = x_ty.getShape();
  mlir::ArrayRef<int64_t> y_shape = y_ty.getShape();

  llvm::SmallVector<int64_t, 4> result_batch_shape;
  llvm::ArrayRef<int64_t> x_batches = x_shape.drop_back(2);
  llvm::ArrayRef<int64_t> y_batches = y_shape.drop_back(2);

  if (!mlir::OpTrait::util::getBroadcastedShape(x_batches, y_batches, result_batch_shape))
  {
    op.emitOpError() << "found incompatible broadcast batch dimensions for lhs shape " << x_ty
                     << " and rhs shape " << y_ty;
    return;
  }

  const auto adj_x = op.getAdjointLhs();
  const auto adj_y = op.getAdjointRhs();
  const auto x_rank = x_shape.size();
  const auto y_rank = y_shape.size();

  auto x_lhs = adj_x ? x_shape[x_rank - 1] : x_shape[x_rank - 2];
  auto x_rhs = adj_x ? x_shape[x_rank - 2] : x_shape[x_rank - 1];
  auto y_lhs = adj_y ? y_shape[y_rank - 1] : y_shape[y_rank - 2];
  auto y_rhs = adj_y ? y_shape[y_rank - 2] : y_shape[y_rank - 1];

  if (x_rhs != y_lhs)
  {
    op.emitOpError() << "found incompatible size for x_rhs " << x_rhs << " and y_lhs shape "
                     << y_lhs;
    return;
  }

  llvm::SmallVector<int64_t, 4> inferred(result_batch_shape.begin(), result_batch_shape.end());
  inferred.push_back(x_lhs);
  inferred.push_back(y_rhs);

  dumpShape<BatchMatMulOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, output_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::inferShapes(void)
{
  CastOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // follow input shape
  auto input_type = mlir::cast<TensorType>(op.getInput().getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<CastOp>(op, inferred);

  // preserve output dtype as-is
  RankedTensorType inferred_type = RankedTensorType::get(inferred, output_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//

int64_t GetConcatenationOpAxis(ConcatenationOp op)
{
  auto output_type = mlir::cast<RankedTensorType>(op.getOutput().getType());
  int32_t axis = op.getAxis();
  if (axis < 0)
    axis += output_type.getRank();
  return axis;
}

void ConcatenationOp::inferShapes()
{
  ConcatenationOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  auto operands = op.getOperands();
  int64_t rank = -1;
  const int64_t axis = GetConcatenationOpAxis(op);

  if (output_type.hasStaticShape())
    return;

  for (auto operand : operands)
  {
    auto shaped_type = mlir::cast<ShapedType>(operand.getType());
    if (!shaped_type.hasRank())
    {
      return;
    }

    if (rank == -1)
    {
      rank = shaped_type.getRank();
    }
    else if (shaped_type.getRank() != rank)
    {
      return;
    }
  }

  // Fill the size of axes other than the connection axis.
  SmallVector<int64_t, 4> new_shape(rank, ShapedType::kDynamic);
  for (auto operand : operands)
  {
    auto shaped_type = mlir::cast<ShapedType>(operand.getType());
    for (int64_t i = 0; i < rank; ++i)
    {
      if (i == axis)
        continue;

      int64_t dim_size = shaped_type.getDimSize(i);
      if (ShapedType::isDynamic(new_shape[i]))
      {
        new_shape[i] = dim_size;
      }
      else if (new_shape[i] != dim_size)
      {
        return;
      }
    }
  }

  // Fill the size of the connection axis
  int64_t axis_dim_size = 0;
  for (auto operand : operands)
  {
    auto shaped_type = mlir::cast<ShapedType>(operand.getType());
    int64_t dim_size = shaped_type.getDimSize(axis);
    if (ShapedType::isDynamic(dim_size))
      axis_dim_size = ShapedType::kDynamic;
    else if (not ShapedType::isDynamic(axis_dim_size))
      axis_dim_size += dim_size;
  }
  new_shape[axis] = axis_dim_size;

  dumpShape<ConcatenationOp>(op, new_shape);

  auto inferred_type =
    mlir::Circle::GetTypeFromTensorShape(new_shape, output_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

void Conv2DOp::inferShapes()
{
  Conv2DOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_ty = mlir::dyn_cast_or_null<RankedTensorType>(input_op.getType());
  if (!input_ty.hasStaticShape())
    return;

  auto filter_op = getOperand(1);
  auto filter_ty = mlir::dyn_cast_or_null<RankedTensorType>(filter_op.getType());
  // If indeed both input type & filter type are ranked type and have ranks.
  // We will need to check their ranks are valid.
  if ((input_ty && input_ty.hasRank() && input_ty.getRank() != 4) ||
      (filter_ty && filter_ty.hasRank() && filter_ty.getRank() != 4))
  {
    return;
  }

  // If either input or filter is unranked, we will just return unranked output shape.
  if (!input_ty || !filter_ty || !input_ty.hasRank() || !filter_ty.hasRank())
  {
    return;
  }

  auto stride_h = op.getStrideHAttr().getInt();
  auto stride_w = op.getStrideWAttr().getInt();
  auto dilation_h = op.getDilationHFactorAttr().getInt();
  auto dilation_w = op.getDilationWFactorAttr().getInt();

  // We don't have EXPLICIT PADDING in Circle.
  auto paddings = op.getPadding();
  mlir::Circle::Padding padding;
  auto padding_is_valid = GetPaddingFromString(paddings.str(), &padding);
  if (!padding_is_valid.ok())
  {
    return;
  }

  // Output always have rank 4. All dimensions are initialized to
  // dynamic size and can be partially inferred.
  // TFL's conv2d is always NHWC format & the filter is OHWI.
  SmallVector<int64_t, 4> return_shape(4, ShapedType::kDynamic);
  return_shape[0] = input_ty.getDimSize(0);
  return_shape[3] = filter_ty.getDimSize(0);

  // Spatial dimensions can be inferred only when both input and filter are
  // ranked because we need to get their spatial dimensions.

  // Height.
  if (!input_ty.isDynamicDim(1) && !filter_ty.isDynamicDim(1))
  {
    int64_t output_height;
    if (failed(ComputeConvWindowedOutputSize(input_ty.getDimSize(1), filter_ty.getDimSize(1),
                                             dilation_h, stride_h, padding, &output_height)))
    {
      return;
    }
    return_shape[1] = output_height;
  }

  // Width.
  if (!input_ty.isDynamicDim(2) && !filter_ty.isDynamicDim(2))
  {
    int64_t output_width;
    if (failed(ComputeConvWindowedOutputSize(input_ty.getDimSize(2), filter_ty.getDimSize(2),
                                             dilation_w, stride_w, padding, &output_width)))
    {
      return;
    }
    return_shape[2] = output_width;
  }

  dumpShape<Conv2DOp>(op, return_shape);

  RankedTensorType inferred_type = RankedTensorType::get(return_shape, input_ty.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

void CosOp::inferShapes(void)
{
  CosOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getY().getType());
  if (output_type.hasStaticShape())
    return;

  // follow input shape
  auto input_type = mlir::cast<TensorType>(op.getX().getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<CosOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

void CustomOp::inferShapes()
{
  CustomOp op = *this;
  auto outputs = op.getOutput();
  bool all_static = true;
  for (auto output : outputs)
  {
    auto output_type = mlir::cast<ShapedType>(output.getType());
    if (not output_type.hasStaticShape())
    {
      all_static = false;
      break;
    }
  }
  if (all_static)
    return;

  if (op.getCustomCode() == "Erf")
  {
    assert(op.getInput().size() == 1);
    assert(op.getOutput().size() == 1);

    auto input_op = getOperand(0);
    auto input_type = mlir::cast<TensorType>(input_op.getType());
    auto input_shape = input_type.getShape();
    llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

    dumpShape<CustomOp>(op, inferred);

    RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
    getResult(0).setType(inferred_type);
  }
}

//===----------------------------------------------------------------------===//
// DepthwiseConv2DOp
//===----------------------------------------------------------------------===//

void DepthwiseConv2DOp::inferShapes()
{
  DepthwiseConv2DOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_ty = mlir::dyn_cast_or_null<RankedTensorType>(input_op.getType());
  if (!input_ty.hasStaticShape())
    return;

  auto filter = getOperand(1);
  auto filter_ty = mlir::dyn_cast_or_null<RankedTensorType>(filter.getType());
  // If indeed both input type & filter type are ranked type and have ranks.
  // We will need to check their ranks are valid.
  if ((input_ty && input_ty.hasRank() && input_ty.getRank() != 4) ||
      (filter_ty && filter_ty.hasRank() && filter_ty.getRank() != 4))
  {
    return;
  }

  // If either input or filter is unranked, we will just return unranked output shape.
  if (!input_ty || !filter_ty || !input_ty.hasRank() || !filter_ty.hasRank())
  {
    return;
  }

  auto stride_h = op.getStrideHAttr().getInt();
  auto stride_w = op.getStrideWAttr().getInt();
  auto dilation_h = op.getDilationHFactorAttr().getInt();
  auto dilation_w = op.getDilationWFactorAttr().getInt();

  // We don't have EXPLICIT PADDING in Circle.
  auto paddings = op.getPadding();
  mlir::Circle::Padding padding;
  auto padding_is_valid = GetPaddingFromString(paddings.str(), &padding);
  if (!padding_is_valid.ok())
  {
    return;
  }

  // Output always have rank 4. All dimensions are initialized to
  // dynamic size and can be partially inferred.
  // DepthwiseConv2D input is NHWC format & the filter is also NHWC.
  SmallVector<int64_t, 4> return_shape(4, ShapedType::kDynamic);
  return_shape[0] = input_ty.getDimSize(0);
  return_shape[3] = filter_ty.getDimSize(3);

  // Spatial dimensions can be inferred only when both input and filter are
  // ranked because we need to get their spatial dimensions.

  // Height.
  if (!input_ty.isDynamicDim(1) && !filter_ty.isDynamicDim(1))
  {
    int64_t output_height;
    if (failed(ComputeConvWindowedOutputSize(input_ty.getDimSize(1), filter_ty.getDimSize(1),
                                             dilation_h, stride_h, padding, &output_height)))
    {
      return;
    }
    return_shape[1] = output_height;
  }

  // Width.
  if (!input_ty.isDynamicDim(2) && !filter_ty.isDynamicDim(2))
  {
    int64_t output_width;
    if (failed(ComputeConvWindowedOutputSize(input_ty.getDimSize(2), filter_ty.getDimSize(2),
                                             dilation_w, stride_w, padding, &output_width)))
    {
      return;
    }
    return_shape[2] = output_width;
  }

  dumpShape<DepthwiseConv2DOp>(op, return_shape);

  RankedTensorType inferred_type = RankedTensorType::get(return_shape, input_ty.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::inferShapes()
{
  DivOp op = *this;
  SmallVector<int64_t, 4> inferred;
  if (!inferBinShapes<DivOp>(op, inferred))
    return;

  auto input0_op = getOperand(0);
  auto input0_type = mlir::cast<TensorType>(input0_op.getType());
  RankedTensorType inferred_type = RankedTensorType::get(inferred, input0_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// FullyConnectedOp
//===----------------------------------------------------------------------===//

void FullyConnectedOp::inferShapes(void)
{
  FullyConnectedOp op = *this;
  auto output_type = mlir::cast<ShapedType>((*op.getOutput().begin()).getType());
  if (output_type.hasStaticShape())
    return;

  auto filter_type = mlir::cast<TensorType>(op.getFilter().getType());
  if (not filter_type.hasStaticShape())
    return;
  auto filter_shape = filter_type.getShape();

  auto input_type = mlir::cast<TensorType>(op.getInput().getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred;

  if (op.getKeepNumDims())
  {
    llvm::SmallVector<int64_t, 4> in_inferred(input_shape.begin(), input_shape.end());
    in_inferred[in_inferred.size() - 1] = filter_shape[0];
    inferred = in_inferred;
  }
  else
  {
    if (input_type.hasStaticShape())
    {
      int64_t ele_size = 1;
      for (int64_t i = 0; i < input_shape.size() - 1; ++i)
        ele_size *= input_shape[i];
      inferred.push_back(ele_size);
    }
    else
    {
      inferred.push_back(ShapedType::kDynamic);
    }
    inferred.push_back(filter_shape[0]);
  }

  dumpShape<FullyConnectedOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, output_type.getElementType());
  getResult(0).setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// InstanceNormOp
//===----------------------------------------------------------------------===//

void InstanceNormOp::inferShapes()
{
  InstanceNormOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  if (!input_type.hasStaticShape())
    return;

  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<InstanceNormOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// LogisticOp
//===----------------------------------------------------------------------===//

void LogisticOp::inferShapes()
{
  LogisticOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getY().getType());
  if (output_type.hasStaticShape())
    return;

  auto input_type = mlir::cast<TensorType>(op.getX().getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<LogisticOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// MaxPool2DOp
//===----------------------------------------------------------------------===//

void MaxPool2DOp::inferShapes()
{
  MaxPool2DOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand();
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  if (!input_type.hasStaticShape() || input_type.getRank() != 4)
    return;
  auto input_shape = input_type.getShape();

  uint32_t input_height = input_shape[1];
  uint32_t input_width = input_shape[2];
  uint32_t stride_height = op.getStrideH();
  uint32_t stride_width = op.getStrideW();
  uint32_t window_height = op.getFilterHeight();
  uint32_t window_width = op.getFilterWidth();
  uint32_t dilation_height = 1;
  uint32_t dilation_width = 1;
  uint32_t effective_window_height = dilation_height * (window_height - 1) + 1;
  uint32_t effective_window_width = dilation_width * (window_width - 1) + 1;

  uint32_t output_height = 0;
  uint32_t output_width = 0;

  if (op.getPadding().str() == "VALID")
  {
    assert(input_height + stride_height > effective_window_height);
    assert(input_width + stride_width > effective_window_width);
    output_height = (input_height + stride_height - effective_window_height) / stride_height;
    output_width = (input_width + stride_width - effective_window_width) / stride_width;
  }
  else if (op.getPadding().str() == "SAME")
  {
    output_height = (input_height + stride_height - 1) / stride_height;
    output_width = (input_width + stride_width - 1) / stride_width;
  }

  llvm::SmallVector<int64_t, 4> inferred;
  inferred.push_back(input_shape[0]);
  inferred.push_back(output_height);
  inferred.push_back(output_width);
  inferred.push_back(input_shape[3]);

  dumpShape<MaxPool2DOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

void MeanOp::inferShapes()
{
  MeanOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  if (!input_type.hasStaticShape())
    return;

  // skip if axes input is not constant
  mlir::Operation *is_const = getOperand(1).getDefiningOp();
  if (!mlir::isa_and_nonnull<ConstOp>(is_const))
    return;
  auto const_op = cast<ConstOp>(is_const);
  std::vector<int64_t> axis_values;
  if (!extractElements(const_op, axis_values))
    return;
  int64_t in_rank = input_type.getRank();
  for (int64_t &axis : axis_values)
  {
    if (axis < 0)
      axis += in_rank;
  }

  auto input_shape = input_type.getShape();

  llvm::SmallVector<int64_t, 4> inferred;

  if (op.getKeepDims())
  {
    inferred.assign(input_shape.begin(), input_shape.end());
    for (int64_t axis : axis_values)
      inferred[axis] = 1;
  }
  else
  {
    std::vector<bool> check_reduce(input_type.getRank(), false);
    for (int64_t i = 0; i < axis_values.size(); ++i)
    {
      int64_t reduce_at = axis_values[i];
      check_reduce.at(reduce_at) = true;
    }

    for (int64_t i = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i) == false)
        inferred.push_back(input_shape[i]);
  }

  dumpShape<MeanOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// MirrorPadOp
//===----------------------------------------------------------------------===//

void MirrorPadOp::inferShapes()
{
  MirrorPadOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  if (!input_type.hasStaticShape())
    return;

  // skip if size input is not constant
  mlir::Operation *is_const = getOperand(1).getDefiningOp();
  if (!mlir::isa_and_nonnull<ConstOp>(is_const))
    return;
  auto const_op = cast<ConstOp>(is_const);
  std::vector<int64_t> padding_values;
  if (!extractElements(const_op, padding_values))
    return;

  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred;
  int64_t num_dims = input_type.getRank();
  for (int64_t i = 0; i < num_dims; ++i)
  {
    const int64_t padding_before = padding_values[i * 2];
    const int64_t padding_after = padding_values[i * 2 + 1];
    assert(padding_before >= 0 && padding_after >= 0);
    int64_t output_val = input_shape[i] + padding_before + padding_after;
    inferred.push_back(output_val);
  }

  dumpShape<MirrorPadOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::inferShapes()
{
  MulOp op = *this;
  SmallVector<int64_t, 4> inferred;
  if (!inferBinShapes<MulOp>(op, inferred))
    return;

  auto input0_op = getOperand(0);
  auto input0_type = mlir::cast<TensorType>(input0_op.getType());
  RankedTensorType inferred_type = RankedTensorType::get(inferred, input0_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

void PadOp::inferShapes()
{
  PadOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  if (!input_type.hasStaticShape())
    return;

  // skip if size input is not constant
  mlir::Operation *is_const = getOperand(1).getDefiningOp();
  if (!mlir::isa_and_nonnull<ConstOp>(is_const))
    return;
  auto const_op = cast<ConstOp>(is_const);
  std::vector<int64_t> padding_values;
  if (!extractElements(const_op, padding_values))
    return;

  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred;
  int64_t num_dims = input_type.getRank();
  for (int64_t i = 0; i < num_dims; ++i)
  {
    const int64_t padding_before = padding_values[i * 2];
    const int64_t padding_after = padding_values[i * 2 + 1];
    assert(padding_before >= 0 && padding_after >= 0);
    int64_t output_val = input_shape[i] + padding_before + padding_after;
    inferred.push_back(output_val);
  }

  dumpShape<PadOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// PReluOp
//===----------------------------------------------------------------------===//

void PReluOp::inferShapes()
{
  PReluOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<PReluOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::inferShapes()
{
  ReluOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getY().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = op.getX();
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<ReluOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

bool GetReshapeOutputType(const Value input, const Value shape, RankedTensorType &output_ty)
{
  auto input_ty = mlir::cast<TensorType>(input.getType());
  auto element_ty = input_ty.getElementType();
  auto shape_ty = mlir::dyn_cast<RankedTensorType>(shape.getType());
  if (!shape_ty)
    return false;

  if (shape_ty.getRank() != 1)
  {
    assert(false);
    return false;
  }

  mlir::DenseIntElementsAttr shape_attr;
  if (!matchPattern(shape, m_Constant(&shape_attr)))
  {
    // If only shape of `shape` is known, return ranked but dynamic output shape.
    if (shape_ty.hasStaticShape())
    {
      llvm::SmallVector<int64_t, 8> dynamic_shape(shape_ty.getDimSize(0), ShapedType::kDynamic);
      output_ty = GetTypeFromTensorShape(dynamic_shape, element_ty);
      return true;
    }
    return false;
  }

  auto input_shape = input_ty.getShape();
  llvm::SmallVector<int64_t, 4> input_sh_vals(input_shape.begin(), input_shape.end());

  // Detect if reshape output shape is folded.
  int unknown_index = -1;
  // The product of constant shape argument excluding unknown dimension.
  int64_t shape_ty_size = 1;
  llvm::SmallVector<int64_t, 8> output_ty_shape;
  output_ty_shape.reserve(shape_attr.getNumElements());
  int64_t dim_index = 0;
  for (const auto &dim : llvm::enumerate(shape_attr.getValues<APInt>()))
  {
    int64_t size = dim.value().getSExtValue();
    if (size == kDynamicSize || size == ShapedType::kDynamic)
    {
      if (unknown_index != -1)
      {
        LLVM_DEBUG({
          llvm::dbgs() << "Got two or more unknown: ";
          llvm::dbgs() << unknown_index << " and " << dim.index();
          llvm::dbgs() << "\n";
        });
        return false;
      }
      unknown_index = dim.index();
    }
    else if (size == 0)
    {
      if (dim_index < input_sh_vals.size())
        size = input_sh_vals[dim_index];
      else
      {
        // stop for debug version to find a mdoel with this case
        assert(false);
        size = 1;
      }
      shape_ty_size *= size;
    }
    else if (size > 0)
    {
      shape_ty_size *= size;
    }
    else
    {
      LLVM_DEBUG({
        llvm::dbgs() << "Got invalid size ";
        llvm::dbgs() << size << " at " << dim.index();
        llvm::dbgs() << "\n";
      });
      return false;
    }
    output_ty_shape.push_back(size);
    dim_index++;
  }

  if (!input_ty.hasStaticShape())
  {
    output_ty = GetTypeFromTensorShape(output_ty_shape, element_ty);
    return true;
  }

  // Compute the value of the unknown dimension.
  if (unknown_index != -1)
  {
    // Compute number of elements in tensor shape.
    int64_t input_ty_size = 1;
    for (const auto &dim : input_ty.getShape())
    {
      // stop for debug version to find this case
      assert(dim > 0);
      if (dim > 0)
      {
        input_ty_size *= dim;
      }
    }

    const int64_t missing_dim = input_ty_size / shape_ty_size;
    if (shape_ty_size * missing_dim != input_ty_size)
    {
      LLVM_DEBUG({
        llvm::dbgs() << "requires 'input' number of elements be a multiple of ";
        llvm::dbgs() << shape_ty_size << ", but got " << input_ty_size;
        llvm::dbgs() << "\n";
      });
      return false;
    }

    // Set the unknown dimension such that total number of elements remain
    // constant.
    output_ty_shape[unknown_index] = missing_dim;
  }

  output_ty = GetTypeFromTensorShape(output_ty_shape, element_ty);
  return true;
}

void ReshapeOp::inferShapes()
{
  ReshapeOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  const Value input = op.getInput();
  const Value shape = op.getShape();

  RankedTensorType inferred_type;
  if (GetReshapeOutputType(input, shape, inferred_type))
  {
    auto output_shape = inferred_type.getShape();
    dumpShape<ReshapeOp>(op, output_shape);

    getResult().setType(inferred_type);
  }
}

//===----------------------------------------------------------------------===//
// ResizeNearestNeighborOp
//===----------------------------------------------------------------------===//

void ResizeNearestNeighborOp::inferShapes()
{
  ResizeNearestNeighborOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // if input is dynamic, skip shape infer
  auto input_op = getOperand(0);
  auto input_type = mlir::cast<TensorType>(input_op.getType());
  if (!input_type.hasStaticShape())
    return;

  // skip if size input is not constant
  mlir::Operation *is_const = getOperand(1).getDefiningOp();
  if (!mlir::isa_and_nonnull<ConstOp>(is_const))
    return;
  auto const_op = cast<ConstOp>(is_const);
  std::vector<int64_t> size_values;
  if (!extractElements(const_op, size_values))
    return;

  auto input_shape = input_type.getShape();
  int64_t size_h = 0, size_w = 0;
  if (size_values.size() == 4)
  {
    // where size is NCHW
    size_h = size_values[2];
    size_w = size_values[3];
  }
  else if (size_values.size() == 2)
  {
    // where size is HW
    size_h = size_values[0];
    size_w = size_values[1];
  }
  else
  {
    // this looks invalid
    assert(false);
    return;
  }

  // now shape can be fixed
  llvm::SmallVector<int64_t, 4> inferred;
  inferred.push_back(input_shape[0]);
  inferred.push_back(size_h);
  inferred.push_back(size_w);
  inferred.push_back(input_shape[3]);

  dumpShape<ResizeNearestNeighborOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

void SinOp::inferShapes(void)
{
  SinOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getY().getType());
  if (output_type.hasStaticShape())
    return;

  // follow input shape
  auto input_type = mlir::cast<TensorType>(op.getX().getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<SinOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

void SqrtOp::inferShapes()
{
  SqrtOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getY().getType());
  if (output_type.hasStaticShape())
    return;

  auto input_type = mlir::cast<TensorType>(op.getX().getType());
  auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

  dumpShape<SqrtOp>(op, inferred);

  RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// StridedSliceOp
//===----------------------------------------------------------------------===//

void StridedSliceOp::inferShapes()
{
  StridedSliceOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  // Currently only support all masks being 0.
  if (getBeginMask() != 0 || getEndMask() != 0 || getEllipsisMask() != 0 || getNewAxisMask() != 0 ||
      getShrinkAxisMask() != 0)
    return;

  // Currently only support all strides being 1.
  mlir::DenseIntElementsAttr strides_dense_elem_attr;
  if (!matchPattern(getStrides(), m_Constant(&strides_dense_elem_attr)))
    return;
  for (auto stride_ele : strides_dense_elem_attr)
  {
    if (stride_ele.getSExtValue() != 1)
      return;
  }

  auto op_type = op.getType();
  auto rank = op_type.getRank();

  mlir::DenseIntElementsAttr begin_element;
  mlir::DenseIntElementsAttr end_element;
  if (!matchPattern(getBegin(), m_Constant(&begin_element)))
    return;
  if (!matchPattern(getEnd(), m_Constant(&end_element)))
    return;

  llvm::SmallVector<int64_t, 4> begins;
  llvm::SmallVector<int64_t, 4> ends;

  auto input_type = getInput().getType();
  auto input_shape = input_type.getShape();
  for (const auto &[i, begin_int] : llvm::enumerate(begin_element.getValues<APInt>()))
  {
    int64_t val = begin_int.getSExtValue();
    begins.push_back((val < 0) ? val + input_shape[i] : val);
  }
  for (const auto &[i, end_int] : llvm::enumerate(end_element.getValues<APInt>()))
  {
    int64_t val = end_int.getSExtValue();
    ends.push_back((val < 0) ? val + input_shape[i] : val);
  }

  llvm::SmallVector<int64_t, 4> inferred;
  for (int i = 0; i < rank; ++i)
  {
    if (begins[i] >= ends[i])
      inferred.push_back(ShapedType::kDynamic);
    else
      inferred.push_back(ends[i] - begins[i]);
  }

  dumpShape<StridedSliceOp>(op, inferred);

  RankedTensorType inferred_type =
    RankedTensorType::get(inferred, FloatType::getF32(op->getContext()));

  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::inferShapes()
{
  SubOp op = *this;
  SmallVector<int64_t, 4> inferred;
  if (!inferBinShapes<SubOp>(op, inferred))
    return;

  auto input0_op = getOperand(0);
  auto input0_type = mlir::cast<TensorType>(input0_op.getType());
  RankedTensorType inferred_type = RankedTensorType::get(inferred, input0_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::inferShapes()
{
  TransposeOp op = *this;
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (output_type.hasStaticShape())
    return;

  auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
  auto perm_type = mlir::cast<ShapedType>(op.getPerm().getType());

  if (input_type.hasStaticShape() && perm_type.hasStaticShape())
  {
    if (perm_type.getNumElements() != input_type.getRank())
    {
      return;
    }
  }

  mlir::DenseIntElementsAttr perm;
  if (!matchPattern(op.getPerm(), m_Constant(&perm)))
  {
    return;
  }

  llvm::SmallVector<int64_t, 4> perm_list;
  for (const auto &perm_element : perm.getValues<APInt>())
  {
    const int64_t val = perm_element.getSExtValue();
    perm_list.push_back(val);
  }

  // Get transposed shape and set it to the output type
  if (input_type.hasStaticShape() && !output_type.hasStaticShape())
  {
    llvm::SmallVector<int64_t, 4> transposed_shape;
    for (int64_t axis : perm_list)
    {
      transposed_shape.push_back(input_type.getDimSize(axis));
    }

    dumpShape<TransposeOp>(op, transposed_shape);

    auto inferred_type =
      mlir::Circle::GetTypeFromTensorShape(transposed_shape, input_type.getElementType());
    getResult().setType(inferred_type);
  }
}

} // namespace Circle
} // namespace mlir
