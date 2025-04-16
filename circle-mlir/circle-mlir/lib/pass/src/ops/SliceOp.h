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

#ifndef __CIRCLE_MLIR_PASS_OPS_SLICE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SLICE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <vector>
#include <algorithm>

namespace mlir
{
namespace Circle
{

class ConvSlice : public mlir::OpConversionPattern<mlir::ONNXSliceOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXSliceOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXSliceOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSliceOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();
    mlir::Value starts = adaptor.getStarts();
    mlir::Value ends = adaptor.getEnds();
    mlir::Value axes = adaptor.getAxes();
    mlir::Value steps = adaptor.getSteps();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvSlice name: " << op_name << "\n"; });

    auto intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    auto inshape = intype.getShape();
    LLVM_DEBUG({ llvm::dbgs() << "ConvSlice intype: " << intype << "\n"; });
    LLVM_DEBUG({
      llvm::dbgs() << "ConvSlice starts: "
                   << starts.getType().dyn_cast_or_null<mlir::RankedTensorType>() << "\n";
    });
    LLVM_DEBUG({
      llvm::dbgs() << "ConvSlice ends: "
                   << ends.getType().dyn_cast_or_null<mlir::RankedTensorType>() << "\n";
    });
    LLVM_DEBUG({
      llvm::dbgs() << "ConvSlice axes: "
                   << axes.getType().dyn_cast_or_null<mlir::RankedTensorType>() << "\n";
    });
    LLVM_DEBUG({
      llvm::dbgs() << "ConvSlice steps: "
                   << steps.getType().dyn_cast_or_null<mlir::RankedTensorType>() << "\n";
    });

    // Only covers when both axes and steps are constants
    if (!(IsConstant(axes) && IsConstant(steps)))
      return mlir::failure();

    std::vector<int64_t> axesValue, stepsValue;
    if (!ExtractConstantValues(axes, axesValue) || !ExtractConstantValues(steps, stepsValue))
      return mlir::failure();

    llvm::SmallVector<int32_t, 4> firstValue, lastValue, stridesValue;
    PrepareDefaultSliceValues(inshape, firstValue, lastValue, stridesValue);

    std::vector<int32_t> normalizedAxes;
    if (!PreProcessValues(axesValue, stepsValue, inshape, lastValue, stridesValue, normalizedAxes))
      return mlir::failure();

    if (IsConstant(starts) && IsConstant(ends))
    {
      std::vector<int64_t> startsValue, endsValue;
      if (!ExtractConstantValues(starts, startsValue) || !ExtractConstantValues(ends, endsValue))
        return mlir::failure();

      AdjustSliceValues(startsValue, endsValue, stridesValue, inshape, normalizedAxes);
      for (size_t i = 0; i < normalizedAxes.size(); ++i)
      {
        int32_t axis = normalizedAxes[i];
        firstValue[axis] = static_cast<int32_t>(startsValue[i]);
        lastValue[axis] = static_cast<int32_t>(endsValue[i]);
      }
      return ReplaceWithStridedSlice(op, rewriter, op_name, input, firstValue, lastValue,
                                     stridesValue);
    }
    else // when either starts or ends is not constant
    {
      return ReplaceWithDynamicStridedSlice(op, rewriter, op_name, input, starts, ends, inshape,
                                            normalizedAxes, firstValue, lastValue, stridesValue);
    }

    return mlir::failure();
  }

private:
  bool IsConstant(mlir::Value &value) const
  {
    if (auto constOp = dyn_cast_or_null<mlir::ONNXConstantOp>(value.getDefiningOp()))
      return true;
    else if (auto constOp2 = dyn_cast_or_null<mlir::Circle::ConstOp>(value.getDefiningOp()))
      return true;
    return false;
  }

  void PrepareDefaultSliceValues(llvm::ArrayRef<int64_t> inshape,
                                 llvm::SmallVector<int32_t, 4> &firstValue,
                                 llvm::SmallVector<int32_t, 4> &lastValue,
                                 llvm::SmallVector<int32_t, 4> &stridesValue) const
  {
    for (size_t d = 0; d < inshape.size(); ++d)
    {
      firstValue.push_back(0);
      lastValue.push_back(inshape[d]);
      stridesValue.push_back(1);
    }
  }

  bool PreProcessValues(const std::vector<int64_t> &axesValue,
                        const std::vector<int64_t> &stepsValue, llvm::ArrayRef<int64_t> inshape,
                        llvm::SmallVector<int32_t, 4> &lastValue,
                        llvm::SmallVector<int32_t, 4> &stridesValue,
                        std::vector<int32_t> &normalizedAxes) const
  {
    for (size_t i = 0; i < axesValue.size(); ++i)
    {
      int32_t axis = axesValue[i];
      if (axis < 0)
      {
        axis += inshape.size();
        if (axis < 0)
          return false;
      }
      normalizedAxes.push_back(axis);
      bool is_inshape_static = not mlir::ShapedType::isDynamic(inshape[axis]);
      if (!is_inshape_static)
        lastValue[axis] = 0;

      assert(stepsValue[i] != 0);
      stridesValue[axis] = static_cast<int32_t>(stepsValue[i]);
    }
    return true;
  }

  void AdjustSliceValues(std::vector<int64_t> &startsValue, std::vector<int64_t> &endsValue,
                         llvm::SmallVector<int32_t, 4> &stridesValue,
                         llvm::ArrayRef<int64_t> inshape,
                         const std::vector<int32_t> &normalizedAxes) const
  {
    for (size_t i = 0; i < normalizedAxes.size(); ++i)
    {
      int32_t axis = normalizedAxes[i];
      bool is_inshape_static = not mlir::ShapedType::isDynamic(inshape[axis]);

      // Clamp 'starts' and 'ends' values based on input shape for static input shapes
      // - If stride > 0: clamp within [-dim, dim]
      // - If stride < 0: clamp within [-dim-1, dim-1] to ensure inclusive range handling
      if (is_inshape_static)
      {
        if (stridesValue[axis] > 0)
        {
          startsValue[i] = std::clamp(startsValue[i], -inshape[axis], inshape[axis]);
          endsValue[i] = std::clamp(endsValue[i], -inshape[axis], inshape[axis]);
        }
        else if (stridesValue[axis] < 0)
        {
          startsValue[i] = std::clamp(startsValue[i], -inshape[axis] - 1, inshape[axis] - 1);
          endsValue[i] = std::clamp(endsValue[i], -inshape[axis] - 1, inshape[axis] - 1);
        }
      }
    }
  }

  mlir::LogicalResult ReplaceWithStridedSlice(mlir::ONNXSliceOp op,
                                              mlir::ConversionPatternRewriter &rewriter,
                                              const std::string &op_name, mlir::Value input,
                                              llvm::SmallVector<int32_t, 4> &firstValue,
                                              llvm::SmallVector<int32_t, 4> &lastValue,
                                              llvm::SmallVector<int32_t, 4> &stridesValue) const
  {
    // TODO fix mask value if there are any problems
    mlir::Location beg_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/begin"));
    mlir::Location end_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/end"));
    mlir::Location str_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/strides"));
    auto mask0 = rewriter.getI32IntegerAttr(0);
    auto begin = rewriter.create<ConstOp>(beg_loc, GetI32ElementsAttr(firstValue, &rewriter));
    auto end = rewriter.create<ConstOp>(end_loc, GetI32ElementsAttr(lastValue, &rewriter));
    auto strides = rewriter.create<ConstOp>(str_loc, GetI32ElementsAttr(stridesValue, &rewriter));
    rewriter.replaceOpWithNewOp<StridedSliceOp>(op, op.getType(), input, begin, end, strides, mask0,
                                                mask0, mask0, mask0, mask0);
    return mlir::success();
  }

  mlir::LogicalResult ReplaceWithDynamicStridedSlice(
    mlir::ONNXSliceOp op, mlir::ConversionPatternRewriter &rewriter, const std::string &op_name,
    mlir::Value input, mlir::Value starts, mlir::Value ends, llvm::ArrayRef<int64_t> inshape,
    const std::vector<int32_t> &normalizedAxes, llvm::SmallVector<int32_t, 4> &firstValue,
    llvm::SmallVector<int32_t, 4> &lastValue, llvm::SmallVector<int32_t, 4> &stridesValue) const
  {
    assert(normalizedAxes.size() == 1 && "Dynamic slice only supports single axis");

    // Dynamic input shape is allowed only the dim of axis is unknown
    // Other dims should be static
    int32_t axis = normalizedAxes[0];
    for (size_t d = 0; d < inshape.size(); ++d)
    {
      if (axis == d)
        continue;
      if (mlir::ShapedType::isDynamic(inshape[d]))
        return mlir::failure();
    }

    // Note: from model with starts and ends are NOT constant,
    // - Cast starts/ends to i32
    // - Select begin/end using axis mask
    // - Create strides and build StridedSliceOp

    llvm::SmallVector<bool, 4> conditionValue;

    for (size_t d = 0; d < inshape.size(); ++d)
      conditionValue.push_back(d == axis);

    Type dummyI32Type = RankedTensorType::get({1}, rewriter.getIntegerType(32));

    mlir::Location startsI32_loc =
      mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/startsI32"));
    mlir::Location endsI32_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/endsI32"));
    auto startsI32 = rewriter.create<CastOp>(startsI32_loc, dummyI32Type, starts);
    auto endsI32 = rewriter.create<CastOp>(endsI32_loc, dummyI32Type, ends);

    mlir::Location first_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/first"));
    mlir::Location last_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/last"));
    mlir::Location condition_loc =
      mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/condition"));
    auto first = rewriter.create<ConstOp>(first_loc, GetI32ElementsAttr(firstValue, &rewriter));
    auto last = rewriter.create<ConstOp>(last_loc, GetI32ElementsAttr(lastValue, &rewriter));
    auto condition =
      rewriter.create<ConstOp>(condition_loc, GetI1ElementsAttr(conditionValue, &rewriter));

    mlir::Location beg_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/begin"));
    mlir::Location end_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/end"));
    auto begin = rewriter.create<SelectV2Op>(beg_loc, first.getType(), condition, startsI32, first);
    auto end = rewriter.create<SelectV2Op>(end_loc, first.getType(), condition, endsI32, last);

    mlir::Location str_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/strides"));
    auto strides = rewriter.create<ConstOp>(str_loc, GetI32ElementsAttr(stridesValue, &rewriter));

    auto mask0 = rewriter.getI32IntegerAttr(0);

    rewriter.replaceOpWithNewOp<StridedSliceOp>(op, op.getType(), input, begin, end, strides, mask0,
                                                mask0, mask0, mask0, mask0);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_SLICE_OP_H__
