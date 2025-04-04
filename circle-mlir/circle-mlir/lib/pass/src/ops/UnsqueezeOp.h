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

#ifndef __CIRCLE_MLIR_PASS_OPS_UNSQUEEZE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_UNSQUEEZE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

namespace mlir
{
namespace Circle
{

namespace
{

bool valididateAxesValues(mlir::RankedTensorType &intype, mlir::RankedTensorType &outtype,
                          std::vector<int32_t> &axesValues)
{
  if (axesValues.size() == 0)
  {
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueeze axesValues none\n"; });
    return false;
  }

  // NOTE order of values in axes does not matter and can come in any order.
  auto inshape = intype.getShape();
  int32_t inshapeSize = static_cast<int32_t>(inshape.size());
  const int32_t outSize = inshapeSize + static_cast<int32_t>(axesValues.size());
  for (int32_t i = 0; i < axesValues.size(); ++i)
  {
    int32_t value = axesValues[i];
    axesValues[i] = value < 0 ? value + outSize : value;
  }
  sort(axesValues.begin(), axesValues.end());

  // NOTE input axes should not contain any duplicate entries
  if (std::adjacent_find(axesValues.begin(), axesValues.end()) != axesValues.end())
  {
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueeze axesValues duplicate\n"; });
    return false;
  }

  return true;
}

mlir::Value getUnsqueezeShape(mlir::ConversionPatternRewriter &rewriter,
                              mlir::RankedTensorType &intype, mlir::RankedTensorType &outtype,
                              std::string name, std::vector<int32_t> &axesValues)
{
  // 1) We can ignore axes and use outtype shape
  // 2) We can produce shape with intype shape + axes and compare with outtype shape
  // --> go with 2
  auto inshape = intype.getShape();
  int32_t inshapeSize = static_cast<int32_t>(inshape.size());

  std::vector<int32_t> values;
  for (int32_t i = 0; i < inshapeSize; ++i)
    values.push_back(static_cast<int32_t>(inshape[i]));
  for (size_t i = 0; i < axesValues.size(); ++i)
  {
    auto idx = axesValues.at(i);
    values.insert(values.begin() + idx, 1);
  }

  // expand to match output shape
  while (values.size() < static_cast<size_t>(inshapeSize) + axesValues.size())
    values.push_back(1);

  // verify that shape values should be same as output shape
  auto outshape = outtype.getShape();
  assert(outshape.size() == values.size());
  for (size_t i = 0; i < outshape.size(); ++i)
  {
    // NOTE we have to maintain uknown as -1
    if (values[i] == 0 && outshape[i] < 0)
      values[i] = -1;
    else
      assert(static_cast<int64_t>(values[i]) == outshape[i]);
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueeze: " << values[i] << " : " << outshape[i] << "\n"; });
  }

  // create new shape as ConstOp
  mlir::Location shapeLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  mlir::Type i32 = rewriter.getI32Type();
  mlir::RankedTensorType stype =
    mlir::RankedTensorType::get({static_cast<int64_t>(values.size())}, i32);
  return rewriter.create<ConstOp>(shapeLoc, mlir::DenseIntElementsAttr::get(stype, values));
}

} // namespace

class ConvUnsqueezeV11 : public mlir::OpConversionPattern<mlir::ONNXUnsqueezeV11Op>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXUnsqueezeV11Op>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXUnsqueezeV11Op::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXUnsqueezeV11Op op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueezeV11 name: " << op_name << "\n"; });

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueezeV11 intype: " << intype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueezeV11 outtype: " << outtype << "\n"; });

    mlir::ArrayAttr axes = adaptor.getAxes();

    std::vector<int32_t> axesValues;
    ExtractArrayAttrValues(axes, axesValues);

    if (!valididateAxesValues(intype, outtype, axesValues))
      return mlir::failure();

    mlir::Value shape =
      getUnsqueezeShape(rewriter, intype, outtype, op_name + "/shape", axesValues);
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), input, shape);

    return mlir::success();
  }
};

class ConvUnsqueeze : public mlir::OpConversionPattern<mlir::ONNXUnsqueezeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXUnsqueezeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXUnsqueezeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXUnsqueezeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();
    mlir::Value axes = adaptor.getAxes();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueeze name: " << op_name << "\n"; });

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueeze intype: " << intype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvUnsqueeze outtype: " << outtype << "\n"; });

    if (intype and outtype)
    {
      std::vector<int32_t> axesValues;
      if (ExtractConstantValues(axes, axesValues))
      {
        if (valididateAxesValues(intype, outtype, axesValues))
        {
          mlir::Value shape =
            getUnsqueezeShape(rewriter, intype, outtype, op_name + "/shape", axesValues);
          rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), input, shape);
          return mlir::success();
        }
      }
    }
    rewriter.replaceOpWithNewOp<UnsqueezeOnnxOp>(op, op.getType(), input, axes);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_UNSQUEEZE_OP_H__
