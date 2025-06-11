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

#ifndef __CIRCLE_MLIR_PASS_OPS_REDUCE_PROD_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_REDUCE_PROD_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <vector>

namespace mlir
{
namespace Circle
{

class ConvReduceProd : public mlir::OpConversionPattern<mlir::ONNXReduceProdOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXReduceProdOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXReduceProdOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReduceProdOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProd name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProd axes: " << op.getAxes() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProd keepdims: " << op.getKeepdims() << "\n"; });

    mlir::Value input = adaptor.getData();
    mlir::Value op_axes = op.getAxes();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProd intype: " << intype << "\n"; });

    auto keep_dims =
      adaptor.getKeepdims() ? rewriter.getBoolAttr(true) : rewriter.getBoolAttr(false);

    int64_t getNoopWithEmptyAxes = op.getNoopWithEmptyAxes();

    // NOTE https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMean-18
    // Default behavior with 'false' is to reduce all axes.
    // When axes is empty and this attribute is set to true,
    // input tensor will not be reduced, and the output tensor would be equivalent to input tensor.
    if (getNoopWithEmptyAxes == 0)
    {
      // default behavior
      if (mlir::isa<mlir::NoneType>(op_axes.getType()))
      {
        std::vector<int32_t> axesValue;
        // set default values from 0 to dim - 1
        int32_t count = static_cast<int32_t>(intype.getShape().size());
        for (int32_t i = 0; i < count; ++i)
          axesValue.push_back(i);
        op_axes = CreateI32Const(rewriter, axesValue, op_name + "/axes");
      }
      rewriter.replaceOpWithNewOp<ReduceProdOp>(op, op.getType(), input, op_axes, keep_dims);
    }
    else
    {
      // TODO fix this (check if op->remove() works)
      return mlir::failure();
    }

    return mlir::success();
  }
};

class ConvReduceProdV13 : public mlir::OpConversionPattern<mlir::ONNXReduceProdV13Op>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXReduceProdV13Op>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXReduceProdV13Op::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReduceProdV13Op op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProdV13 name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProdV13 axes: " << op.getAxes() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProdV13 keepdims: " << op.getKeepdims() << "\n"; });

    mlir::Value input = adaptor.getData();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceProdV13 intype: " << intype << "\n"; });

    auto keep_dims =
      adaptor.getKeepdims() ? rewriter.getBoolAttr(true) : rewriter.getBoolAttr(false);

    std::vector<int32_t> axesValue;
    prepareAxes(op, intype, axesValue);

    mlir::Value axes = CreateI32Const(rewriter, axesValue, op_name + "/axes");

    rewriter.replaceOpWithNewOp<ReduceProdOp>(op, op.getType(), input, axes, keep_dims);

    return mlir::success();
  }

private:
  void prepareAxes(mlir::ONNXReduceProdV13Op &op, mlir::RankedTensorType &intype,
                   std::vector<int32_t> &values) const
  {
    auto axes = op.getAxes();
    if (axes.has_value())
    {
      values.clear();
      auto value = axes.value();
      for (int i = 0; i < value.size(); ++i)
      {
        auto v = GetIntValue<int32_t>(value, i);
        values.push_back(v);
      }
    }
    else
    {
      // set default values from 0 to dim - 1
      int32_t count = static_cast<int32_t>(intype.getShape().size());
      for (int32_t i = 0; i < count; ++i)
        values.push_back(i);
    }
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_REDUCE_PROD_OP_H__
