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

#ifndef __CIRCLE_MLIR_PASS_OPS_SUM_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SUM_OP_H__

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

class ConvReduceSum : public mlir::OpConversionPattern<mlir::ONNXReduceSumOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXReduceSumOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXReduceSumOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReduceSumOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSum name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSum axes: " << op.getAxes() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSum keepdims: " << op.getKeepdims() << "\n"; });
    LLVM_DEBUG({
      llvm::dbgs() << "ConvReduceSum noopWithEmptyAxes: " << op.getNoopWithEmptyAxes() << "\n";
    });

    if (notYetImplemented(op, adaptor))
      return mlir::failure();

    mlir::Value input = adaptor.getData();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSum intype: " << intype << "\n"; });

    auto keep_dims =
      adaptor.getKeepdims() ? rewriter.getBoolAttr(true) : rewriter.getBoolAttr(false);

    // TODO Enable to check if axes is constant
    std::vector<int32_t> axesValue;
    prepareAxes(op, intype, axesValue);

    mlir::Value axes = CreateI32Const(rewriter, axesValue, op_name + "/axes");

    rewriter.replaceOpWithNewOp<SumOp>(op, op.getType(), input, axes, keep_dims);

    return mlir::success();
  }

private:
  void prepareAxes(mlir::ONNXReduceSumOp &op, mlir::RankedTensorType &intype,
                   std::vector<int32_t> &values) const
  {
    mlir::Value op_axes = op.getAxes();
    bool axesNone = op_axes.getType().isa<mlir::NoneType>();

    if (axesNone)
    {
      // The attribute noop_with_empty_axes set as true is not supported yet.
      assert(op.getNoopWithEmptyAxes() == 0);

      // Default behavior with 'false' is to reduce all axes if 'axes' is empty.
      // set default values from 0 to dim - 1
      int32_t count = static_cast<int32_t>(intype.getShape().size());
      for (int32_t i = 0; i < count; ++i)
        values.push_back(i);
    }
    else
    {
      mlir::Value op_axes = op.getAxes();
      ExtractConstantValues(op_axes, values);
    }
  }

  bool notYetImplemented(mlir::ONNXReduceSumOp &op, OpAdaptor &adaptor) const
  {
    // TODO Support this op when NoopWithEmptyAxes is set to true.
    // When axes is empty and NoopWithEmptyAxes attribute is set to true, input tensor will not be
    // reduced, and the output tensor would be equivalent to input tensor.
    if (op.getNoopWithEmptyAxes() != 0)
      return true;

    return false;
  }
};

class ConvReduceSumV11 : public mlir::OpConversionPattern<mlir::ONNXReduceSumV11Op>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXReduceSumV11Op>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXReduceSumV11Op::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReduceSumV11Op op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSumV11 name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSumV11 axes: " << op.getAxes() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSumV11 keepdims: " << op.getKeepdims() << "\n"; });

    mlir::Value input = adaptor.getData();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvReduceSumV11 intype: " << intype << "\n"; });

    auto keep_dims =
      adaptor.getKeepdims() ? rewriter.getBoolAttr(true) : rewriter.getBoolAttr(false);

    std::vector<int32_t> axesValue;
    prepareAxes(op, intype, axesValue);

    mlir::Value axes = CreateI32Const(rewriter, axesValue, op_name + "/axes");

    rewriter.replaceOpWithNewOp<SumOp>(op, op.getType(), input, axes, keep_dims);

    return mlir::success();
  }

private:
  void prepareAxes(mlir::ONNXReduceSumV11Op &op, mlir::RankedTensorType &intype,
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

#endif // __CIRCLE_MLIR_PASS_OPS_SUM_OP_H__
