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

#ifndef __CIRCLE_MLIR_PASS_OPS_CONSTANT_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_CONSTANT_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>
#include <src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp>

namespace mlir
{
namespace Circle
{

// NOTE ONNX Constant --> Circle Const just copy
class ConvConstant : public mlir::OpConversionPattern<mlir::ONNXConstantOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXConstantOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXConstantOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConstantOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    auto valueAttr = op.getValueAttr().dyn_cast_or_null<mlir::DenseElementsAttr>();
    if (valueAttr == nullptr)
    {
      auto disValueAttr = op.getValueAttr().dyn_cast_or_null<mlir::DisposableElementsAttr>();
      if (disValueAttr)
      {
        // TODO revise this: not sure using toDenseElementsAttr is good or not
        valueAttr = disValueAttr.toDenseElementsAttr();
        if (valueAttr == nullptr)
          return mlir::failure();
      }
    }

    rewriter.replaceOpWithNewOp<ConstOp>(op, op.getType(), valueAttr);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CONSTANT_OP_H__
