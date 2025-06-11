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

#ifndef __CIRCLE_MLIR_PASS_OPS_CONSTANT_OF_SHAPE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_CONSTANT_OF_SHAPE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

namespace
{

// TODO move to helper when used more
template <typename TYPE>
mlir::DenseElementsAttr Expand(mlir::DenseElementsAttr &valueDens, std::vector<int64_t> &shapeVals)
{
  int32_t numElements = 1;
  for (int64_t v : shapeVals)
    numElements = numElements * v;

  mlir::Type valueType = valueDens.getElementType();
  mlir::RankedTensorType rType = RankedTensorType::get(shapeVals, valueType);

  // TODO fix for non single item
  TYPE value = valueDens.getValues<TYPE>()[0];
  llvm::SmallVector<TYPE, 4> constValue;
  for (int64_t idx = 0; idx < numElements; ++idx)
  {
    constValue.push_back(value);
  }
  return mlir::DenseElementsAttr::get(rType, mlir::ArrayRef<TYPE>(constValue));
}

} // namespace

// NOTE ONNX Constant --> Circle Const just copy
class ConvConstantOfShape : public mlir::OpConversionPattern<mlir::ONNXConstantOfShapeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXConstantOfShapeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXConstantOfShapeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConstantOfShapeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    // `input` is shape to create
    // `value` is single element to set
    mlir::Value input = adaptor.getInput();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvConstantOfShape intype: " << intype << "\n"; });

    if (!op.getValue().has_value())
    {
      LLVM_DEBUG({ llvm::dbgs() << "ConvConstantOfShape value none\n"; });
      return mlir::failure();
    }
    mlir::Attribute valueAttr = op.getValue().value();
    auto valueDens = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(valueAttr);
    if (!valueDens)
    {
      auto disValueAttr = mlir::dyn_cast_or_null<mlir::DisposableElementsAttr>(valueAttr);
      if (!disValueAttr)
      {
        LLVM_DEBUG({ llvm::dbgs() << "ConvConstantOfShape value not dense\n"; });
        return mlir::failure();
      }
      valueDens = disValueAttr.toDenseElementsAttr();
    }

    mlir::Type valueType = valueDens.getElementType();
    assert(valueDens.size() > 0);

    std::vector<int64_t> shapeVals;
    if (!ExtractConstantValues(input, shapeVals))
    {
      // input may not be a constant.
      // replace with equivalent ExpandOnnxOp so that folding to constant can be done later
      mlir::Value value = rewriter.create<ConstOp>(opLoc, valueDens);
      rewriter.replaceOpWithNewOp<ExpandOnnxOp>(op, op.getType(), value, input);
      return mlir::success();
    }

    mlir::DenseElementsAttr constAttr;
    if (valueType.isF32())
    {
      constAttr = Expand<float>(valueDens, shapeVals);
    }
    else if (valueType.isSignlessInteger(64))
    {
      constAttr = Expand<int64_t>(valueDens, shapeVals);
    }
    // TODO support more type
    else
    {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ConstOp>(op, op.getType(), constAttr);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CONSTANT_OF_SHAPE_OP_H__
