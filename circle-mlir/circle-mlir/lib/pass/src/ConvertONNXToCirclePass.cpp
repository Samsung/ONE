/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019-2022 The IBM Research Authors.
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

#include "ConvertONNXToCirclePass.h"

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

// NOTE lets use names from ONNX Op for the conversion class and the file name.
//    ONNX: ONNXAbcdOp
//   class: ConvAbcd
//    file: AbcdOp.h
#include "ops/ConstantOp.h"
#include "ops/TransposeOp.h"

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/IR/Operation.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Support/LogicalResult.h>

// from onnx-mlir source
#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <set>

namespace mlir
{
namespace Circle
{

namespace
{

// Convert for binary input with Activation; such as Add, Sub, Mul, Div, ...
template <typename ONNXOpT, typename CircleOpT>
class ConvBinaryT : public mlir::OpConversionPattern<ONNXOpT>
{
public:
  using mlir::OpConversionPattern<ONNXOpT>::OpConversionPattern;
  using OpAdaptor = typename ONNXOpT::Adaptor;

  mlir::LogicalResult matchAndRewrite(ONNXOpT op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value lhs = adaptor.getA();
    mlir::Value rhs = adaptor.getB();

    rewriter.replaceOpWithNewOp<CircleOpT>(op, op.getType(), lhs, rhs, "NONE");

    return mlir::success();
  }
};

} // namespace

namespace
{

inline bool isCircleFloat(mlir::Type type)
{
  return type.isa<mlir::Float16Type, mlir::Float32Type, mlir::Float64Type>();
}

inline bool isCircleInt(mlir::Type type)
{
  mlir::IntegerType intType = type.dyn_cast<mlir::IntegerType>();
  if (intType)
  {
    std::set<unsigned> intWidth{1, 8, 16, 32, 64};
    auto w = intType.getWidth();
    if (intWidth.find(w) != intWidth.end())
    {
      return intType.isSignless() || (w == 16 && intType.isSigned()) ||
             (w == 8 && intType.isUnsigned());
    }
  }
  return false;
}

} // namespace

struct ConvertONNXToCirclePass
  : public mlir::PassWrapper<ConvertONNXToCirclePass, mlir::OperationPass<mlir::func::FuncOp>>
{
  ConvertONNXToCirclePass() = default;
  ConvertONNXToCirclePass(const ConvertONNXToCirclePass &pass)
    : mlir::PassWrapper<ConvertONNXToCirclePass, OperationPass<mlir::func::FuncOp>>()
  {
    // Do nothing
  }

  llvm::StringRef getArgument() const override { return "onnx-to-circle"; }

  llvm::StringRef getDescription() const override { return "ONNX to Circle"; }

  Option<std::string> target{*this, "target", ::llvm::cl::desc("ONNX dialect to Circle dialect"),
                             ::llvm::cl::init("")};

  void runOnOperation() final;
};

void ConvertONNXToCirclePass::runOnOperation()
{
  mlir::func::FuncOp function = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::ConversionTarget target(getContext());

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    // TODO support mode dtypes
    // NOTE Conv2D without bias is NoneType
    if (isCircleFloat(type) || isCircleInt(type) || type.isa<mlir::NoneType>())
      return type;
    LLVM_DEBUG({ llvm::dbgs() << "TypeConverter Type None\n"; });
    return std::nullopt;
  });
  typeConverter.addConversion([&](TensorType type) -> std::optional<Type> {
    if (typeConverter.isLegal(type.getElementType()))
      return type;
    LLVM_DEBUG({ llvm::dbgs() << "TypeConverter TensorType None\n"; });
    return std::nullopt;
  });

  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::Circle::CIRDialect>();

  mlir::RewritePatternSet patterns(context);
  // NOTE use name from ONNX Op, suffix T for templates
  patterns.insert<ConvBinaryT<mlir::ONNXAddOp, mlir::Circle::AddOp>>(typeConverter, context);

  patterns.insert<ConvConstant>(typeConverter, context);
  patterns.insert<ConvTranspose>(typeConverter, context);

  auto res = mlir::applyFullConversion(function, target, std::move(patterns));
  if (mlir::failed(res))
  {
    return signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createConvertONNXToCirclePass(void)
{
  return std::make_unique<mlir::Circle::ConvertONNXToCirclePass>();
}

} // namespace Circle
} // namespace mlir
