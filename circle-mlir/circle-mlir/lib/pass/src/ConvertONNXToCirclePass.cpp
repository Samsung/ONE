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
#include "ops/ArgMaxOp.h"
#include "ops/AveragePoolOp.h"
#include "ops/BatchNormalizationInferenceModeOp.h"
#include "ops/CastOp.h"
#include "ops/ClipOp.h"
#include "ops/ConcatOp.h"
#include "ops/ConstantOp.h"
#include "ops/ConstantOfShapeOp.h"
#include "ops/ConvOp.h"
#include "ops/ConvTransposeOp.h"
#include "ops/CosOp.h"
#include "ops/CumsumOp.h"
#include "ops/DequantizeLinearOp.h"
#include "ops/EqualOp.h"
#include "ops/ErfOp.h"
#include "ops/ExpOp.h"
#include "ops/ExpandOp.h"
#include "ops/FlattenOp.h"
#include "ops/FloorOp.h"
#include "ops/GatherOp.h"
#include "ops/GemmOp.h"
#include "ops/GlobalAveragePoolOp.h"
#include "ops/GreaterOp.h"
#include "ops/HardSigmoidOp.h"
#include "ops/HardSwishOp.h"
#include "ops/IdentityOp.h"
#include "ops/InstanceNormalizationOp.h"
#include "ops/LeakyReluOp.h"
#include "ops/LogOp.h"
#include "ops/MatMulOp.h"
#include "ops/MaxOp.h"
#include "ops/MaxPoolSingleOutOp.h"
#include "ops/MinOp.h"
#include "ops/NegOp.h"
#include "ops/NoneOp.h"
#include "ops/NotOp.h"
#include "ops/PadOp.h"
#include "ops/PowOp.h"
#include "ops/PReluOp.h"
#include "ops/QuantizeLinearOp.h"
#include "ops/RangeOp.h"
#include "ops/ReciprocalOp.h"
#include "ops/ReduceMaxOp.h"
#include "ops/ReduceMeanOp.h"
#include "ops/ReduceProdOp.h"
#include "ops/ReduceSumOp.h"
#include "ops/ReduceSumSquareOp.h"
#include "ops/ReluOp.h"
#include "ops/ReshapeOp.h"
#include "ops/ResizeOp.h"
#include "ops/ShapeOp.h"
#include "ops/SigmoidOp.h"
#include "ops/SinOp.h"
#include "ops/SliceOp.h"
#include "ops/SoftmaxOp.h"
#include "ops/SplitOp.h"
#include "ops/SqrtOp.h"
#include "ops/SqueezeOp.h"
#include "ops/TanhOp.h"
#include "ops/TileOp.h"
#include "ops/TransposeOp.h"
#include "ops/UnsqueezeOp.h"
#include "ops/WhereOp.h"

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
  return mlir::isa<mlir::Float16Type, mlir::Float32Type, mlir::Float64Type>(type);
}

inline bool isCircleInt(mlir::Type type)
{
  mlir::IntegerType intType = mlir::dyn_cast<mlir::IntegerType>(type);
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
    if (isCircleFloat(type) || isCircleInt(type) || mlir::isa<mlir::NoneType>(type))
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
  patterns.insert<ConvBinaryT<mlir::ONNXDivOp, mlir::Circle::DivOp>>(typeConverter, context);
  patterns.insert<ConvBinaryT<mlir::ONNXMulOp, mlir::Circle::MulOp>>(typeConverter, context);
  patterns.insert<ConvBinaryT<mlir::ONNXSubOp, mlir::Circle::SubOp>>(typeConverter, context);

  patterns.insert<ConvArgMax>(typeConverter, context);
  patterns.insert<ConvAveragePool>(typeConverter, context);
  patterns.insert<ConvBatchNormalizationInferenceMode>(typeConverter, context);
  patterns.insert<ConvCast>(typeConverter, context);
  patterns.insert<ConvClip>(typeConverter, context);
  patterns.insert<ConvConcat>(typeConverter, context);
  patterns.insert<ConvConstant>(typeConverter, context);
  patterns.insert<ConvConstantOfShape>(typeConverter, context);
  patterns.insert<ConvConv>(typeConverter, context);
  patterns.insert<ConvConvTranspose>(typeConverter, context);
  patterns.insert<ConvCos>(typeConverter, context);
  patterns.insert<ConvCumsum>(typeConverter, context);
  patterns.insert<ConvDequantizeLinear>(typeConverter, context);
  patterns.insert<ConvEqual>(typeConverter, context);
  patterns.insert<ConvErf>(typeConverter, context);
  patterns.insert<ConvExp>(typeConverter, context);
  patterns.insert<ConvExpand>(typeConverter, context);
  patterns.insert<ConvFlatten>(typeConverter, context);
  patterns.insert<ConvFloor>(typeConverter, context);
  patterns.insert<ConvGather>(typeConverter, context);
  patterns.insert<ConvGemm>(typeConverter, context);
  patterns.insert<ConvGlobalAveragePool>(typeConverter, context);
  patterns.insert<ConvGreater>(typeConverter, context);
  patterns.insert<ConvHardSigmoid>(typeConverter, context);
  patterns.insert<ConvHardSwish>(typeConverter, context);
  patterns.insert<ConvIdentity>(typeConverter, context);
  patterns.insert<ConvInstanceNormalization>(typeConverter, context);
  patterns.insert<ConvLeakyRelu>(typeConverter, context);
  patterns.insert<ConvLog>(typeConverter, context);
  patterns.insert<ConvMatMul>(typeConverter, context);
  patterns.insert<ConvMax>(typeConverter, context);
  patterns.insert<ConvMaxPoolSingleOut>(typeConverter, context);
  patterns.insert<ConvMin>(typeConverter, context);
  patterns.insert<ConvNeg>(typeConverter, context);
  patterns.insert<ConvNone>(typeConverter, context);
  patterns.insert<ConvNot>(typeConverter, context);
  patterns.insert<ConvPad>(typeConverter, context);
  patterns.insert<ConvPow>(typeConverter, context);
  patterns.insert<ConvPRelu>(typeConverter, context);
  patterns.insert<ConvQuantizeLinear>(typeConverter, context);
  patterns.insert<ConvRange>(typeConverter, context);
  patterns.insert<ConvReciprocal>(typeConverter, context);
  patterns.insert<ConvReduceMax>(typeConverter, context);
  patterns.insert<ConvReduceMaxV13>(typeConverter, context);
  patterns.insert<ConvReduceMean>(typeConverter, context);
  patterns.insert<ConvReduceMeanV13>(typeConverter, context);
  patterns.insert<ConvReduceProd>(typeConverter, context);
  patterns.insert<ConvReduceProdV13>(typeConverter, context);
  patterns.insert<ConvReduceSum>(typeConverter, context);
  patterns.insert<ConvReduceSumV11>(typeConverter, context);
  patterns.insert<ConvReduceSumSquareV13>(typeConverter, context);
  patterns.insert<ConvRelu>(typeConverter, context);
  patterns.insert<ConvReshape>(typeConverter, context);
  patterns.insert<ConvResize>(typeConverter, context);
  patterns.insert<ConvResizeV13>(typeConverter, context);
  patterns.insert<ConvShape>(typeConverter, context);
  patterns.insert<ConvSigmoid>(typeConverter, context);
  patterns.insert<ConvSin>(typeConverter, context);
  patterns.insert<ConvSlice>(typeConverter, context);
  patterns.insert<ConvSoftmax>(typeConverter, context);
  patterns.insert<ConvSoftmaxV11>(typeConverter, context);
  patterns.insert<ConvSplit>(typeConverter, context);
  patterns.insert<ConvSqrt>(typeConverter, context);
  patterns.insert<ConvSqueeze>(typeConverter, context);
  patterns.insert<ConvTanh>(typeConverter, context);
  patterns.insert<ConvTile>(typeConverter, context);
  patterns.insert<ConvTranspose>(typeConverter, context);
  patterns.insert<ConvUnsqueeze>(typeConverter, context);
  patterns.insert<ConvWhere>(typeConverter, context);

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
