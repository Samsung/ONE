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

#ifndef __CIRCLE_MLIR_PASS_OPS_MAX_POOL_SINGLE_OUT_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_MAX_POOL_SINGLE_OUT_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include "ConvertHelper.h"

namespace mlir
{
namespace Circle
{

class ConvMaxPoolSingleOut : public mlir::OpConversionPattern<mlir::ONNXMaxPoolSingleOutOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXMaxPoolSingleOutOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXMaxPoolSingleOutOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getX();
    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());

    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    CHECK_VALID_RANK_4(outtype);

    auto op_name = GetOperationName(op.getOperation());

    // TODO process op.auto_pad(), op.pads(), op.dilations(), op.ceil_mode()
    LLVM_DEBUG({ llvm::dbgs() << "MaxPoolSingleOut name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "MaxPoolSingleOut auto_pad: " << op.getAutoPad() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "MaxPoolSingleOut pads: " << op.getPads() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "MaxPoolSingleOut dilations: " << op.getDilations() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "MaxPoolSingleOut ceil_mode: " << op.getCeilMode() << "\n"; });

    if (notYetImplemented(op))
      return mlir::failure();

    mlir::ArrayAttr kernelShape = op.getKernelShape();
    mlir::ArrayAttr strides = op.getStrides().value();

    auto filter_height = GetIntValue<int32_t>(kernelShape, 0);
    auto filter_width = GetIntValue<int32_t>(kernelShape, 1);
    auto stride_h = GetIntValue<int32_t>(strides, 0);
    auto stride_w = GetIntValue<int32_t>(strides, 1);

    auto output_type = GetChnLastType(outtype);

    mlir::Value inputPreTr = input;
    // for op.pads != [0,0,0,0]
    std::vector<int32_t> padsValue;
    if (GetPads(op.getPads(), padsValue))
    {
      auto in_dtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
      // TODO support more dtypes
      if (not(in_dtype.getElementType().isF32() || in_dtype.getElementType().isF64()))
        return mlir::failure();
      mlir::RankedTensorType scalar_type =
        mlir::RankedTensorType::get({}, in_dtype.getElementType());
      mlir::Value constant_values;
      mlir::Location cv_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/constval"));
      if (in_dtype.getElementType().isF32())
        constant_values = rewriter.create<ConstOp>(
          cv_loc, DenseElementsAttr::get(scalar_type, {-std::numeric_limits<float>::max()}));
      else if (in_dtype.getElementType().isF64())
        constant_values = rewriter.create<ConstOp>(
          cv_loc, DenseElementsAttr::get(scalar_type, {-std::numeric_limits<double>::max()}));

      mlir::Type i32 = rewriter.getI32Type();
      mlir::RankedTensorType ptype = RankedTensorType::get({4, 2}, i32);
      llvm::SmallVector<int32_t, 8> pvalue = {
        0, 0, 0, 0, padsValue[0], padsValue[2], padsValue[1], padsValue[3]};
      mlir::Location pv_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/padsval"));
      mlir::Value paddings =
        rewriter.create<ConstOp>(pv_loc, DenseIntElementsAttr::get(ptype, pvalue));

      // calc output type+shape of Pad
      auto shape = intype.getShape();
      assert(shape.size() == 4);
      auto padH = shape[2] + padsValue[0] + padsValue[2];
      auto padW = shape[3] + padsValue[1] + padsValue[3];
      auto padShape = {shape[0], shape[1], padH, padW}; // order is NCHW
      LLVM_DEBUG(
        { llvm::dbgs() << "ConvMaxPoolSingleOut padH: " << padH << ", padW: " << padW << "\n"; });
      auto padType = mlir::RankedTensorType::get(padShape, outtype.getElementType());

      // change pre Transpose input to this new Pad
      mlir::Location pads_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/pads"));
      inputPreTr = rewriter.create<PadV2Op>(pads_loc, padType, input, paddings, constant_values);
    }

    mlir::Value pre_tran = CreatePreTranspose(rewriter, inputPreTr, op_name);
    mlir::Value maxpool2d = rewriter.create<MaxPool2DOp>(opLoc, output_type, pre_tran,
                                                         /*padding*/ "VALID", stride_w, stride_h,
                                                         filter_width, filter_height,
                                                         /*fused_activation_function*/ "NONE");
    ReplaceOpWithPostTranspose(rewriter, op, maxpool2d, op.getType(), op_name);

    return mlir::success();
  }

private:
  bool notYetImplemented(mlir::ONNXMaxPoolSingleOutOp &op) const
  {
    // TODO support other auto_pad: 'SAME_UPPER', 'VALID', 'SAME_LOWER'
    if (!op.getAutoPad().equals_insensitive("NOTSET"))
      return true;

    // TODO support dilations other than [1,1]
    auto dilations = op.getDilations();
    if (dilations.has_value())
    {
      auto value = dilations.value();
      for (int i = 0; i < value.size(); ++i)
        if (GetIntValue<int64_t>(value, i) != 1L)
          return true;
    }

    if (op.getCeilMode() != 0)
      return true;

    mlir::ArrayAttr kernelShape = op.getKernelShape();
    if (kernelShape.size() != 2)
      return true;

    mlir::ArrayAttr strides = op.getStrides().value();
    if (strides.size() != 2)
      return true;

    return false;
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_MAX_POOL_SINGLE_OUT_OP_H__
