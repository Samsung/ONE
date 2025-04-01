/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019-2020 The IBM Research Authors.
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

#ifndef __CIRCLE_MLIR_PASS_OPS_CONV_TRANSPOSE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_CONV_TRANSPOSE_OP_H__

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

// NOTE Name ConvConvTranspose is from Convert Transpose Convolution
class ConvConvTranspose : public mlir::OpConversionPattern<mlir::ONNXConvTransposeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXConvTransposeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXConvTransposeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConvTransposeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose auto_pad: " << op.getAutoPad() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose dilations: " << op.getDilations() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose group: " << op.getGroup() << "\n"; });
    LLVM_DEBUG(
      { llvm::dbgs() << "ConvConvTranspose kernel_shape: " << op.getKernelShape() << "\n"; });
    LLVM_DEBUG(
      { llvm::dbgs() << "ConvConvTranspose output_padding: " << op.getOutputPadding() << "\n"; });
    LLVM_DEBUG(
      { llvm::dbgs() << "ConvConvTranspose output_shape: " << op.getOutputShape() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose pads: " << op.getPads() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose strides: " << op.getStrides() << "\n"; });

    if (notYetImplemented(op))
      return mlir::failure();

    mlir::Value input = adaptor.getX();
    mlir::Value filter = adaptor.getW();
    mlir::Value bias = adaptor.getB();
    // NOTE bias.getType().isa<NoneType>() is true if there is no bias

    auto filter_name = GetOperationName(filter.getDefiningOp());
    if (filter_name.empty())
      filter_name = op_name + "/filter";

    mlir::Location opLoc = op->getLoc();

    // TODO support other ranks for I/O

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose intype: " << intype << "\n"; });
    CHECK_VALID_RANK_4(intype);

    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose outtype: " << outtype << "\n"; });
    CHECK_VALID_RANK_4(outtype);

    mlir::Value pre_tran = CreatePreTranspose(rewriter, input, op_name);
    // ONNX kernel is (I O H W) --> convert to Circle (O H W I)
    llvm::SmallVector<int32_t, 4> ker_perm{1, 2, 3, 0};
    mlir::Value filter_tran = CreateTranspose(rewriter, filter, ker_perm, filter_name);

    // input shape
    auto inshape = intype.getShape();
    // filter shape
    mlir::RankedTensorType filtertype = filter.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    auto filtershape = filtertype.getShape(); // IOHW

    int32_t stride_h = 1;
    int32_t stride_w = 1;
    auto strides = op.getStrides();
    if (strides.has_value())
    {
      auto value = strides.value();
      if (value.size() != 2)
        return mlir::failure();

      stride_h = GetIntValue<int32_t>(value, 0);
      stride_w = GetIntValue<int32_t>(value, 1);
    }
    // TODO support dilation
    int32_t dilation_h = 1;
    int32_t dilation_w = 1;

    int32_t output_padding_h = 0;
    int32_t output_padding_w = 0;
    auto output_padding = op.getOutputPadding();
    if (output_padding.has_value())
    {
      auto value = output_padding.value();
      if (value.size() != 2)
        return mlir::failure();
      output_padding_h = GetIntValue<int32_t>(value, 0);
      output_padding_w = GetIntValue<int32_t>(value, 1);
    }

    // create output_shape constant
    mlir::Value output_shape;
    mlir::SmallVector<int32_t, 4> os_i32;
    {
      int32_t hin = static_cast<int32_t>(inshape[2]);
      int32_t win = static_cast<int32_t>(inshape[3]);
      int32_t hfs = static_cast<int32_t>(filtershape[2]);
      int32_t wfs = static_cast<int32_t>(filtershape[3]);
      int32_t hout = (hin - 1) * stride_h + dilation_h * (hfs - 1) + output_padding_h + 1;
      int32_t wout = (win - 1) * stride_w + dilation_w * (wfs - 1) + output_padding_w + 1;
      int32_t nin = static_cast<int32_t>(inshape[0]);
      int32_t ofs = static_cast<int32_t>(filtershape[1]);
      os_i32.push_back(nin);
      os_i32.push_back(hout);
      os_i32.push_back(wout);
      os_i32.push_back(ofs); // from IOHW

      mlir::Location shape_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/shape"));
      mlir::Type i32 = rewriter.getI32Type();
      mlir::RankedTensorType ostype = RankedTensorType::get({4}, i32);
      output_shape = rewriter.create<ConstOp>(shape_loc, DenseIntElementsAttr::get(ostype, os_i32));
    }

    mlir::SmallVector<int64_t> trconv2d_shape({os_i32[0], os_i32[1], os_i32[2], os_i32[3]});
    auto trconv_output_type = mlir::RankedTensorType::get(trconv2d_shape, outtype.getElementType());
    mlir::Value trconv2d = rewriter.create<TransposeConvOp>(
      opLoc, trconv_output_type, output_shape, filter_tran, pre_tran, bias,
      /*padding=*/"VALID", stride_h, stride_w);

    // if padding exist, insert `Slice`
    mlir::Value inPostTrConv = trconv2d;
    std::vector<int32_t> padsValue;
    if (GetPads(op.getPads(), padsValue))
    {
      mlir::Type i32 = rewriter.getI32Type();
      mlir::RankedTensorType bstype = RankedTensorType::get({4}, i32);

      mlir::Location sb_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/slice/begin"));
      mlir::SmallVector<int32_t, 4> begin_i32;
      begin_i32.push_back(0);
      begin_i32.push_back(padsValue[0]);
      begin_i32.push_back(padsValue[1]);
      begin_i32.push_back(0);
      auto beginConst =
        rewriter.create<ConstOp>(sb_loc, DenseIntElementsAttr::get(bstype, begin_i32));

      mlir::Location ss_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/slice/size"));
      mlir::SmallVector<int32_t, 4> size_i32;
      size_i32.push_back(os_i32[0]);
      size_i32.push_back(os_i32[1] - 2 * padsValue[0]);
      size_i32.push_back(os_i32[2] - 2 * padsValue[1]);
      size_i32.push_back(os_i32[3]);
      auto sizeConst =
        rewriter.create<ConstOp>(ss_loc, DenseIntElementsAttr::get(bstype, size_i32));

      mlir::Location s_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/slice"));
      mlir::SmallVector<int64_t, 4> slice_shape;
      for (int i = 0; i < 4; ++i)
        slice_shape.push_back(static_cast<int64_t>(size_i32[i]));
      auto stype = mlir::RankedTensorType::get(slice_shape, outtype.getElementType());
      inPostTrConv = rewriter.create<SliceOp>(s_loc, stype, trconv2d, beginConst, sizeConst);
    }

    // TODO insert bias as ConstOp

    LLVM_DEBUG({ llvm::dbgs() << "ConvConvTranspose PostTr: " << op.getType() << "\n"; });
    ReplaceOpWithPostTranspose(rewriter, op, inPostTrConv, op.getType(), op_name);

    return mlir::success();
  }

private:
  bool notYetImplemented(mlir::ONNXConvTransposeOp &op) const
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

    // TODO support group other than 1
    auto group = op.getGroup();
    if (group != 1)
      return true;

    return false;
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CONV_TRANSPOSE_OP_H__
