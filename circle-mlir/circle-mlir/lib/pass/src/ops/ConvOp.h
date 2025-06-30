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

#ifndef __CIRCLE_MLIR_PASS_OPS_CONV_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_CONV_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <vector>
#include <limits>

namespace mlir
{
namespace Circle
{

// NOTE Name ConvConv is from Convert Convolution
class ConvConv : public mlir::OpConversionPattern<mlir::ONNXConvOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXConvOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXConvOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConvOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvConv name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv auto_pad: " << op.getAutoPad() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv dilations: " << op.getDilations() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv group: " << op.getGroup() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv kernel_shape: " << op.getKernelShape() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv pads: " << op.getPads() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv strides: " << op.getStrides() << "\n"; });

    if (notYetImplemented(op))
      return mlir::failure();

    mlir::Value input = adaptor.getX();
    mlir::Value filter = adaptor.getW();
    mlir::Value bias = adaptor.getB();
    bool biasNone = mlir::isa<mlir::NoneType>(bias.getType());

    mlir::Location opLoc = op->getLoc();

    // TODO support other ranks for I/O

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv intype: " << intype << "\n"; });
    if (intype.getRank() != 4)
      assert(false);
    CHECK_VALID_RANK_4(intype);

    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv outtype: " << outtype << "\n"; });
    if (outtype.getRank() != 4)
      assert(false);
    CHECK_VALID_RANK_4(outtype);

    mlir::Value inputPreTr = input;
    // for op.pads != [0,0,0,0]
    std::vector<int32_t> padsValue;
    if (GetPads(op.getPads(), padsValue))
      inputPreTr = insertPad(rewriter, op_name, input, outtype, padsValue);

    int32_t stride_h = 1;
    int32_t stride_w = 1;
    if (!getStrides(op, stride_h, stride_w))
      return mlir::failure();

    int64_t dilation_h_factor = 1;
    int64_t dilation_w_factor = 1;
    if (!getDilations(op, dilation_h_factor, dilation_w_factor))
      return mlir::failure();

    // NOTE luci-interpreter fails to execute when bias is none.
    // we can (1) fix luci-interpreter (2) update bias to have zero values.
    // onnx-tensorflow works like (2) so we follow this.
    if (biasNone)
      bias = getZeroBias(rewriter, op_name, filter);

    auto filter_name = GetOperationName(filter.getDefiningOp());
    if (filter_name.empty())
      filter_name = op_name + "/filter";

    mlir::Value pre_tran = CreatePreTranspose(rewriter, inputPreTr, op_name);

    auto group = op.getGroup();
    if (group == 1)
    {
      mlir::Value filter_tran = CreatePreTranspose(rewriter, filter, filter_name);

      auto conv_output_type = GetChnLastType(outtype);
      // TODO support activation != NONE
      // TODO support padding != VALID
      mlir::Value conv2d = rewriter.create<Conv2DOp>(opLoc, conv_output_type, pre_tran, filter_tran,
                                                     bias, dilation_h_factor, dilation_w_factor,
                                                     /*fused_activation_function=*/"NONE",
                                                     /*padding=*/"VALID", stride_h, stride_w);

      ReplaceOpWithPostTranspose(rewriter, op, conv2d, op.getType(), op_name);
    }
    else if (group > 1)
    {
      auto inshape = intype.getShape();
      auto outshape = outtype.getShape();
      int32_t ic = inshape[1];
      int32_t oc = outshape[1];

      // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
      // - groups controls the connections between inputs and outputs.
      // - in_channels and out_channels must both be divisible by groups.
      int32_t depth_multiplier = ic / group;
      if (ic > 0)
      {
        // check only for known shape
        if (depth_multiplier * group != ic)
          return mlir::failure();
      }
      depth_multiplier = oc / group;
      LLVM_DEBUG({ llvm::dbgs() << "depth_multiplier: " << depth_multiplier << "\n"; });
      if (depth_multiplier * group != oc)
        return mlir::failure();

      // ONNX kernel is (I O H W) --> convert to Circle (O H W I)
      llvm::SmallVector<int32_t, 4> ker_perm{1, 2, 3, 0};
      mlir::Value filter_tran = CreateTranspose(rewriter, filter, ker_perm, filter_name);

      auto dwconv_output_type = GetChnLastType(outtype);
      mlir::Value dwconv2d = rewriter.create<DepthwiseConv2DOp>(
        opLoc, dwconv_output_type, pre_tran, filter_tran, bias, dilation_h_factor,
        dilation_w_factor,
        /*fused_activation_function=*/"NONE",
        /*padding=*/"VALID", stride_h, stride_w, depth_multiplier);

      ReplaceOpWithPostTranspose(rewriter, op, dwconv2d, op.getType(), op_name);
    }
    else
      return mlir::failure();

    return mlir::success();
  }

private:
  bool notYetImplemented(mlir::ONNXConvOp &op) const
  {
    // TODO support other auto_pad: 'SAME_UPPER', 'VALID', 'SAME_LOWER'
    if (!op.getAutoPad().equals_insensitive("NOTSET"))
      return true;

    return false;
  }

  mlir::Value insertPad(mlir::ConversionPatternRewriter &rewriter, const std::string &op_name,
                        const mlir::Value &input, const mlir::RankedTensorType &outtype,
                        const std::vector<int32_t> &padsValue) const
  {
    mlir::Type i32 = rewriter.getI32Type();
    mlir::RankedTensorType ptype = RankedTensorType::get({4, 2}, i32);
    llvm::SmallVector<int32_t, 8> pvalue = {
      0, 0, 0, 0, padsValue[0], padsValue[2], padsValue[1], padsValue[3]};
    mlir::Location padsval_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/padsval"));
    mlir::Value paddings =
      rewriter.create<ConstOp>(padsval_loc, DenseIntElementsAttr::get(ptype, pvalue));

    // calc output type+shape of Pad
    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    auto shape = intype.getShape();
    assert(shape.size() == 4);
    int64_t padH = 0, padW = 0;
    // NOTE if input is unknown, set padH, padW as unknown.
    // these will be resolved in shape inference.
    auto int64_min = std::numeric_limits<int64_t>::min();
    padH = (shape[2] == int64_min ? shape[2] : shape[2] + padsValue[0] + padsValue[2]);
    padW = (shape[3] == int64_min ? shape[3] : shape[3] + padsValue[1] + padsValue[3]);
    auto padShape = {shape[0], shape[1], padH, padW}; // order is NCHW
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv padH: " << padH << ", padW: " << padW << "\n"; });
    auto padType = mlir::RankedTensorType::get(padShape, outtype.getElementType());

    // change pre Transpose input to this new Pad
    mlir::Location pads_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/pads"));
    LLVM_DEBUG({ llvm::dbgs() << "ConvConv Pad: " << pads_loc << "\n"; });
    return rewriter.create<PadOp>(pads_loc, padType, input, paddings);
  }

  bool getStrides(mlir::ONNXConvOp op, int32_t &stride_h, int32_t &stride_w) const
  {
    auto strides = op.getStrides();
    if (strides.has_value())
    {
      auto value = strides.value();
      if (value.size() != 2)
        return false;

      stride_h = GetIntValue<int32_t>(value, 0);
      stride_w = GetIntValue<int32_t>(value, 1);
    }
    return true;
  }

  bool getDilations(mlir::ONNXConvOp op, int64_t &dilation_h, int64_t &dilation_w) const
  {
    auto dilations = op.getDilations();
    if (dilations.has_value())
    {
      auto value = dilations.value();
      if (value.size() != 2)
        return false;

      dilation_h = GetIntValue<int64_t>(value, 0);
      dilation_w = GetIntValue<int64_t>(value, 1);
    }
    return true;
  }

  mlir::Value getZeroBias(mlir::ConversionPatternRewriter &rewriter, const std::string &op_name,
                          mlir::Value &filter) const
  {
    auto ftype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(filter.getType());
    assert(ftype.getElementType().isF32());
    auto shape = ftype.getShape();
    int32_t num = shape[0]; // dim 0 from OIHW
    mlir::RankedTensorType type = RankedTensorType::get({num}, ftype.getElementType());
    std::vector<float> val;
    for (int32_t c = 0; c < num; ++c)
      val.push_back(0.0f);
    mlir::Location nobias_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/nobias"));
    return rewriter.create<ConstOp>(nobias_loc, DenseFPElementsAttr::get(type, val));
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CONV_OP_H__
