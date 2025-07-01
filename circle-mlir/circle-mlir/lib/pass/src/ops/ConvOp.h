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

private:
  struct ConvAttrs
  {
    int32_t stride_h;
    int32_t stride_w;
    int64_t dilation_h;
    int64_t dilation_w;
  };

public:
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

      auto filtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(filter.getType());
      bool depthwise = (inshape.size() == 4 && filtype.getShape().size() == 4) && (group == ic);
      if (depthwise)
      {
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
      {
        // Convert to network as
        //     input - Transpose - Split - [Conv] - Concat - Add - Transpose - output
        //    filter - Transpose - Split /
        // Use from above
        //    pre_tran: input - (Pad) - Transpose
        ConvAttrs convAttrs{stride_h, stride_w, dilation_h_factor, dilation_w_factor};
        mlir::Value add_op;
        // splitConv() will return new Add in add_op
        if (!splitConv(rewriter, group, pre_tran, filter, bias, intype, outtype, op_name,
                       filter_name, convAttrs, add_op))
        {
          return mlir::failure();
        }

        ReplaceOpWithPostTranspose(rewriter, op, add_op, op.getType(), op_name);

        return mlir::success();
      }
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

  bool splitConv(mlir::ConversionPatternRewriter &rewriter, int64_t group, mlir::Value &pre_tran,
                 mlir::Value &filter, mlir::Value &bias, const mlir::RankedTensorType &intype,
                 const mlir::RankedTensorType &outtype, const std::string &op_name,
                 const std::string &filter_name, const ConvAttrs &convAttrs,
                 mlir::Value &add_op) const
  {
    // This will create a small network with
    //     pre_tran - Transpose - Split - [Conv] - Concat - Add
    //       filter - Transpose - Split /            bias /
    // and return Add as add_op
    auto elementType = outtype.getElementType();
    auto inshape = intype.getShape();
    auto outshape = outtype.getShape();
    auto filtype = mlir::dyn_cast<mlir::RankedTensorType>(filter.getType());

    int64_t channel_split = 1;
    mlir::RankedTensorType insplttype;
    mlir::RankedTensorType fisplttype;
    {
      // get each split shape, which should be same as each Transpose output shape
      //    input (N C H W) -> Tr (N H W C) -> Split (N H W C/split)
      //   filter (O I H W) -> Tr (O H W I) -> Split (O/split H W I)
      channel_split = inshape[1] / group;
      if (channel_split * group != inshape[1])
      {
        // TODO support not divisable
        LLVM_DEBUG({ llvm::dbgs() << "Channel is not divisable with group\r\n"; });
        return false;
      }
      auto in_split = {inshape[0], inshape[2], inshape[3], channel_split};
      insplttype = mlir::RankedTensorType::get(in_split, elementType);

      // for filter O I H W
      auto fishape = filtype.getShape();
      // filter perm = 0, 2, 3, 1
      channel_split = fishape[0] / group;
      auto fi_split = {channel_split, fishape[2], fishape[3], fishape[1]};
      fisplttype = mlir::RankedTensorType::get(fi_split, elementType);
    }

    // prepare array of outputs of two Splits, one for input, another for filter
    llvm::SmallVector<mlir::Type, 4> split_out_types;
    llvm::SmallVector<mlir::Type, 4> split_fil_types;
    for (int64_t split = 0; split < group; ++split)
    {
      split_out_types.push_back(insplttype);
      split_fil_types.push_back(fisplttype);
    }

    // prepare SplitOp attributes
    mlir::Value split_i_dim = CreateI32Const(rewriter, 3, op_name + "/split_input_dim");
    mlir::Value split_f_dim = CreateI32Const(rewriter, 0, op_name + "/split_filter_dim");
    uint32_t num_splits = static_cast<uint32_t>(group);

    mlir::Value filter_tran = CreatePreTranspose(rewriter, filter, filter_name);
    mlir::Location spf_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/split_filter"));
    auto filter_split =
      rewriter.create<SplitOp>(spf_loc, split_fil_types, split_f_dim, filter_tran, num_splits);
    mlir::SmallVector<mlir::Value, 4> filter_splits(filter_split.getOutputs());

    mlir::Location spi_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/split_input"));
    auto input_split =
      rewriter.create<SplitOp>(spi_loc, split_out_types, split_i_dim, pre_tran, num_splits);
    mlir::SmallVector<mlir::Value, 4> input_splits(input_split.getOutputs());

    // prepare ConvOp output type
    mlir::RankedTensorType conv2d_split_type;
    {
      // get splitted Conv2d type
      channel_split = outshape[1] / group;
      if (channel_split * group != outshape[1])
      {
        // TODO support not dividable
        LLVM_DEBUG({ llvm::dbgs() << "Channel is not dividable with group\r\n"; });
        return false;
      }
      // NCHW to NHWC
      auto to_nhwc = {outshape[0], outshape[2], outshape[3], channel_split};
      conv2d_split_type = mlir::RankedTensorType::get(to_nhwc, elementType);
    }
    mlir::Value bias_split_zero;
    {
      mlir::RankedTensorType type = RankedTensorType::get({channel_split}, elementType);
      mlir::Location bias0_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/bias0"));
      bias_split_zero = CreateConst(rewriter, bias0_loc, type, 0.0f);
    }

    // prepare array of ConvOp
    llvm::SmallVector<mlir::Value, 4> ops;
    auto ch_last_outtype = GetChnLastType(outtype);
    for (int64_t split = 0; split < group; ++split)
    {
      mlir::Location conv_loc =
        mlir::NameLoc::get(rewriter.getStringAttr(op_name + "_" + std::to_string(split)));
      auto conv = rewriter.create<Conv2DOp>(
        conv_loc, conv2d_split_type, input_splits[split], filter_splits[split], bias_split_zero,
        convAttrs.dilation_h, convAttrs.dilation_w, /*fused_activation_function=*/"NONE",
        /*padding=*/"VALID", convAttrs.stride_h, convAttrs.stride_w);
      ops.push_back(conv);
    }
    mlir::ValueRange conv_ops(ops);
    mlir::Location concat_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/concat"));
    mlir::Value concat_op =
      rewriter.create<ConcatenationOp>(concat_loc, ch_last_outtype, conv_ops, -1, "NONE");

    mlir::Location add_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/bias"));
    add_op = rewriter.create<AddOp>(add_loc, ch_last_outtype, concat_op, bias, "NONE");

    return true;
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CONV_OP_H__
