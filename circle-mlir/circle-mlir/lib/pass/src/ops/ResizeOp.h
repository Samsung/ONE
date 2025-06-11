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

#ifndef __CIRCLE_MLIR_PASS_OPS_RESIZE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_RESIZE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace mlir
{
namespace Circle
{

class ConvResize : public mlir::OpConversionPattern<mlir::ONNXResizeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXResizeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXResizeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXResizeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getX();
    mlir::Value roi = adaptor.getRoi();
    mlir::Value scales = adaptor.getScales();
    mlir::Value sizes = adaptor.getSizes();
    llvm::StringRef coordinate_transformation_mode = adaptor.getCoordinateTransformationMode();
    llvm::APFloat cubic_coeff_a = adaptor.getCubicCoeffA();
    int64_t exclude_outside = adaptor.getExcludeOutside();
    llvm::APFloat extrapolation_value = adaptor.getExtrapolationValue();
    llvm::StringRef mode = adaptor.getMode();
    llvm::StringRef nearest_mode = adaptor.getNearestMode();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvResize name: " << op_name << "\n"; });

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvResize intype: " << intype << "\n"; });
    // TODO support other ranks
    CHECK_VALID_RANK_4(intype);

    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvResize outtype: " << outtype << "\n"; });
    CHECK_VALID_RANK_4(outtype);

    mlir::RankedTensorType sizestype =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(sizes.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvResize sizestype: " << sizestype << "\n"; });

    if (notYetImplemented(op, adaptor))
      return mlir::failure();

    // TODO support roi (roi can be None or <0xf32>)
    (void)roi;
    // TODO support cubic_coeff_a, exclude_outside, extrapolation_value
    (void)cubic_coeff_a;
    (void)exclude_outside;
    (void)extrapolation_value;
    // TODO support nearest_mode;
    (void)nearest_mode;

    // Let default size input be sizes
    mlir::Value size = sizes;
    // if output shape is normal, we can use it
    auto outshape = outtype.getShape();
    if (outshape[2] > 0 && outshape[3] > 0)
    {
      std::vector<int32_t> sizeValue = {static_cast<int32_t>(outshape[2]),
                                        static_cast<int32_t>(outshape[3])};
      size = CreateI32Const(rewriter, sizeValue, op_name + "/size");
    }

    mlir::BoolAttr align_corners;
    mlir::BoolAttr half_pixel_centers;
    align_corners = rewriter.getBoolAttr(false);
    half_pixel_centers = rewriter.getBoolAttr(false);
    if (coordinate_transformation_mode == "align_corners")
      align_corners = rewriter.getBoolAttr(true);
    if (coordinate_transformation_mode == "half_pixel")
      half_pixel_centers = rewriter.getBoolAttr(true);
    if (coordinate_transformation_mode == "asymmetric")
    {
      // for this case, constant size from outshape is dropped.
      // usually, outshape is unknown.
      std::vector<float> scales_values;
      if (ExtractConstantValues(scales, scales_values))
      {
        auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(intype);
        if (shaped_type.hasStaticShape())
        {
          auto inshape = intype.getShape();
          if (inshape.size() == scales_values.size())
          {
            int32_t h = static_cast<int32_t>(inshape[2] * scales_values[2]);
            int32_t w = static_cast<int32_t>(inshape[3] * scales_values[3]);
            assert(h > 0 && w > 0);
            std::vector<int32_t> sizeValue = {h, w};
            size = CreateI32Const(rewriter, sizeValue, op_name + "/size");
          }
        }
        else
        {
          // postpone conversion as we don't know the input H/W but need to preserve "scales"
          // NOTE pass align_corners and half_pixel_centers ?
          auto resize_output_type = GetChnLastType(outtype);
          mlir::Value pre_tran = CreatePreTranspose(rewriter, input, op_name);
          mlir::Value resizeop =
            rewriter.create<ResizeOnnxOp>(opLoc, resize_output_type, pre_tran, scales, mode);
          ReplaceOpWithPostTranspose(rewriter, op, resizeop, op.getType(), op_name);
          return mlir::success();
        }
      }
    }

    auto resize_output_type = GetChnLastType(outtype);

    mlir::Value pre_tran = CreatePreTranspose(rewriter, input, op_name);

    mlir::Value resize;
    if (mode == "nearest")
    {
      resize = rewriter.create<ResizeNearestNeighborOp>(opLoc, resize_output_type, pre_tran, size,
                                                        align_corners, half_pixel_centers);
    }
    else if (mode == "linear")
    {
      resize = rewriter.create<ResizeBilinearOp>(opLoc, resize_output_type, pre_tran, size,
                                                 align_corners, half_pixel_centers);
    }
    else
      return mlir::failure();

    ReplaceOpWithPostTranspose(rewriter, op, resize, op.getType(), op_name);

    return mlir::success();
  }

private:
  bool notYetImplemented(mlir::ONNXResizeOp &op, OpAdaptor &adaptor) const
  {
    if (adaptor.getMode() != "nearest" && adaptor.getMode() != "linear")
      return true;
    if (adaptor.getNearestMode() != "round_prefer_floor" && adaptor.getNearestMode() != "floor")
      return true;

    return false;
  }
};

class ConvResizeV13 : public mlir::OpConversionPattern<mlir::ONNXResizeV13Op>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXResizeV13Op>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXResizeV13Op::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXResizeV13Op op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    throw std::runtime_error("NYI ConvResizeV13");
    // TODO implement when necessary
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_RESIZE_OP_H__
