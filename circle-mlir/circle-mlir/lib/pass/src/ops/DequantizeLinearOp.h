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

#ifndef __CIRCLE_MLIR_PASS_OPS_DEQUANTIZE_LINEAR_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_DEQUANTIZE_LINEAR_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <flatbuffers/flexbuffers.h>

namespace mlir
{
namespace Circle
{

class ConvDequantizeLinear : public mlir::OpConversionPattern<mlir::ONNXDequantizeLinearOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXDequantizeLinearOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXDequantizeLinearOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXDequantizeLinearOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value XZeroPoint = adaptor.getXZeroPoint();
    int64_t Axis = adaptor.getAxis();

    // if XZeroPoint (or XScale) is scalar, no need to add axis
    auto zp_dtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(XZeroPoint.getType());

    llvm::StringRef op_name = "ONNXDequantizeLinear";

    auto flex_buffers = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_buffers->StartMap();
    flex_buffers->Int("T", circle::TensorType_FLOAT32);
    if (zp_dtype.getRank())
      flex_buffers->Int("axis", Axis);
    flex_buffers->EndMap(map_start);
    flex_buffers->Finish();

    // flexbuffer::GetBuffer() return vector<uint8_t> but StringRef requires char*
    const char *buffer = reinterpret_cast<const char *>(flex_buffers->GetBuffer().data());
    auto strbuffer = mlir::StringRef(buffer, flex_buffers->GetSize());
    auto attr = ConstBytesAttr::get(rewriter.getContext(), strbuffer);

    rewriter.replaceOpWithNewOp<CustomOp>(op, op->getResultTypes(), op->getOperands(), op_name,
                                          attr);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_DEQUANTIZE_LINEAR_OP_H__
