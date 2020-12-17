/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/ops/Deconv2DOp.h"

namespace mir
{
namespace ops
{

// See the formulas at https://github.com/onnx/onnx/blob/master/docs/Operators.md#convtranspose.
void DeConv2DOp::inferPaddings()
{
  assert(_attributes.padding_type != PaddingType::Explicit);

  const auto &input_shape = getInputShape(0);
  const auto &kernel_shape = getInputShape(1);
  const auto &output_shape = getOutputShape(0);

  constexpr int num_spatial_dims = 2;

  for (int i = 0; i < num_spatial_dims; ++i)
  {
    const int spatial_dim_index = getDataSpatialDimIndex(_attributes.data_format, i);
    const std::int32_t total_padding =
      (input_shape.dim(spatial_dim_index) - 1) * _attributes.strides[i] + kernel_shape.dim(i) -
      output_shape.dim(spatial_dim_index);

    switch (_attributes.padding_type)
    {
      case PaddingType::Valid:
        // TODO Figure out what to do.
        assert(false);
        break;
      case PaddingType::SameLower:
        _attributes.padding_after[i] = total_padding / 2;
        _attributes.padding_before[i] = total_padding - _attributes.padding_after[i];
        break;
      case PaddingType::SameUpper:
        _attributes.padding_before[i] = total_padding / 2;
        _attributes.padding_after[i] = total_padding - _attributes.padding_before[i];
        break;
      default:
        assert(false);
    }
  }
}

// See the formulas at https://github.com/onnx/onnx/blob/master/docs/Operators.md#convtranspose.
void DeConv2DOp::inferOutputTypes()
{
  assert(_attributes.padding_type == PaddingType::Explicit);

  // Kernel shape: [Hk, Wk, Co, Ci]
  const auto &input_shape = getInputShape(0);
  const auto &kernel_shape = getInputShape(1);
  const int batch_dim_index = getDataBatchDimIndex(_attributes.data_format);
  const int channel_dim_index = getDataChannelDimIndex(_attributes.data_format);

  assert(input_shape.rank() == 4);
  assert(kernel_shape.rank() == 4);
  assert(kernel_shape.dim(3) == input_shape.dim(channel_dim_index));

  Shape output_shape(4);

  output_shape.dim(batch_dim_index) = input_shape.dim(batch_dim_index);
  output_shape.dim(channel_dim_index) = kernel_shape.dim(2);

  constexpr int num_spatial_dims = 2;

  for (int i = 0; i < num_spatial_dims; i++)
  {
    const int spatial_dim_index = getDataSpatialDimIndex(_attributes.data_format, i);
    output_shape.dim(spatial_dim_index) =
      (input_shape.dim(spatial_dim_index) - 1) * _attributes.strides[i] + kernel_shape.dim(i) -
      (_attributes.padding_before.at(i) + _attributes.padding_after.at(i));
  }

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
