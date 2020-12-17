/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/ops/AvgPool2DOp.h"

namespace mir
{
namespace ops
{

void AvgPool2DOp::inferOutputTypes()
{
  const auto &input_shape = getInputShape(0);
  const int batch_dim_index = getDataBatchDimIndex(_attributes.data_format);
  const int channel_dim_index = getDataChannelDimIndex(_attributes.data_format);

  constexpr int num_spatial_dims = 2;

  assert(input_shape.rank() == 4);
  assert(_attributes.window.size() == num_spatial_dims);
  assert(_attributes.strides.size() == num_spatial_dims);
  assert(_attributes.padding_before.size() == num_spatial_dims);
  assert(_attributes.padding_after.size() == num_spatial_dims);

  Shape output_shape(4);

  output_shape.dim(batch_dim_index) = input_shape.dim(batch_dim_index);
  output_shape.dim(channel_dim_index) = input_shape.dim(channel_dim_index);

  for (int i = 0; i < num_spatial_dims; i++)
  {
    const int spatial_dim_index = getDataSpatialDimIndex(_attributes.data_format, i);
    const std::int32_t padded_input = input_shape.dim(spatial_dim_index) +
                                      _attributes.padding_before.at(i) +
                                      _attributes.padding_after.at(i);
    // out_size = ceil((in_size - window_size + 1) / stride) =
    //   (in_size - window_size + 1 + stride - 1) / stride =
    //   (in_size - window_size) / stride + 1
    output_shape.dim(spatial_dim_index) =
      (padded_input - _attributes.window[i]) / _attributes.strides[i] + 1;
  }

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
