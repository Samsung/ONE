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

#include "mir/ops/Conv2DOp.h"

namespace mir
{
namespace ops
{

void Conv2DOp::inferOutputTypes()
{
  // Kernel shape: [O, H, W, I / M].
  const auto &input_shape = getInputShape(0);
  const auto &kernel_shape = getInputShape(1);
  const int batch_dim_index = getDataBatchDimIndex(_attributes.data_format);
  const int channel_dim_index = getDataChannelDimIndex(_attributes.data_format);

  constexpr int num_spatial_dims = 2;

  assert(input_shape.rank() == 2 + num_spatial_dims);
  assert(kernel_shape.rank() == 2 + num_spatial_dims);
  assert(kernel_shape.dim(3) * _attributes.num_groups == input_shape.dim(channel_dim_index));
  assert(kernel_shape.dim(0) % _attributes.num_groups == 0);

  assert(_attributes.strides.size() == num_spatial_dims);
  assert(_attributes.padding_before.size() == num_spatial_dims);
  assert(_attributes.padding_after.size() == num_spatial_dims);

  Shape output_shape(2 + num_spatial_dims);

  output_shape.dim(batch_dim_index) = input_shape.dim(batch_dim_index);
  output_shape.dim(channel_dim_index) = kernel_shape.dim(0);

  for (int i = 0; i < num_spatial_dims; i++)
  {
    const int spatial_dim_index = getDataSpatialDimIndex(_attributes.data_format, i);
    const std::int32_t padded_input = input_shape.dim(spatial_dim_index) +
                                      _attributes.padding_before[i] + _attributes.padding_after[i];
    // out_size = ceil((in_size - kernel_size + 1) / stride) =
    //   (in_size - kernel_size + 1 + stride - 1) / stride =
    //   (in_size - kernel_size) / stride + 1
    output_shape.dim(spatial_dim_index) =
      (padded_input - kernel_shape.dim(1 + i)) / _attributes.strides[i] + 1;
  }

  auto dt = getInput(0)->getElementType();
  assert(dt == getInput(1)->getElementType() && "kernel should have same data type as input");

  setOutputType(0, {dt, output_shape});
}

} // namespace ops
} // namespace mir
