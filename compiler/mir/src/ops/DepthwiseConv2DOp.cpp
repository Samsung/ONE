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

#include "mir/ops/DepthwiseConv2DOp.h"

namespace mir
{
namespace ops
{

void DepthwiseConv2DOp::inferOutputTypes()
{
  // Kernel shape: [Hk, Wk, Ci, M].
  const auto &input_shape = getInputShape(0);
  const auto &kernel_shape = getInputShape(1);
  const int batch_dim_index = getDataBatchDimIndex(_attributes.data_format);
  const int channel_dim_index = getDataChannelDimIndex(_attributes.data_format);

  assert(input_shape.rank() == 4);
  assert(kernel_shape.rank() == 4);
  assert(input_shape.dim(channel_dim_index) == kernel_shape.dim(2));
  assert(_attributes.strides.size() == 2);
  assert(_attributes.padding_before.size() == 2);
  assert(_attributes.padding_after.size() == 2);

  Shape output_shape(4);

  output_shape.dim(batch_dim_index) = input_shape.dim(batch_dim_index);
  output_shape.dim(channel_dim_index) = input_shape.dim(channel_dim_index) * kernel_shape.dim(3);

  for (int i = 0; i < 2; i++)
  {
    const int spatial_dim_index = getDataSpatialDimIndex(_attributes.data_format, i);
    const std::int32_t padded_input = input_shape.dim(spatial_dim_index) +
                                      _attributes.padding_before[i] + _attributes.padding_after[i];
    // out_size = ceil((in_size - kernel_size + 1) / stride) =
    //   (in_size - kernel_size + 1 + stride - 1) / stride =
    //   (in_size - kernel_size) / stride + 1
    output_shape.dim(spatial_dim_index) =
      (padded_input - kernel_shape.dim(i)) / _attributes.strides[i] + 1;
  }

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
