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

#include "ConvPoolHelpers.h"

#include <algorithm>
#include <cassert>

namespace mir_onnx
{

void inferAutoPadding(const std::string &pad_type, const mir::Shape &input_shape,
                      const std::vector<std::int32_t> &dilations,
                      const std::vector<std::int32_t> &strides,
                      const std::vector<std::int32_t> &window_size,
                      std::vector<std::int32_t> &padding_before,
                      std::vector<std::int32_t> &padding_after)
{
  constexpr int num_spatial_dims = 2;

  if (pad_type == "NOTSET")
  {
    // Do nothing.
  }
  else if (pad_type == "VALID")
  {
    padding_before.assign(num_spatial_dims, 0);
    padding_after.assign(num_spatial_dims, 0);
  }
  else
  {
    padding_before.resize(num_spatial_dims);
    padding_after.resize(num_spatial_dims);

    assert(dilations.size() == num_spatial_dims);
    assert(strides.size() == num_spatial_dims);
    assert(window_size.size() == num_spatial_dims);

    for (int i = 0; i < num_spatial_dims; ++i)
    {
      const std::int32_t eff_window_size = (window_size[i] - 1) * dilations[i] + 1;
      // Assuming input has NCHW format.
      const std::int32_t residual = input_shape.dim(2 + i) % strides[i];
      const std::int32_t total_pad = std::max(
        INT32_C(0), residual == 0 ? eff_window_size - strides[i] : eff_window_size - residual);
      if (pad_type == "SAME_UPPER")
      {
        padding_before[i] = total_pad / 2;
        padding_after[i] = (total_pad + 1) / 2;
      }
      else
      {
        assert(pad_type == "SAME_LOWER");
        padding_before[i] = (total_pad + 1) / 2;
        padding_after[i] = total_pad / 2;
      }
    }
  }
}

std::vector<std::int32_t> fixPads(const mir::Shape &input_shape,
                                  const std::vector<std::int32_t> &pads,
                                  const std::vector<std::int32_t> &strides,
                                  const std::vector<std::int32_t> &dilation,
                                  const std::vector<std::int32_t> &kernel_shape)
{
  assert(pads.size() % 2 == 0);
  int spatial_dimensions = pads.size() / 2;
  std::vector<std::int32_t> fixed_pads(pads);
  for (int i = 0; i < spatial_dimensions; ++i)
  {
    auto effective_window_dim = (kernel_shape[i] - 1) * dilation[i] + 1;
    auto effective_input_dim = input_shape.dim(i + 2) + pads[i] + pads[i + spatial_dimensions];
    // Computing number of "redundant" elements at the end of input dimension
    // for example we have effective_input_dim == 8, effective_window)dim == 3 and stride == 2:
    // [1][2][3][4][5][6][7][8]  - input
    //  *  *  *  .  .  .  .      - first kernel application
    //  .  .  *  *  *  .  .      - second kernel application
    //  .  .  .  .  *  *  *      - third kernel application
    // element 8 is unused (remainder should be 1)
    //
    // glossary:
    // i - effective input size
    // w - effective window size
    // s - stride
    // n - number of kernel applications (3 in example)
    //
    // i = s * (n-1) + w + r
    // r = i - w - s * (n-1)
    // n - is the maximum number of windows we can fit into input, so this formula is equal to
    // r = (i - w) % s
    auto remainder = (effective_input_dim - effective_window_dim) % strides[i];

    // remove redundant pad, but no more than there are padding
    fixed_pads[i + spatial_dimensions] -= std::min(remainder, pads[i + spatial_dimensions]);
  }
  return fixed_pads;
}

} // namespace mir_onnx
