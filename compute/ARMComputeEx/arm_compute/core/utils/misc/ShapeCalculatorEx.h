/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2016-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __ARM_COMPUTE_MISC_SHAPE_CALCULATOR_EX_H__
#define __ARM_COMPUTE_MISC_SHAPE_CALCULATOR_EX_H__

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "arm_compute/core/utils/helpers/tensor_transform.h"

#include <cmath>

namespace arm_compute
{
namespace misc
{
namespace shape_calculator
{

/** Calculate the upsampled output shape used for transpose convolution
 *
 * @param[in] input              Input tensor info
 * @param[in] weights            Weights tensor shape
 * @param[in] info               Padding and stride info
 * @param[in] out_dims           Output shape dimensions
 * @param[in] invalid_right      The number of zeros added to right edge of the output.
 * @param[in] invalid_bottom     The number of zeros added to bottom edge of the output.
 * @param[out] pad_left          Padding on left
 * @param[out] pad_right         Padding on right
 * @param[out] pad_top           Padding on top
 * @param[out] pad_bottom        Padding on bottom
 *
 * @return the calculated shape
 */
inline TensorShape compute_transposeconv_upsampled_shape(
    const ITensorInfo &input, const ITensorInfo &weights, const PadStrideInfo &info,
    std::pair<unsigned int, unsigned int> &out_dims, unsigned int invalid_right,
    unsigned int invalid_bottom, unsigned int &pad_left, unsigned int &pad_right,
    unsigned int &pad_top, unsigned int &pad_bottom)
{
  unsigned int sx = info.stride().first;
  unsigned int sy = info.stride().second;
  const DataLayout data_layout = input.data_layout();
  const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

  // Find the upsampled dimensions
  // transpose conv out:
  //    tconv_out + pad = 1 + (in - 1) * stride + invalid
  //    tconv_out = 1 + (in - 1) * stride + invalid - pad
  // upsample out:
  //    upsample_out = 1 + (in - 1) * stride
  unsigned int out_x = (input.dimension(idx_w) - 1) * sx + 1;
  unsigned int out_y = (input.dimension(idx_h) - 1) * sy + 1;

  // Find the padding needed for the convolution with stride 1 in order to match output shape
  // upsample+pad out:
  //    upsample_out + pad = tconv_out + kernel - 1
  //    pad = tconv_out + kernel - 1 - upsample_out
  unsigned int padx = out_dims.first - (out_x - weights.dimension(idx_w) + 1);
  unsigned int pady = out_dims.second - (out_y - weights.dimension(idx_h) + 1);
  out_x += padx;
  out_y += pady;

  unsigned int padx_all_except_invallid = padx + info.pad_left() + info.pad_right() - invalid_right;
  unsigned int pady_all_except_invallid =
      pady + info.pad_top() + info.pad_bottom() - invalid_bottom;
  pad_left = (padx_all_except_invallid + 1) / 2 - info.pad_left();
  pad_right = pady_all_except_invallid / 2 - info.pad_right() + invalid_right;
  pad_top = (padx_all_except_invallid + 1) / 2 - info.pad_top();
  pad_bottom = pady_all_except_invallid / 2 - info.pad_bottom() + invalid_bottom;

  TensorShape scale_out_shape(input.tensor_shape());
  scale_out_shape.set(idx_w, out_x);
  scale_out_shape.set(idx_h, out_y);

  return scale_out_shape;
}

/** Calculate the output shape of the transpose convolution layer
 *
 * @param[in] out_dims Output x and y shape dimensions
 * @param[in] input    Input tensor info
 * @param[in] weights  Weights tensor shape
 *
 * @return the calculated shape
 */
inline TensorShape
compute_transposeconv_output_shape(const std::pair<unsigned int, unsigned int> &out_dims,
                                   const ITensorInfo &input, const ITensorInfo &weights)
{
  const TensorShape input_shape{input.tensor_shape()};
  const TensorShape weights_shape{weights.tensor_shape()};

  const DataLayout data_layout = input.data_layout();
  const int width_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const int height_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
  const int channel_idx =
      get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
  const int batch_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

  TensorShape out_shape{input_shape};
  out_shape.set(width_idx, out_dims.first);
  out_shape.set(height_idx, out_dims.second);
  out_shape.set(channel_idx, weights_shape[batch_idx]);
  return out_shape;
}

/** Calculate the depth to space output shape of a tensor
 *
 * @param[in] input Input tensor info
 * @param[in] block Block shape value
 *
 * @return the calculated shape
 */
inline TensorShape compute_depth_to_space_shape_ex(const ITensorInfo *input, int block)
{
  ARM_COMPUTE_ERROR_ON(block < 2);

  const DataLayout data_layout = input->data_layout();
  const int idx_width = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
  const int idx_channel =
      get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

  TensorShape output_shape{input->tensor_shape()};
  output_shape.set(idx_width, input->dimension(idx_width) * block);
  output_shape.set(idx_height, input->dimension(idx_height) * block);
  output_shape.set(idx_channel, input->dimension(idx_channel) / (block * block));

  return output_shape;
}

/** Calculate the space to batch output shape of a tensor
 *
 * @param[in] input       Input tensor info
 * @param[in] block_shape Block shape value
 *
 * @return the calculated shape
 */
inline TensorShape compute_space_to_depth_shape_ex(const ITensorInfo *input, int32_t block_shape)
{
  ARM_COMPUTE_ERROR_ON(block_shape < 2);
  TensorShape output_shape{input->tensor_shape()};

  const DataLayout data_layout = input->data_layout();
  const int idx_width = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
  const int idx_depth = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

  output_shape.set(idx_width, input->tensor_shape()[idx_width] * block_shape);
  output_shape.set(idx_height, input->tensor_shape()[idx_height] * block_shape);
  output_shape.set(idx_depth, input->tensor_shape()[idx_depth] / (block_shape * block_shape));

  return output_shape;
}

/** Calculate the gather output shape of a tensor
 *
 * @param[in] input_shape   Input tensor shape
 * @param[in] indices_shape Indices tensor shape
 * @param[in] actual_axis   The axis to be gathered
 *
 * @return the calculated shape
 */
inline TensorShape compute_gather_shape_ex(const TensorShape &input_shape,
                                           const TensorShape &indices_shape, uint32_t actual_axis)
{
  ARM_COMPUTE_ERROR_ON(indices_shape.num_dimensions() > 3);
  ARM_COMPUTE_ERROR_ON(input_shape.num_dimensions() > 4);
  ARM_COMPUTE_ERROR_ON(input_shape.num_dimensions() + indices_shape.num_dimensions() - 1 > 4);
  ARM_COMPUTE_ERROR_ON(actual_axis >= input_shape.num_dimensions());

  TensorShape output_shape = input_shape;
  if (indices_shape.num_dimensions() == 1)
  {
    output_shape[actual_axis] = indices_shape[0];
  }
  else if (indices_shape.num_dimensions() > 1)
  {
    output_shape.shift_right(indices_shape.num_dimensions() - 1);

    for (uint32_t i = 0, o = 0; o < output_shape.num_dimensions(); ++o, ++i)
    {
      if (o == actual_axis)
      {
        ++i;
        for (uint32_t in = 0; in < indices_shape.num_dimensions(); ++in, ++o)
        {
          output_shape[o] = indices_shape[in];
        }
      }
      else
      {
        output_shape[o] = input_shape[i];
      }
    }
  }
  return output_shape;
}

/** Calculate the gather output shape of a tensor
 *
 * @param[in] input_shape   Input tensor shape
 * @param[in] indices_shape Indices tensor shape
 * @param[in] actual_axis   The axis to be gathered
 *
 * @return the calculated shape
 */
inline TensorShape compute_onehot_shape_ex(const TensorShape &indices_shape, uint32_t depth,
                                           uint32_t actual_axis)
{
  ARM_COMPUTE_ERROR_ON(indices_shape.num_dimensions() > 3);
  ARM_COMPUTE_ERROR_ON(actual_axis > indices_shape.num_dimensions());

  TensorShape output_shape;
  output_shape.set(actual_axis, depth);

  unsigned int i_shift = 0;
  for (unsigned int i = 0; i < indices_shape.num_dimensions(); ++i)
  {
    if (i == actual_axis)
    {
      i_shift++;
    }
    output_shape.set(i + i_shift, indices_shape[i]);
  }

  return output_shape;
}

} // namespace shape_calculator
} // namespace misc
} // namespace arm_compute

#endif // __ARM_COMPUTE_MISC_SHAPE_CALCULATOR_EX_H__
