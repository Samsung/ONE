/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_PAD_H__
#define __NNFW_CKER_TRAIN_OPERATION_PAD_H__

#include "cker/operation/Pad.h"

namespace nnfw
{
namespace cker
{
namespace train
{

/*
 * input_data will be transformed by PAD operation with padding options(such as constant C) to
 * output_data
 *
 * input_data  ->  output_data
 *   [0,1]     ->    [C,C,C,C]
 *   [2,3]     ->    [C,0,1,C]
 *             ->    [C,2,3,C]
 *             ->    [C,C,C,C]
 */
/*
 * input_data(backward_output_data) will be transformed by backward of PAD operation (Depad) with
 * padding options to output_data(backward_input_data)
 *
 * input_data(backward_output_data)  ->  output_data(backward_input_data)
 *   [C,C,C,C]                       ->    [0,1]
 *   [C,0,1,C]                       ->    [2,3]
 *   [C,2,3,C]                       ->
 *   [C,C,C,C]                       ->
 */
template <typename T>
inline void Depad(const int32_t *padding_data, int32_t pad_rank, const Shape &input_shape,
                  const T *input_data, const Shape &output_shape, T *output_data)
{
  using PaddingInfo = std::pair<int32_t, int32_t>;
  using PaddingList = std::vector<PaddingInfo>;

  assert(output_shape.DimensionsCount() == input_shape.DimensionsCount());
  assert(output_shape.DimensionsCount() == pad_rank);

  PaddingList padding_list(pad_rank);
  for (int32_t n = 0; n < pad_rank; ++n)
  {
    const int32_t *from = padding_data + (n * 2);
    assert(from[0] >= 0 && from[1] >= 0);
    padding_list[n] = {from[0], from[1]};
  }
  for (int32_t i = 0; i < pad_rank; ++i)
  {
    assert(output_shape.Dims(i) ==
           input_shape.Dims(i) - padding_list[i].first - padding_list[i].second);
  }

  switch (pad_rank)
  {
    case 0:
    case 1:
    {
      const int32_t out_width = output_shape.Dims(0);
      const int32_t padding_left = padding_list[0].first;
      std::memcpy(output_data, input_data + padding_left, out_width * sizeof(T));
      break;
    }
    case 2: // HW
    {
      const int32_t out_height = output_shape.Dims(0);
      const int32_t out_width = output_shape.Dims(1);
      const int32_t in_width = input_shape.Dims(1);
      const int32_t padding_top = padding_list[0].first;
      const int32_t padding_left = padding_list[1].first;
      for (auto h = 0; h < out_height; ++h)
      {
        const auto in_offset = (h + padding_top) * in_width + padding_left;
        const auto out_offset = h * out_width;
        // copy a row of input data to output data
        std::memcpy(output_data + out_offset, input_data + in_offset, out_width * sizeof(T));
      }
      break;
    }
    case 3: // HWC
    {
      const int32_t out_depth = output_shape.Dims(0);
      const int32_t out_height = output_shape.Dims(1);
      const int32_t out_width = output_shape.Dims(2);
      const int32_t out_plain_size = out_height * out_width;
      const int32_t in_height = input_shape.Dims(1);
      const int32_t in_width = input_shape.Dims(2);
      const int32_t in_plain_size = in_height * in_width;
      const int32_t padding_depth = padding_list[0].first;
      const int32_t padding_top = padding_list[1].first;
      const int32_t padding_left = padding_list[2].first;
      for (auto d = 0; d < out_depth; ++d)
      {
        for (auto h = 0; h < out_height; ++h)
        {
          const auto in_offset =
            (d + padding_depth) * in_plain_size + (h + padding_top) * in_width + (padding_left);
          const auto out_offset = (d * out_plain_size) + (h * out_width);
          // copy a row of input data to output data
          std::memcpy(output_data + out_offset, input_data + in_offset, out_width * sizeof(T));
        }
      }
      break;
    }
    // NOTE: Assume that tensors' memory format is NHWC. In cker, it cannot be checked.
    case 4:
    {
      const int32_t out_cube = output_shape.Dims(0);
      const int32_t out_depth = output_shape.Dims(1);
      const int32_t out_height = output_shape.Dims(2);
      const int32_t out_width = output_shape.Dims(3);
      const int32_t out_plain_size = out_height * out_width;
      const int32_t out_cube_size = out_depth * out_plain_size;
      const int32_t in_depth = input_shape.Dims(1);
      const int32_t in_height = input_shape.Dims(2);
      const int32_t in_width = input_shape.Dims(3);
      const int32_t in_plain_size = in_height * in_width;
      const int32_t in_cube_size = in_depth * in_plain_size;
      const int32_t padding_cube = padding_list[0].first;
      const int32_t padding_depth = padding_list[1].first;
      const int32_t padding_top = padding_list[2].first;
      const int32_t padding_left = padding_list[3].first;
      for (auto c = 0; c < out_cube; ++c)
      {
        for (auto d = 0; d < out_depth; ++d)
        {
          for (auto h = 0; h < out_height; ++h)
          {
            const auto in_offset = (c + padding_cube) * in_cube_size +
                                   (d + padding_depth) * in_plain_size +
                                   (h + padding_top) * in_width + (padding_left);
            const auto out_offset = (c * out_cube_size) + (d * out_plain_size) + (h * out_width);
            // copy a row of input data to output data
            std::memcpy(output_data + out_offset, input_data + in_offset, out_width * sizeof(T));
          }
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Padding for rank > 4 NYI");
      break;
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_PAD_H__
