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

#ifndef __NNFW_CKER_PAD_H__
#define __NNFW_CKER_PAD_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include <stdexcept>
#include <iostream>
namespace nnfw
{
namespace cker
{
template <typename T>
inline void Pad(const int32_t *padding_data, int32_t pad_rank, const Shape &input_shape,
                const T *input_data, const Shape &output_shape, T *output_data,
                const T *constant_value_data)
{
  // Note, this is pad with mode=`CONSTANT`: it doesn't support `REFLECT` and `SYMMETRIC`
  // TODO: come up with more subtle solution that uses subtensors like arm compute
  // TODO: Check if it works for all layouts

  using PaddingInfo = std::pair<int32_t, int32_t>;
  /** List of padding information */
  using PaddingList = std::vector<PaddingInfo>;

  const T constant_value = constant_value_data ? *constant_value_data : 0;
  assert(output_shape.DimensionsCount() == input_shape.DimensionsCount());

  PaddingList padding_list(pad_rank);
  for (int32_t n = 0; n < pad_rank; ++n)
  {
    const int32_t *from = padding_data + (n * 2);
    padding_list[n] = {from[0], from[1]};
  }
  for (int32_t i = 0; i < pad_rank; ++i)
  {
    assert(output_shape.Dims(i) ==
           input_shape.Dims(i) + padding_list[i].first + padding_list[i].second);
  }
  /* Use pad_rank since given input/output shapes are expanded to 4d before calling all cker
     functions:
     1. to prevent access violation in padding_list;
     2. handling as 4d is slower than as 2d/3d.
  */
  switch (pad_rank)
  {
    case 0:
    case 1:
    {
      const int32_t in_row_len = input_shape.Dims(0);
      [[maybe_unused]] auto [pad_before, pad_after] = padding_list[0];
      std::fill_n(output_data, pad_before, constant_value);
      std::memcpy(output_data + pad_before, input_data, in_row_len * sizeof(T));
      std::fill_n(output_data + pad_before + in_row_len, pad_after, constant_value);
      break;
    }
    case 2: // HW
    {
      const int32_t in_row_len = input_shape.Dims(1);
      const int32_t out_row_size = output_shape.Dims(1);

      auto [pad_top, pad_bottom] = padding_list[0];
      auto [pad_left, pad_right] = padding_list[1];

      // Prepend padding rows
      std::fill_n(output_data, pad_top * out_row_size, constant_value);

      const auto r_h_inp_lim = input_shape.Dims(0) + pad_top;
      for (auto i = pad_top, j = 0; i < r_h_inp_lim; ++i, ++j)
      {
        auto out_offset = i * out_row_size;
        const auto in_offset = j * in_row_len;

        // Prepend padding values
        std::fill_n(output_data + out_offset, pad_left, constant_value);
        out_offset += pad_left;

        // Copy a row of input data
        memcpy(output_data + out_offset, input_data + in_offset, in_row_len * sizeof(T));
        out_offset += in_row_len;

        // Append padding values
        std::fill_n(output_data + out_offset, pad_right, constant_value);
      }

      // Append padding rows
      std::fill_n(output_data + r_h_inp_lim * out_row_size, pad_bottom * out_row_size,
                  constant_value);
      break;
    }
    case 3: // HWC
    {
      const int32_t in_row_len = input_shape.Dims(2);
      const int32_t out_row_size = output_shape.Dims(2);
      const auto plain_size = out_row_size * output_shape.Dims(1);

      auto [pad_batches_before, pad_batches_after] = padding_list[0];
      auto [pad_parallelepipes_before, pad_parallelepipes_after] = padding_list[1];
      auto [pad_plains_before, pad_plains_after] = padding_list[2];

      // Prepend padding plains
      std::fill_n(output_data, pad_batches_before * plain_size, constant_value);

      const auto r_h_inp_lim = input_shape.Dims(0) + pad_batches_before;
      for (auto i = pad_batches_before, i_inp = 0; i < r_h_inp_lim; ++i, ++i_inp)
      {
        const auto out_w_offset = (i * output_shape.Dims(1)) * output_shape.Dims(2);

        // Prepend padding rows
        std::fill_n(output_data + out_w_offset, pad_parallelepipes_before * out_row_size,
                    constant_value);

        const auto r_w_inp_lim = input_shape.Dims(1) + pad_parallelepipes_before;
        for (auto j = pad_parallelepipes_before, j_inp = 0; j < r_w_inp_lim; ++j, ++j_inp)
        {
          auto out_offset = (i * output_shape.Dims(1) + j) * output_shape.Dims(2);
          const auto in_offset = (i_inp * input_shape.Dims(1) + j_inp) * input_shape.Dims(2);

          // Prepend padding values
          std::fill_n(output_data + out_offset, pad_plains_before, constant_value);
          out_offset += pad_plains_before;

          // Copy a row of input data
          memcpy(output_data + out_offset, input_data + in_offset, in_row_len * sizeof(T));
          out_offset += in_row_len;

          // Append padding values
          std::fill_n(output_data + out_offset, pad_plains_after, constant_value);
        }

        // Append padding rows
        std::fill_n(output_data + out_w_offset + r_w_inp_lim * out_row_size,
                    pad_parallelepipes_after * out_row_size, constant_value);
      }

      // Append padding plains
      std::fill_n(output_data + r_h_inp_lim * plain_size, pad_batches_after * plain_size,
                  constant_value);
      break;
    }
    case 4:
    {
      auto get_offset = [](const Shape &shape, int32_t n, int32_t h, int32_t w) -> int32_t {
        return ((n * shape.Dims(1) + h) * shape.Dims(2) + w) * shape.Dims(3);
      };
      const int32_t in_row_len = input_shape.Dims(3);
      const int32_t out_row_size = output_shape.Dims(3);
      const auto plain_size = out_row_size * output_shape.Dims(2);
      const auto parallelepiped_size = plain_size * output_shape.Dims(1);

      auto [pad_batches_before, pad_batches_after] = padding_list[0];
      auto [pad_parallelepipes_before, pad_parallelepipes_after] = padding_list[1];
      auto [pad_plains_before, pad_plains_after] = padding_list[2];
      auto [pad_rows_before, pad_rows_after] = padding_list[3];

      // Prepend padding parallelepipeds
      std::fill_n(output_data, pad_batches_before * parallelepiped_size, constant_value);

      const auto r_b_inp_lim = input_shape.Dims(0) + pad_batches_before;
      for (auto i = pad_batches_before, i_inp = 0; i < r_b_inp_lim; ++i, ++i_inp)
      {
        const auto out_h_offset = get_offset(output_shape, i, 0, 0);
        // Prepend padding plains
        std::fill_n(output_data + out_h_offset, pad_parallelepipes_before * plain_size,
                    constant_value);

        const auto r_h_inp_lim = input_shape.Dims(1) + pad_parallelepipes_before;
        for (auto j = pad_parallelepipes_before, j_inp = 0; j < r_h_inp_lim; ++j, ++j_inp)
        {
          const auto out_w_offset = get_offset(output_shape, i, j, 0);

          // Prepend padding rows
          std::fill_n(output_data + out_w_offset, pad_plains_before * out_row_size, constant_value);

          const auto r_w_inp_lim = input_shape.Dims(2) + pad_plains_before;
          for (auto k = pad_plains_before, k_inp = 0; k < r_w_inp_lim; ++k, ++k_inp)
          {
            auto out_c_offset = get_offset(output_shape, i, j, k);
            const auto in_offset = get_offset(input_shape, i_inp, j_inp, k_inp);

            // Prepend padding values
            std::fill_n(output_data + out_c_offset, pad_rows_before, constant_value);
            out_c_offset += pad_rows_before;

            // Copy a row of input data
            memcpy(output_data + out_c_offset, input_data + in_offset, in_row_len * sizeof(T));
            out_c_offset += in_row_len;

            // Append padding values
            std::fill_n(output_data + out_c_offset, pad_rows_after, constant_value);
          }

          // Append padding rows
          std::fill_n(output_data + out_w_offset + r_w_inp_lim * out_row_size,
                      pad_plains_after * out_row_size, constant_value);
        }

        // Append padding plains
        std::fill_n(output_data + out_h_offset + r_h_inp_lim * plain_size,
                    pad_parallelepipes_after * plain_size, constant_value);
      }

      // Append padding parallelepipeds
      std::fill_n(output_data + r_b_inp_lim * parallelepiped_size,
                  pad_batches_after * parallelepiped_size, constant_value);
      break;
      break;
    }
    default:
      throw std::runtime_error("Padding for rank > 4 NYI");
      break;
  }
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_PAD_H__
