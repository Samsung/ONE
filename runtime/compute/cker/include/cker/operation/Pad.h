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
      std::fill_n(output_data, padding_list[0].first, constant_value);
      std::memcpy(output_data + padding_list[0].first, input_data, in_row_len * sizeof(T));
      std::fill_n(output_data + padding_list[0].first + in_row_len, padding_list[0].second,
                  constant_value);
      break;
    }
    case 2: // HW
    {
      const int32_t in_row_len = input_shape.Dims(1);
      const int32_t out_row_size = output_shape.Dims(1);

      // prepend padding rows
      std::fill_n(output_data, padding_list[0].first * out_row_size, constant_value);

      const auto r_h_inp_lim = input_shape.Dims(0) + padding_list[0].first;
      for (auto i = padding_list[0].first, j = 0; i < r_h_inp_lim; ++i, ++j)
      {
        auto out_offset = i * out_row_size;
        const auto in_offset = j * in_row_len;

        // prepend padding values
        std::fill_n(output_data + out_offset, padding_list[1].first, constant_value);

        out_offset += padding_list[1].first;

        // copy a row of input data
        memcpy(output_data + out_offset, input_data + in_offset, in_row_len * sizeof(T));

        out_offset += in_row_len;

        // append padding values
        std::fill_n(output_data + out_offset, padding_list[1].second, constant_value);
      }

      // append padding rows
      std::fill_n(output_data + r_h_inp_lim * out_row_size, padding_list[0].second * out_row_size,
                  constant_value);
      break;
    }
    case 3: // HWC
    {
      const int32_t in_row_len = input_shape.Dims(2);
      const int32_t out_row_size = output_shape.Dims(2);
      const auto plain_size = out_row_size * output_shape.Dims(1);

      // prepend padding plains
      std::fill_n(output_data, padding_list[0].first * plain_size, constant_value);

      const auto r_h_inp_lim = input_shape.Dims(0) + padding_list[0].first;
      for (auto i = padding_list[0].first, i_inp = 0; i < r_h_inp_lim; ++i, ++i_inp)
      {
        const auto out_w_offset = (i * output_shape.Dims(1) + 0) * output_shape.Dims(2);

        // prepend padding rows
        std::fill_n(output_data + out_w_offset, padding_list[1].first * out_row_size,
                    constant_value);

        const auto r_w_inp_lim = input_shape.Dims(1) + padding_list[1].first;
        for (auto j = padding_list[1].first, j_inp = 0; j < r_w_inp_lim; ++j, ++j_inp)
        {
          auto out_offset = (i * output_shape.Dims(1) + j) * output_shape.Dims(2);
          const auto in_offset = (i_inp * input_shape.Dims(1) + j_inp) * input_shape.Dims(2);

          // prepend padding values
          std::fill_n(output_data + out_offset, padding_list[2].first, constant_value);

          out_offset += padding_list[2].first;

          // copy a row of input data
          memcpy(output_data + out_offset, input_data + in_offset, in_row_len * sizeof(T));

          out_offset += in_row_len;

          // append padding values
          std::fill_n(output_data + out_offset, padding_list[2].second, constant_value);
        }

        // append padding rows
        std::fill_n(output_data + out_w_offset + r_w_inp_lim * out_row_size,
                    padding_list[1].second * out_row_size, constant_value);
      }

      // append padding plains
      std::fill_n(output_data + r_h_inp_lim * plain_size, padding_list[0].second * plain_size,
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

      // prepend padding parallelepipeds
      std::fill_n(output_data, padding_list[0].first * parallelepiped_size, constant_value);

      const auto r_b_inp_lim = input_shape.Dims(0) + padding_list[0].first;
      for (auto i = padding_list[0].first, i_inp = 0; i < r_b_inp_lim; ++i, ++i_inp)
      {
        const auto out_h_offset = get_offset(output_shape, i, 0, 0);
        // prepend padding plains
        std::fill_n(output_data + out_h_offset, padding_list[1].first * plain_size, constant_value);

        const auto r_h_inp_lim = input_shape.Dims(1) + padding_list[1].first;
        for (auto j = padding_list[1].first, j_inp = 0; j < r_h_inp_lim; ++j, ++j_inp)
        {
          const auto out_w_offset = get_offset(output_shape, i, j, 0);

          // prepend padding rows
          std::fill_n(output_data + out_w_offset, padding_list[2].first * out_row_size,
                      constant_value);

          const auto r_w_inp_lim = input_shape.Dims(2) + padding_list[2].first;
          for (auto k = padding_list[2].first, k_inp = 0; k < r_w_inp_lim; ++k, ++k_inp)
          {
            auto out_c_offset = get_offset(output_shape, i, j, k);
            const auto in_offset = get_offset(input_shape, i_inp, j_inp, k_inp);

            // prepend padding values
            std::fill_n(output_data + out_c_offset, padding_list[3].first, constant_value);

            out_c_offset += padding_list[3].first;

            // copy a row of input data
            memcpy(output_data + out_c_offset, input_data + in_offset, in_row_len * sizeof(T));

            out_c_offset += in_row_len;

            // append padding values
            std::fill_n(output_data + out_c_offset, padding_list[3].second, constant_value);
          }

          // append padding rows
          std::fill_n(output_data + out_w_offset + r_w_inp_lim * out_row_size,
                      padding_list[2].second * out_row_size, constant_value);
        }

        // append padding plains
        std::fill_n(output_data + out_h_offset + r_h_inp_lim * plain_size,
                    padding_list[1].second * plain_size, constant_value);
      }
      // append padding parallelepipeds
      std::fill_n(output_data + r_b_inp_lim * parallelepiped_size,
                  padding_list[0].second * parallelepiped_size, constant_value);
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
