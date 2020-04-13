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

#ifndef __NNOP_CONV2D_H__
#define __NNOP_CONV2D_H__

#include "nnop/PadInfo.h"
#include "nnop/StrideInfo.h"

#include <nncc/core/ADT/feature/Shape.h>
#include <nncc/core/ADT/feature/Reader.h>
#include <nncc/core/ADT/feature/Accessor.h>

#include <nncc/core/ADT/kernel/Shape.h>
#include <nncc/core/ADT/kernel/Reader.h>

namespace nnop
{

template <typename OutputDType, typename InputDType, typename KernelDType>
void conv(const nncc::core::ADT::feature::Shape &out_shape,
          nncc::core::ADT::feature::Accessor<OutputDType> &out_data,
          const nncc::core::ADT::feature::Shape &in_shape,
          const nncc::core::ADT::feature::Reader<InputDType> &in_data,
          const nncc::core::ADT::kernel::Shape &ker_shape,
          const nncc::core::ADT::kernel::Reader<KernelDType> &ker_data, const PadInfo &pad_info,
          const StrideInfo &stride_info)
{
  for (uint32_t out_ch = 0; out_ch < out_shape.depth(); ++out_ch)
  {
    for (uint32_t out_row = 0; out_row < out_shape.height(); ++out_row)
    {
      for (uint32_t out_col = 0; out_col < out_shape.width(); ++out_col)
      {
        OutputDType out_value = 0;

        for (uint32_t ker_ch = 0; ker_ch < ker_shape.depth(); ++ker_ch)
        {
          for (uint32_t ker_row = 0; ker_row < ker_shape.height(); ++ker_row)
          {
            for (uint32_t ker_col = 0; ker_col < ker_shape.width(); ++ker_col)
            {
              const int64_t vertical_stride = static_cast<int64_t>(stride_info.vertical());
              const int64_t horizontal_stride = static_cast<int64_t>(stride_info.horizontal());
              const int64_t top_padding = static_cast<int64_t>(pad_info.top());
              const int64_t left_padding = static_cast<int64_t>(pad_info.left());

              const uint32_t in_ch = ker_ch;
              const int64_t in_row = vertical_stride * out_row - top_padding + ker_row;
              const int64_t in_col = horizontal_stride * out_col - left_padding + ker_col;

              const bool is_padding = (in_row < 0) || (in_row >= in_shape.height()) ||
                                      (in_col < 0) || (in_col >= in_shape.width());

              const auto in_value = (is_padding) ? 0
                                                 : in_data.at(in_ch, static_cast<uint32_t>(in_row),
                                                              static_cast<uint32_t>(in_col));

              const auto ker_value = ker_data.at(out_ch, in_ch, ker_row, ker_col);

              out_value += in_value * ker_value;
            }
          }
        }

        out_data.at(out_ch, out_row, out_col) = out_value;
      }
    }
  }
}

} // namespace nnop

#endif // __NNOP_CONV2D_H__
