/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_CMSISNN_H
#define CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_CMSISNN_H

#include "IScratchpadHelper.h"

namespace circle_planner
{

namespace
{

inline int32_t computePadding(int32_t stride, int32_t dilation_rate, int32_t in_size,
                              int32_t filter_size, int32_t out_size)
{
  const int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  const int32_t padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

} // namespace

class ScratchpadHelperCMSISNN : public IScratchpadHelper
{
public:
  explicit ScratchpadHelperCMSISNN(bool use_dsp) : _use_dsp(use_dsp)
  {
    // Do nothing
  }

  uint32_t ComputeScratchpadSizeConv2d(const luci::CircleConv2D *conv) final
  {
    const auto dilation_height_factor = static_cast<int32_t>(conv->dilation()->h());
    const auto dilation_width_factor = static_cast<int32_t>(conv->dilation()->w());

    if (dilation_width_factor != 1 and dilation_height_factor != 1)
      return 0;

    auto conv_input = loco::must_cast<luci::CircleNode *>(conv->input());
    auto filter = loco::must_cast<luci::CircleNode *>(conv->filter());

    assert(conv_input->dtype() == loco::DataType::S8 && "CMSISNN works with int8 models");
    const auto input_depth = static_cast<int32_t>(conv_input->dim(3).value());

    const auto input_height = static_cast<int32_t>(conv_input->dim(1).value());
    const auto input_width = static_cast<int32_t>(conv_input->dim(2).value());

    const auto filter_height = static_cast<int32_t>(filter->dim(1).value());
    const auto filter_width = static_cast<int32_t>(filter->dim(2).value());

    const auto stride_height = static_cast<int32_t>(conv->stride()->h());
    const auto stride_width = static_cast<int32_t>(conv->stride()->w());

    const auto output_height = static_cast<int32_t>(conv->dim(1).value());
    const auto output_width = static_cast<int32_t>(conv->dim(2).value());

    assert(conv_input->quantparam()->zerop.size() == 1);
    assert(conv->quantparam()->zerop.size() == 1);

    const auto padding_height = computePadding(stride_height, dilation_height_factor, input_height,
                                               filter_height, output_height);
    const auto padding_width =
      computePadding(stride_width, dilation_width_factor, input_width, filter_width, output_width);

    if ((padding_width == 0) && (padding_height == 0) && (input_depth % 4 == 0) &&
        (stride_width == 1) && (stride_height == 1) && (filter_width == 1) && (filter_height == 1))
    {
      return 0;
    }

    if (_use_dsp)
      return (2 * input_depth * filter_width * filter_height) * sizeof(int16_t);

    return 0;
  }

private:
  bool _use_dsp;
};

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_CMSISNN_H
