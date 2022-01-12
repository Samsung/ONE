/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_LINUX_H
#define CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_LINUX_H

#include "IScratchpadHelper.h"

namespace circle_planner
{

class ScratchpadHelperLinux : public IScratchpadHelper
{
public:
  uint32_t ComputeScratchpadSizeConv2d(const luci::CircleConv2D *conv) final
  {
    const auto conv_input = loco::must_cast<luci::CircleNode *>(conv->input());
    const auto filter = loco::must_cast<luci::CircleNode *>(conv->filter());

    const uint32_t stride_height = conv->stride()->h();
    const uint32_t stride_width = conv->stride()->w();

    const uint32_t dilation_height_factor = conv->dilation()->h();
    const uint32_t dilation_width_factor = conv->dilation()->w();

    const uint32_t filter_height = filter->dim(1).value();
    const uint32_t filter_width = filter->dim(2).value();

    const bool need_dilated_im2col = dilation_height_factor != 1 || dilation_width_factor != 1;
    const bool need_non_dilated_im2col =
      stride_height != 1 || stride_width != 1 || filter_height != 1 || filter_width != 1;
    const bool need_im2col = conv_input->dtype() != loco::DataType::S16 &&
                             (need_dilated_im2col || need_non_dilated_im2col);

    if (!need_im2col)
    {
      return 0;
    }

    const uint32_t input_depth = conv_input->dim(3).value();
    const uint32_t batches = conv_input->dim(0).value();

    const uint32_t output_height = conv->dim(1).value();
    const uint32_t output_width = conv->dim(2).value();

    return batches * output_height * output_width * input_depth * filter_height * filter_width *
           size(conv_input->dtype());
  }
};

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_LINUX_H
