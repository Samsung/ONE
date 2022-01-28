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

#ifndef CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_MCU_H
#define CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_MCU_H

#include "IScratchpadHelper.h"

namespace circle_planner
{

class ScratchpadHelperMCU : public IScratchpadHelper
{
public:
  uint32_t ComputeScratchpadSizeAveragePool2d(const luci::CircleAveragePool2D *avg_pool) final
  {
    // for mcu AveragePool2d scratchpad tensors size = 0
    return 0;
  }

  uint32_t ComputeScratchpadSizeConv2d(const luci::CircleConv2D *) final
  {
    // for mcu scratchpad size = 0
    return 0;
  }

  uint32_t
  ComputeScratchpadSizeDepthwiseConv2d(const luci::CircleDepthwiseConv2D *depthwise_conv) final
  {
    // for mcu DepthwiseConv2d scratchpad tensors size = 0
    return 0;
  }

  std::vector<uint32_t> ComputeScratchpadSizeSVDF(const luci::CircleSVDF *svdf) final
  {
    const auto svdf_input = loco::must_cast<luci::CircleNode *>(svdf->input());
    const auto weight_feature_input = loco::must_cast<luci::CircleNode *>(svdf->weight_feature());

    if (svdf_input->dtype() == loco::DataType::FLOAT32 and
        (weight_feature_input->dtype() == loco::DataType::S8 or
         weight_feature_input->dtype() == loco::DataType::U8))
    {
      throw std::runtime_error("Hybrid type is not currently supported for linux platform");
    }

    std::vector<uint32_t> scratchpad_sizes;

    const auto batch_size = svdf_input->dim(0).value();
    const auto num_filters = weight_feature_input->dim(0).value();
    const auto rank = svdf->svdf_rank();
    const auto num_units = num_filters / rank;

    if (svdf_input->dtype() == loco::DataType::S8)
    {
      scratchpad_sizes.push_back(batch_size * num_filters * sizeof(int32_t));
      scratchpad_sizes.push_back(batch_size * num_units * sizeof(int32_t));
    }
    else
    {
      scratchpad_sizes.push_back(batch_size * num_filters * sizeof(float));
    }

    return scratchpad_sizes;
  }
};

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_MCU_H
