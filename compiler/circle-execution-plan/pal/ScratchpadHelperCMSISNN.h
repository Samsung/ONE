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

#ifndef CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_CMSISNN_H
#define CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_CMSISNN_H

#include "IScratchpadHelper.h"
#include <cassert>

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

  uint32_t ComputeScratchpadSizeAveragePool2d(const luci::CircleAveragePool2D *avg_pool) final
  {
    // Main logic of arm_avgpool_s8_get_buffer_size

    const auto avg_pool_input = loco::must_cast<luci::CircleNode *>(avg_pool->value());

    if (avg_pool_input->dtype() != loco::DataType::S8 or !_use_dsp)
      return 0;

    const auto depth = static_cast<int32_t>(avg_pool_input->dim(3).value());

    return depth * sizeof(int32_t);
  }

  uint32_t ComputeScratchpadSizeConv2d(const luci::CircleConv2D *conv) final
  {
    // Main logic of arm_convolve_wrapper_s8_get_buffer_size

    const auto dilation_height_factor = static_cast<int32_t>(conv->dilation()->h());
    const auto dilation_width_factor = static_cast<int32_t>(conv->dilation()->w());

    const auto conv_input = loco::must_cast<luci::CircleNode *>(conv->input());
    const auto filter = loco::must_cast<luci::CircleNode *>(conv->filter());

    if (dilation_width_factor != 1 or dilation_height_factor != 1 or
        conv_input->dtype() != loco::DataType::S8)
    {
      return 0;
    }

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
    {
      return (2 * input_depth * filter_width * filter_height) * sizeof(int16_t);
    }

    return 0;
  }

  uint32_t
  ComputeScratchpadSizeDepthwiseConv2d(const luci::CircleDepthwiseConv2D *depthwise_conv) final
  {
    // Main logic of arm_depthwise_conv_wrapper_s8_get_buffer_size

    const auto dilation_height_factor = static_cast<int32_t>(depthwise_conv->dilation()->h());
    const auto dilation_width_factor = static_cast<int32_t>(depthwise_conv->dilation()->w());

    const auto depthwise_conv_input = loco::must_cast<luci::CircleNode *>(depthwise_conv->input());
    const auto filter = loco::must_cast<luci::CircleNode *>(depthwise_conv->filter());

    if (dilation_width_factor != 1 or dilation_height_factor != 1 or
        depthwise_conv_input->dtype() != loco::DataType::S8)
    {
      return 0;
    }

    const auto input_depth = static_cast<int32_t>(depthwise_conv_input->dim(3).value());
    const auto output_depth = static_cast<int32_t>(depthwise_conv->dim(3).value());
    const auto batch_size = static_cast<int32_t>(depthwise_conv_input->dim(0).value());

    if (input_depth != output_depth or batch_size != 1 or !_use_dsp)
      return 0;

    const auto filter_height = static_cast<int32_t>(filter->dim(1).value());
    const auto filter_width = static_cast<int32_t>(filter->dim(2).value());

    return input_depth * filter_height * filter_width * sizeof(int16_t);
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

private:
  bool _use_dsp;
};

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_CMSISNN_H
