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
#include <luci/IR/DataTypeHelper.h>
#include <loco/IR/DataTypeTraits.h>

namespace circle_planner
{

class ScratchpadHelperLinux : public IScratchpadHelper
{
public:
  uint32_t ComputeScratchpadSizeAveragePool2d(const luci::CircleAveragePool2D *avg_pool) final
  {
    // for linux AveragePool2d scratchpad tensors size = 0
    return 0;
  }

  std::vector<uint32_t>
  ComputeScratchpadSizeBatchMatMul(const luci::CircleBatchMatMul *batch_mat_mul) final
  {
    const auto lhs = loco::must_cast<luci::CircleNode *>(batch_mat_mul->x());
    const auto rhs = loco::must_cast<luci::CircleNode *>(batch_mat_mul->y());

    std::vector<uint32_t> scratchpad_sizes;

    // Scratchpad for lhs
    uint32_t scratchpad_size = 1;
    for (int32_t i = 0; i < lhs->rank(); ++i)
      scratchpad_size *= lhs->dim(i).value();

    scratchpad_sizes.push_back(scratchpad_size * luci::size(lhs->dtype()));

    // Scratchpad for rhs
    scratchpad_size = 1;
    for (int32_t i = 0; i < rhs->rank(); ++i)
      scratchpad_size *= rhs->dim(i).value();

    scratchpad_sizes.push_back(scratchpad_size * luci::size(rhs->dtype()));

    return scratchpad_sizes;
  }

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

  uint32_t
  ComputeScratchpadSizeDepthwiseConv2d(const luci::CircleDepthwiseConv2D *depthwise_conv) final
  {
    // for linux DepthwiseConv2d scratchpad tensors size = 0
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

#endif // CIRCLE_EXECUTION_PLAN_SCRATCHPAD_HELPER_LINUX_H
