/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __KBENCHMARK_KERNELS_ACL_COMMON_UTILS_H__
#define __KBENCHMARK_KERNELS_ACL_COMMON_UTILS_H__

#include <algorithm>

using namespace arm_compute;

namespace kbenchmark
{
namespace kernels
{
namespace acl_common
{

struct PaddingInfo
{
  uint32_t top;
  uint32_t bottom;
  uint32_t left;
  uint32_t right;
};

PaddingInfo calculatePadding(const std::string &padding_name, const uint32_t ifm_H,
                             const uint32_t ifm_W, const uint32_t ofm_H, const uint32_t ofm_W,
                             const uint32_t vertical_stride, const uint32_t horizontal_stride,
                             const uint32_t ker_H, const uint32_t ker_W)
{
  uint32_t top;
  uint32_t bottom;
  uint32_t left;
  uint32_t right;

  if (padding_name == "VALID")
  {
    top = bottom = left = right = 0;
  }
  else if (padding_name == "SAME")
  {
    const int32_t vertical_needed_input = (ofm_H - 1) * vertical_stride + ker_H;
    const int32_t vertical_total_padding = std::max(0, vertical_needed_input - (int32_t)ifm_H);

    const int32_t horizontal_needed_input = (ofm_W - 1) * horizontal_stride + ker_W;
    const int32_t horizontal_total_padding = std::max(0, horizontal_needed_input - (int32_t)ifm_W);

    top = vertical_total_padding / 2;
    bottom = (vertical_total_padding + 1) / 2;
    left = horizontal_total_padding / 2;
    right = (horizontal_total_padding + 1) / 2;
  }

  return PaddingInfo{top, bottom, left, right};
}

PadStrideInfo asPadStrideInfo(const PaddingInfo &padding, uint32_t vertical_stride,
                              uint32_t horizontal_stride)
{
  return PadStrideInfo{horizontal_stride,
                       vertical_stride,
                       padding.left,
                       padding.right,
                       padding.top,
                       padding.bottom,
                       DimensionRoundingType::FLOOR};
}

ActivationLayerInfo asActivationLayerInfo(const std::string &act_name)
{
  if (act_name == "NONE")
  {
    return ActivationLayerInfo{};
  }
  else if (act_name == "RELU")
  {
    return ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::RELU};
  }
  else
  {
    throw std::runtime_error{"Not support activation layer info"};
  }
}

} // namespace acl_common
} // namespace kernels
} // namespace kbenchmark

#endif // __KBENCHMARK_KERNELS_ACL_COMMON_UTILS_H__
