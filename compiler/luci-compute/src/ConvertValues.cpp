/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvertValues.h"

#include <limits> // std::numeric_limits
#include <stdexcept>

namespace luci
{
namespace compute
{

void get_act_minmax(const FusedActFunc act, float &act_min, float &act_max)
{
  switch (act)
  {
    case FusedActFunc::NONE:
    case FusedActFunc::TANH:
      act_min = std::numeric_limits<float>::lowest();
      act_max = std::numeric_limits<float>::max();
      break;
    case FusedActFunc::RELU:
      act_min = 0;
      act_max = std::numeric_limits<float>::max();
      break;
    case FusedActFunc::RELU_N1_TO_1:
      act_min = -1;
      act_max = 1;
      break;
    case FusedActFunc::RELU6:
      act_min = 0;
      act_max = 6;
      break;
    default:
      throw std::runtime_error("luci-comp get_act_minmax unsupported type.");
  }
}

} // namespace compute
} // namespace luci
