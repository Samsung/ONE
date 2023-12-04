/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Compute.h"

namespace luci
{

bool to_compute(const Padding padding, compute::PaddingType &padding_type)
{
  switch (padding)
  {
    case Padding::SAME:
      padding_type = compute::PaddingType::kSame;
      break;

    case Padding::VALID:
      padding_type = compute::PaddingType::kValid;
      break;

    default:
      return false;
  }
  return true;
}

bool to_compute(const FusedActFunc act, compute::FusedActFunc &act_func)
{
  switch (act)
  {
    case FusedActFunc::NONE:
      act_func = compute::FusedActFunc::NONE;
      break;

    case FusedActFunc::TANH:
      act_func = compute::FusedActFunc::TANH;
      break;

    case FusedActFunc::RELU:
      act_func = compute::FusedActFunc::RELU;
      break;

    case FusedActFunc::RELU_N1_TO_1:
      act_func = compute::FusedActFunc::RELU_N1_TO_1;
      break;

    case FusedActFunc::RELU6:
      act_func = compute::FusedActFunc::RELU6;
      break;

    default:
      return false;
  }
  return true;
}

} // namespace luci
