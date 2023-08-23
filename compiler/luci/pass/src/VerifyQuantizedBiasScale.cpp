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

#include "VerifyQuantizedBiasScale.h"

#include <cmath>

// This macro is undef at the end of the file
#define RETURN_FALSE_UNLESS(ARG) \
  if (not(ARG))                  \
  {                              \
    return false;                \
  }

namespace
{

bool same(float a, float b)
{
  constexpr float epsilon = 1e-10;
  return std::abs(a - b) < epsilon;
}

// Check bias scale = input scale * weight scale
// This function checks both LWQ and CWQ
bool check_bias_scale(const loco::Node *input, const loco::Node *weights, const loco::Node *bias)
{
  auto input_node = loco::must_cast<const luci::CircleNode *>(input);
  auto input_qparam = input_node->quantparam();
  RETURN_FALSE_UNLESS(input_qparam != nullptr);

  auto weights_node = loco::must_cast<const luci::CircleNode *>(weights);
  auto weights_qparam = weights_node->quantparam();
  RETURN_FALSE_UNLESS(weights_qparam != nullptr);

  auto bias_node = loco::must_cast<const luci::CircleNode *>(bias);
  auto bias_qparam = bias_node->quantparam();
  RETURN_FALSE_UNLESS(bias_qparam != nullptr);

  RETURN_FALSE_UNLESS(input_qparam->scale.size() == 1);
  RETURN_FALSE_UNLESS(weights_qparam->scale.size() == bias_qparam->scale.size());

  auto input_scale = input_qparam->scale[0];
  for (uint32_t i = 0; i < weights_qparam->scale.size(); i++)
  {
    auto weights_scale = weights_qparam->scale[i];
    auto bias_scale = bias_qparam->scale[i];
    RETURN_FALSE_UNLESS(same(bias_scale, input_scale * weights_scale));
  }
  return true;
}

} // namespace

namespace luci
{

bool VerifyQuantizedBiasScale::visit(const luci::CircleConv2D *node)
{
  RETURN_FALSE_UNLESS(check_bias_scale(node->input(), node->filter(), node->bias()));
  return true;
}

bool VerifyQuantizedBiasScale::visit(const luci::CircleDepthwiseConv2D *node)
{
  RETURN_FALSE_UNLESS(check_bias_scale(node->input(), node->filter(), node->bias()));
  return true;
}

bool VerifyQuantizedBiasScale::visit(const luci::CircleFullyConnected *node)
{
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
  if (bias != nullptr)
  {
    RETURN_FALSE_UNLESS(check_bias_scale(node->input(), node->weights(), node->bias()));
  }
  return true;
}

bool VerifyQuantizedBiasScale::visit(const luci::CircleTransposeConv *node)
{
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
  if (bias != nullptr)
  {
    RETURN_FALSE_UNLESS(check_bias_scale(node->outBackprop(), node->filter(), node->bias()));
  }
  return true;
}

} // namespace luci

#undef RETURN_FALSE_UNLESS
