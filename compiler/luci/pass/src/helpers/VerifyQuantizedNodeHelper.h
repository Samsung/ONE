/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_VERIFY_QUANTIZED_NODE_HELPER_H__
#define __LUCI_VERIFY_QUANTIZED_NODE_HELPER_H__

#include <luci/IR/CircleNodes.h>
#include <luci/Pass/QuantizationParameters.h>

namespace luci
{

namespace verify_quantization
{

using Granularity = luci::QuantizationGranularity;
using Type = loco::DataType;

uint32_t rank(const loco::Node *node)
{
  auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
  return circle_node->rank();
}

bool has_type(const loco::Node *node, Type dtype)
{
  auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
  return circle_node->dtype() == dtype;
}

bool is_lwq_const(const loco::Node *node)
{
  auto circle_node = loco::must_cast<const luci::CircleConst *>(node);

  if (circle_node->quantparam()->scale.size() != 1)
    return false;

  if (circle_node->quantparam()->zerop.size() != 1)
    return false;

  return true;
}

bool is_cwq_const(const loco::Node *node, uint32_t channel_dim)
{
  auto circle_node = loco::must_cast<const luci::CircleConst *>(node);

  assert(channel_dim < circle_node->rank()); // FIX_CALLER_UNLESS
  auto channel_size = circle_node->dim(channel_dim).value();

  if (circle_node->quantparam()->scale.size() != channel_size)
    return false;

  if (circle_node->quantparam()->zerop.size() != channel_size)
    return false;

  return true;
}

} // namespace verify_quantization
} // namespace luci

#endif // __LUCI_VERIFY_QUANTIZED_NODE_HELPER_H__
