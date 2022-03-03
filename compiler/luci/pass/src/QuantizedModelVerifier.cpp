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

#include "QuantizedModelVerifier.h"

#include "VerifyQuantizedNodeGranularity.h"
#include "VerifyQuantizedNodeType.h"
#include "VerifyQuantizedBiasScale.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

using LayerInfoMap = std::unordered_map<std::string, luci::LayerInfo *>;

namespace luci
{

LayerInfoMap QuantizedModelVerifier::create_layer_info_map(loco::Graph *g) const
{
  LayerInfoMap info_by_name;

  for (auto &&info : _ctx->layers_info)
  {
    auto name = info.name;
    bool found = false;
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      if (cnode->name() == name)
      {
        if (info_by_name.find(name) != info_by_name.end())
        {
          throw std::runtime_error("Duplicate layer name " + name +
                                   ". Check layer names in the quantization configuration file.");
        }

        info_by_name[name] = &info;
        found = true;
        break;
      }
    }

    if (not found)
      throw std::runtime_error("No such layer named " + name +
                               ". Check layer names in the quantization configuration file.");
  }

  assert(info_by_name.size() == _ctx->layers_info.size()); // FIX_ME_UNLESS

  return info_by_name;
}

void QuantizedModelVerifier::verify(loco::Graph *g)
{
  if (_ctx->granularity != Granularity::ChannelWise && _ctx->granularity != Granularity::LayerWise)
    throw std::runtime_error("Unsupported granularity");

  auto info_by_name = create_layer_info_map(g);

  auto quantize_dtype = [&](const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization dtype
    if (iter != info_by_name.end())
      return iter->second->dtype;

    // Return default quantization dtype
    return _ctx->output_model_dtype;
  };

  auto quantize_granularity = [&](const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization granularity
    if (iter != info_by_name.end())
      return iter->second->granularity;

    // Return default quantization granularity
    return _ctx->granularity;
  };

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    auto node_name = [&circle_node]() {
      if (circle_node->name().length() == 0)
        return std::string("(noname)");

      return circle_node->name();
    };

    // Verify Type
    if (!VerifyQuantizedNodeType::create(quantize_dtype(circle_node))->verify(circle_node))
      throw std::runtime_error("Wrong data type detected in " + node_name());

    // Verify Granularity
    if (!circle_node->accept(
          VerifyQuantizedNodeGranularity::create(quantize_granularity(circle_node)).get()))
      throw std::runtime_error("Wrong granularity detected in " + node_name());

    // Verify Bias scale
    if (!VerifyQuantizedBiasScale::create()->verify(circle_node))
      throw std::runtime_error("Wrong bias scale detected in " + node_name());
  }
}

} // namespace luci
