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

#include "LayerInfoMap.h"

#include <luci/IR/CircleNode.h>

#include <cassert>

namespace luci
{
namespace
{
bool is_multiple_output_node(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // The following nodes are multiple-output nodes. They do not produce tensors, the tensors
    // are produced by the corresponding *Out nodes instead.
    case luci::CircleOpcode::CUSTOM:
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::NON_MAX_SUPPRESSION_V4:
    case luci::CircleOpcode::NON_MAX_SUPPRESSION_V5:
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::SPLIT_V:
    case luci::CircleOpcode::TOPK_V2:
    case luci::CircleOpcode::UNIQUE:
    case luci::CircleOpcode::UNPACK:
    case luci::CircleOpcode::WHILE:
      return true;
    // TODO: support this op
    case luci::CircleOpcode::BIDIRECTIONAL_SEQUENCE_LSTM:
      throw std::runtime_error("Unsupported op now");
    default:
      return false;
  }
}

luci::CircleNode *get_multi_output_node(luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // The following nodes denote outputs of multiple-output nodes.
    case luci::CircleOpcode::CIRCLECUSTOMOUT:
    {
      const auto custom_out = loco::must_cast<CircleCustomOut *>(node);
      return loco::must_cast<luci::CircleNode *>(custom_out->input());
    }
    case luci::CircleOpcode::CIRCLEIFOUT:
    {
      const auto if_out = loco::must_cast<CircleIfOut *>(node);
      return loco::must_cast<luci::CircleNode *>(if_out->input());
    }
    case luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV4OUT:
    {
      const auto non_max_sus_v4_out = loco::must_cast<CircleNonMaxSuppressionV4Out *>(node);
      return loco::must_cast<luci::CircleNode *>(non_max_sus_v4_out->input());
    }
    case luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV5OUT:
    {
      const auto non_max_sus_v5_out = loco::must_cast<CircleNonMaxSuppressionV5Out *>(node);
      return loco::must_cast<luci::CircleNode *>(non_max_sus_v5_out->input());
    }
    case luci::CircleOpcode::CIRCLESPLITOUT:
    {
      const auto split_out = loco::must_cast<CircleSplitOut *>(node);
      return loco::must_cast<luci::CircleNode *>(split_out->input());
    }
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    {
      const auto splitv_out = loco::must_cast<CircleSplitVOut *>(node);
      return loco::must_cast<luci::CircleNode *>(splitv_out->input());
    }
    case luci::CircleOpcode::CIRCLETOPKV2OUT:
    {
      const auto top_kv2_out = loco::must_cast<CircleTopKV2Out *>(node);
      return loco::must_cast<luci::CircleNode *>(top_kv2_out->input());
    }
    case luci::CircleOpcode::CIRCLEUNIQUEOUT:
    {
      const auto unique_out = loco::must_cast<CircleUniqueOut *>(node);
      return loco::must_cast<luci::CircleNode *>(unique_out->input());
    }
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    {
      const auto unpack_out = loco::must_cast<CircleUnpackOut *>(node);
      return loco::must_cast<luci::CircleNode *>(unpack_out->input());
    }
    case luci::CircleOpcode::CIRCLEWHILEOUT:
    {
      const auto while_out = loco::must_cast<CircleWhileOut *>(node);
      return loco::must_cast<luci::CircleNode *>(while_out->input());
    }
    case luci::CircleOpcode::CIRCLEBIDIRECTIONAL_SEQUENCE_LSTM_OUT:
      throw std::runtime_error("Unsupported op now");
    default:
      return nullptr;
  }
}

bool check_layer_info_equal(LayerInfo &left, LayerInfo &right)
{
  return left.dtype == right.dtype and left.granularity == right.granularity;
}

void add_multi_output_node(LayerInfoMap &info_by_name, LayerInfo &layer_info,
                           luci::CircleNode *node)
{
  const auto succs_nodes = loco::succs(node);
  auto name = node->name();

  if (info_by_name.find(name) != info_by_name.end())
  {
    // Check that all outputs have equal dtype and granularity
    for (const auto succs_node : succs_nodes)
    {
      const auto succs_circle_node = loco::must_cast<luci::CircleNode *>(succs_node);
      name = succs_circle_node->name();

      const auto it = info_by_name.find(name);
      if (it != info_by_name.end() and not check_layer_info_equal(layer_info, *(it->second)))
        throw std::runtime_error("Outputs of multiple-output nodes should have equal dtype and "
                                 "granularity. Check the quantization configuration file");
    }
    return;
  }

  // Add all output nodes to info_by_name
  info_by_name[name] = &layer_info;
  for (const auto succs_node : succs_nodes)
  {
    const auto succs_circle_node = loco::must_cast<luci::CircleNode *>(succs_node);
    name = succs_circle_node->name();

    info_by_name[name] = &layer_info;
  }
}

} // namespace

LayerInfoMap layer_info_map(loco::Graph *g, std::vector<LayerInfo> &layers_info)
{
  LayerInfoMap info_by_name;

  for (auto &&info : layers_info)
  {
    auto name = info.name;
    bool found = false;
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      if (cnode->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
        continue;

      if (cnode->name() == name)
      {
        // Check and add multiple-output node and its outputs to info_by_name
        if (is_multiple_output_node(cnode))
        {
          add_multi_output_node(info_by_name, info, cnode);
          found = true;
          continue;
        }
        if (auto multi_output = get_multi_output_node(cnode))
        {
          add_multi_output_node(info_by_name, info, multi_output);
          found = true;
          continue;
        }

        if (info_by_name.find(name) != info_by_name.end())
        {
          throw std::runtime_error("Duplicate layer name " + name +
                                   ". Check layer names in the quantization configuration file.");
        }

        info_by_name[name] = &info;
        found = true;
        continue;
      }
    }

    if (not found)
      throw std::runtime_error("No such layer named " + name +
                               ". Check layer names in the quantization configuration file.");
  }

  return info_by_name;
}

} // namespace luci
