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
    // The following nodes have multiple outputs. Output tensors are not produced by themselves but
    // by the corresponding *Out nodes.
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::SPLIT_V:
    case luci::CircleOpcode::TOPK_V2:
    case luci::CircleOpcode::UNIQUE:
    case luci::CircleOpcode::UNPACK:
      return true;
    // TODO: Support ops
    case luci::CircleOpcode::BIDIRECTIONAL_SEQUENCE_LSTM:
    case luci::CircleOpcode::CUSTOM:
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::NON_MAX_SUPPRESSION_V4:
    case luci::CircleOpcode::NON_MAX_SUPPRESSION_V5:
    case luci::CircleOpcode::WHILE:
      throw std::runtime_error("Unsupported op now");
    default:
      return false;
  }
}

const luci::CircleNode *get_multi_output_node(const luci::CircleNode *node)
{
  if (is_multiple_output_node(node))
    return node;

  switch (node->opcode())
  {
    // The following nodes denote outputs of multiple-output nodes.
    case luci::CircleOpcode::CIRCLESPLITOUT:
    {
      const auto split_out = loco::must_cast<const CircleSplitOut *>(node);
      return loco::must_cast<luci::CircleNode *>(split_out->input());
    }
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    {
      const auto splitv_out = loco::must_cast<const CircleSplitVOut *>(node);
      return loco::must_cast<luci::CircleNode *>(splitv_out->input());
    }
    case luci::CircleOpcode::CIRCLETOPKV2OUT:
    {
      const auto top_kv2_out = loco::must_cast<const CircleTopKV2Out *>(node);
      return loco::must_cast<luci::CircleNode *>(top_kv2_out->input());
    }
    case luci::CircleOpcode::CIRCLEUNIQUEOUT:
    {
      const auto unique_out = loco::must_cast<const CircleUniqueOut *>(node);
      return loco::must_cast<luci::CircleNode *>(unique_out->input());
    }
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    {
      const auto unpack_out = loco::must_cast<const CircleUnpackOut *>(node);
      return loco::must_cast<luci::CircleNode *>(unpack_out->input());
    }
    // TODO: Support these ops
    case luci::CircleOpcode::CIRCLEBIDIRECTIONAL_SEQUENCE_LSTM_OUT:
    case luci::CircleOpcode::CIRCLECUSTOMOUT:
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV4OUT:
    case luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV5OUT:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      throw std::runtime_error("Unsupported op now");
    default:
      return nullptr;
  }
}

bool same_setting(const LayerInfo &left, const LayerInfo &right)
{
  return left.dtype == right.dtype and left.granularity == right.granularity;
}

void add_multi_output_node(LayerInfoMap &info_by_name, LayerInfo &layer_info,
                           const luci::CircleNode *node)
{
  assert(is_multiple_output_node(node)); // FIX_CALLER_UNLESS

  const auto succs_nodes = loco::succs(node);
  const auto name = node->name();

  if (info_by_name.find(name) != info_by_name.end())
  {
    // Check that all outputs have equal dtype and granularity
    for (const auto succs_node : succs_nodes)
    {
      const auto succs_circle_node = loco::must_cast<luci::CircleNode *>(succs_node);

      const auto it = info_by_name.find(succs_circle_node->name());
      if (it != info_by_name.end() and not same_setting(layer_info, (it->second)))
        throw std::runtime_error("Outputs of multiple-output nodes should have equal dtype and "
                                 "granularity. Check the quantization configuration file");
    }
    return;
  }

  // Add multiple output node to info_by_name
  info_by_name[name] = {name, layer_info.dtype, layer_info.granularity};

  // Add outputs node to info_by_name
  for (const auto succs_node : succs_nodes)
  {
    const auto succs_circle_node = loco::must_cast<luci::CircleNode *>(succs_node);
    const auto succs_circle_node_name = succs_circle_node->name();
    info_by_name[succs_circle_node_name] = {succs_circle_node_name, layer_info.dtype,
                                            layer_info.granularity};
  }
}

} // namespace

LayerInfoMap layer_info_map(loco::Graph *g, std::vector<LayerInfo> &layers_info)
{
  LayerInfoMap info_by_name;

  for (auto &&info : layers_info)
  {
    auto &name = info.name;
    bool found = false;
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      if (cnode->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
        continue;

      if (cnode->name() == name)
      {
        // Check and add multiple-output node and its outputs to info_by_name
        if (const auto multi_output = get_multi_output_node(cnode))
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

        info_by_name[name] = info;
        found = true;
        continue;
      }
    }

    if (not found)
      throw std::runtime_error("No such layer named " + name +
                               ". Check layer names in the quantization configuration file.");
  }

  // TODO Check all names in layers_info exist in the info_by_name
  // TODO Check names in info_by_name but not in layers_info are from virtual outputs

  return info_by_name;
}

} // namespace luci
