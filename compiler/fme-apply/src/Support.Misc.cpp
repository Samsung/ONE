/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Support.Misc.h"

namespace fme_apply
{

void copy_shape(luci::CircleNode *from, luci::CircleNode *to)
{
  if (not from)
    throw std::invalid_argument("from");

  if (not to)
    throw std::invalid_argument("to");

  to->rank(from->rank());
  for (uint32_t i = 0; i < from->rank(); ++i)
  {
    to->dim(i) = from->dim(i);
  }
}

/**
 * It returns given node's input.
 */
loco::Node *get_input(luci::CircleNode *node)
{
  switch (node->opcode())
  {
    case luci::CircleOpcode::CONV_2D:
    {
      auto conv = loco::must_cast<luci::CircleConv2D *>(node);
      return conv->input();
    }
    case luci::CircleOpcode::DEPTHWISE_CONV_2D:
    {
      auto dconv = loco::must_cast<luci::CircleDepthwiseConv2D *>(node);
      return dconv->input();
    }
    case luci::CircleOpcode::FULLY_CONNECTED:
    {
      auto fc = loco::must_cast<luci::CircleFullyConnected *>(node);
      return fc->input();
    }
    case luci::CircleOpcode::GELU:
    {
      auto gelu = loco::must_cast<luci::CircleGelu *>(node);
      return gelu->features();
    }
    case luci::CircleOpcode::LEAKY_RELU:
    {
      auto relu = loco::must_cast<luci::CircleLeakyRelu *>(node);
      return relu->features();
    }
    case luci::CircleOpcode::MAX_POOL_2D:
    {
      auto maxpool = loco::must_cast<luci::CircleMaxPool2D *>(node);
      return maxpool->value();
    }
    case luci::CircleOpcode::PAD:
    {
      auto pad = loco::must_cast<luci::CirclePad *>(node);
      return pad->input();
    }
    case luci::CircleOpcode::RELU:
    {
      auto relu = loco::must_cast<luci::CircleLeakyRelu *>(node);
      return relu->features();
    }
    case luci::CircleOpcode::TRANSPOSE_CONV:
    {
      auto tconv = loco::must_cast<luci::CircleTransposeConv *>(node);
      return tconv->outBackprop();
    }
    default:
    {
      throw std::runtime_error("(get_input) NYI operator: " + node->name());
    }
  }
}

/**
 * It sets given 'input' to node's input.
 */
void set_input(luci::CircleNode *node, luci::CircleCustom *input)
{
  if (input == nullptr)
  {
    throw std::runtime_error("Invalid input.");
  }

  switch (node->opcode())
  {
    case luci::CircleOpcode::CONV_2D:
    {
      auto conv = loco::must_cast<luci::CircleConv2D *>(node);
      conv->input(input);
      break;
    }
    case luci::CircleOpcode::DEPTHWISE_CONV_2D:
    {
      auto dconv = loco::must_cast<luci::CircleDepthwiseConv2D *>(node);
      dconv->input(input);
      break;
    }
    case luci::CircleOpcode::FULLY_CONNECTED:
    {
      auto fc = loco::must_cast<luci::CircleFullyConnected *>(node);
      fc->input(input);
      break;
    }
    case luci::CircleOpcode::GELU:
    {
      auto gelu = loco::must_cast<luci::CircleGelu *>(node);
      gelu->features(input);
      break;
    }
    case luci::CircleOpcode::LEAKY_RELU:
    {
      auto relu = loco::must_cast<luci::CircleLeakyRelu *>(node);
      relu->features(input);
      break;
    }
    case luci::CircleOpcode::MAX_POOL_2D:
    {
      auto maxpool = loco::must_cast<luci::CircleMaxPool2D *>(node);
      maxpool->value(input);
      break;
    }
    case luci::CircleOpcode::PAD:
    {
      auto pad = loco::must_cast<luci::CirclePad *>(node);
      pad->input(input);
      break;
    }
    case luci::CircleOpcode::RELU:
    {
      auto relu = loco::must_cast<luci::CircleLeakyRelu *>(node);
      relu->features(input);
      break;
    }
    case luci::CircleOpcode::TRANSPOSE_CONV:
    {
      auto tconv = loco::must_cast<luci::CircleTransposeConv *>(node);
      tconv->outBackprop(input);
      break;
    }
    default:
    {
      throw std::runtime_error("(set_input) NYI operator: " + node->name());
    }
  }
}

/**
 * It returns one of given node's arguments whose name is "name".
 *
 * According to the depth, it finds from more preceded nodes.
 */
luci::CircleNode *find_arg_with_name(const luci::CircleNode *node, const std::string &name,
                                     const uint32_t &depth)
{
  if (depth == 0)
    return nullptr;

  const auto arity = node->arity();
  for (uint32_t idx = 0; idx < arity; idx++)
  {
    auto front_node = loco::must_cast<luci::CircleNode *>(node->arg(idx));
    if (front_node->name() == name)
      return front_node;
    front_node = find_arg_with_name(front_node, name, depth - 1);
    if (front_node)
      return front_node;
  }
  return nullptr;
}

} // namespace fme_apply
