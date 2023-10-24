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

#include "luci/Pass/FuseHorizontalFCLayers.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
bool check_type_and_shape_equality(const luci::CircleNode *left, const luci::CircleNode *right)
{
  if (left->dtype() != right->dtype())
    return false;

  if (left->rank() != right->rank())
    return false;

  for (uint32_t i = 0; i < left->rank(); ++i)
  {
    if (left->dim(i).value() != right->dim(i).value())
      return false;
  }

  return true;
}

// Add right const to left const (left is updated)
template <loco::DataType D>
void sum_const_values(luci::CircleConst *left, const luci::CircleConst *right)
{
  assert(check_type_and_shape_equality(left, right)); // FIX CALLER UNLESS
  const auto size = left->template size<D>();

  for (uint32_t i = 0; i < size; ++i)
  {
    left->template at<D>(i) += right->template at<D>(i);
  }
}

bool fuse_horizontal_fc_nodes(luci::CircleAdd *add_node)
{
  // Let's check left and right FC nodes
  auto left_fc_node = dynamic_cast<luci::CircleFullyConnected *>(add_node->x());
  auto right_fc_node = dynamic_cast<luci::CircleFullyConnected *>(add_node->y());

  if (left_fc_node == nullptr or right_fc_node == nullptr)
    return false;

  if (not check_type_and_shape_equality(left_fc_node, right_fc_node))
    return false;

  if (left_fc_node->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  if (right_fc_node->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  // Let's check that FC nodes have the same input
  if (left_fc_node->input() != right_fc_node->input())
    return false;

  // Lets check left and right FC weights: type and shape
  auto left_fc_weights = dynamic_cast<luci::CircleConst *>(left_fc_node->weights());
  auto right_fc_weights = dynamic_cast<luci::CircleConst *>(right_fc_node->weights());

  if (left_fc_weights == nullptr or right_fc_weights == nullptr)
    return false;

  if (not check_type_and_shape_equality(left_fc_weights, right_fc_weights))
    return false;

  // Lets check left and right FC bias: type and shape
  auto left_fc_bias = dynamic_cast<luci::CircleConst *>(left_fc_node->bias());
  auto right_fc_bias = dynamic_cast<luci::CircleConst *>(right_fc_node->bias());

  // Support only if both biases are const, or both are non-const
  // TODO Support the case that one FC has a const bias and another FC has no bias.
  if ((left_fc_bias == nullptr and right_fc_bias != nullptr) or
      (left_fc_bias != nullptr and right_fc_bias == nullptr))
  {
    return false;
  }

  // Both left/right bias are const. Check dtype/shape.
  if (left_fc_bias != nullptr and not check_type_and_shape_equality(left_fc_bias, right_fc_bias))
    return false;

  // Both left/right bias are non-const. Check left/right fc has no bias.
  if (left_fc_bias == nullptr)
  {
    auto left_no_bias = dynamic_cast<luci::CircleOutputExclude *>(left_fc_node->bias());
    auto right_no_bias = dynamic_cast<luci::CircleOutputExclude *>(right_fc_node->bias());
    if (not left_no_bias or not right_no_bias)
      return false;
  }

  // Lets create fused FC weights and bias
  auto fused_fc_weights = luci::clone(left_fc_weights);
  luci::add_origin(fused_fc_weights, luci::composite_origin({luci::get_origin(left_fc_weights),
                                                             luci::get_origin(right_fc_weights)}));

  luci::CircleConst *fused_fc_bias = nullptr;
  if (left_fc_bias != nullptr)
  {
    fused_fc_bias = luci::clone(left_fc_bias);
    luci::add_origin(fused_fc_bias, luci::composite_origin({luci::get_origin(left_fc_bias),
                                                            luci::get_origin(right_fc_bias)}));
  }

  switch (left_fc_weights->dtype())
  {
    case loco::DataType::FLOAT32:
      sum_const_values<loco::DataType::FLOAT32>(fused_fc_weights, right_fc_weights);
      break;
    default:
      return false;
  }

  if (fused_fc_bias != nullptr)
  {
    switch (left_fc_bias->dtype())
    {
      case loco::DataType::FLOAT32:
        sum_const_values<loco::DataType::FLOAT32>(fused_fc_bias, right_fc_bias);
        break;
      default:
        return false;
    }
  }

  // Create fused FC node
  auto graph = left_fc_node->graph();
  auto fused_fc_node = graph->nodes()->create<luci::CircleFullyConnected>();
  fused_fc_node->input(left_fc_node->input());
  fused_fc_node->weights(fused_fc_weights);
  if (fused_fc_bias)
  {
    fused_fc_node->bias(fused_fc_bias);
  }
  else
  {
    assert(nullptr !=
           dynamic_cast<luci::CircleOutputExclude *>(left_fc_node->bias())); // FIX ME UNLESS
    fused_fc_node->bias(left_fc_node->bias());
  }

  fused_fc_node->fusedActivationFunction(luci::FusedActFunc::NONE);
  fused_fc_node->name(left_fc_node->name() + "_" + right_fc_node->name() + "_fused");

  luci::add_origin(fused_fc_node, luci::composite_origin({luci::get_origin(left_fc_node),
                                                          luci::get_origin(right_fc_node),
                                                          luci::get_origin(add_node)}));

  replace(add_node).with(fused_fc_node);

  return true;
}

} // namespace

namespace luci
{

bool FuseHorizontalFullyConnectedPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add_node = dynamic_cast<luci::CircleAdd *>(node);
    if (not add_node)
      continue;

    if (fuse_horizontal_fc_nodes(add_node))
      changed = true;
  }

  return changed;
}

} // namespace luci
