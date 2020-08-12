/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "luci/Pass/RequantizePass.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <oops/UserExn.h>

#include <iostream>
#include <cmath>

namespace luci
{

namespace
{

// Check if the node is the bias of Conv2D, DepthwiseConv2D, or FullyConnected layer
// If true, return <input, weight> pair of the successor node (used to quantize bias)
// If false, return <nullptr, nullptr>
std::pair<loco::Node *, loco::Node *> get_input_weight_of_bias(CircleNode *node)
{
  auto circle_const = dynamic_cast<CircleConst *>(node);
  if (circle_const == nullptr)
    return std::make_pair(nullptr, nullptr);

  auto succs = loco::succs(node);
  if (succs.size() != 1) // assume bias is used by only one node
    return std::make_pair(nullptr, nullptr);

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->bias() == circle_const)
    {
      assert(conv->input() != nullptr);
      assert(conv->filter() != nullptr);
      return std::make_pair(conv->input(), conv->filter());
    }
    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->bias() == circle_const)
    {
      assert(dw_conv->input() != nullptr);
      assert(dw_conv->filter() != nullptr);
      return std::make_pair(dw_conv->input(), dw_conv->filter());
    }
    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->bias() == circle_const)
    {
      assert(fc->input() != nullptr);
      assert(fc->weights() != nullptr);
      return std::make_pair(fc->input(), fc->weights());
    }
    // TODO: add TransposeConv when bias is supported in CircleTransposeConv
  }
  return std::make_pair(nullptr, nullptr);
}

// Requantize weights from symmetric int8 to asymmetric uint8
// Original: -127 ~ 127 zp = 0
// After requantization: 1 ~ 255 zp = 128
void wrequant_int8_to_uint8(CircleConst *node)
{
  assert(node->dtype() == loco::DataType::S8);

  uint32_t size = node->size<loco::DataType::S8>();
  std::vector<int32_t> requantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    int32_t data = node->at<loco::DataType::S8>(i);
    requantized_values[i] = data + 128;
  }

  node->dtype(loco::DataType::U8); // change the type of tensor
  node->size<loco::DataType::U8>(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    assert(0 <= requantized_values[i] && requantized_values[i] <= 255);
    node->at<loco::DataType::U8>(i) = requantized_values[i];
  }
}

// Check if node is weights of conv2d, depthwise_conv2d, transpose_conv, or fully_connected layer
bool is_weights(CircleNode *node)
{
  auto circle_const = dynamic_cast<CircleConst *>(node);
  if (circle_const == nullptr)
    return false;

  auto succs = loco::succs(node);
  if (succs.size() != 1) // assume weights is used by only one node
    return false;

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->filter() == circle_const)
      return true;

    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->filter() == circle_const)
      return true;

    auto t_conv = dynamic_cast<CircleTransposeConv *>(out);
    if (t_conv != nullptr && t_conv->filter() == circle_const && circle_const->rank() == 4)
      return true;

    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->weights() == circle_const)
      return true;
  }
  return false;
}

/**
 * @brief RequantizeActivation requantizes tensors for activations
 */
struct RequantizeActivation final : public luci::CircleNodeMutableVisitor<bool>
{
  RequantizeActivation(loco::DataType input, loco::DataType output)
      : _input_type(input), _output_type(output)
  {
  }

  loco::DataType _input_type;
  loco::DataType _output_type;

  // Requantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "RequantizeActivation visit node: " << node->name() << std::endl;
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->arg(i);
      auto circle_node = loco::must_cast<luci::CircleNode *>(input_node);

      // Check if this was already requantized
      if (circle_node->dtype() == _output_type)
        continue;

      // Check if this was not quantized (only quantized tensors are requantized)
      if (circle_node->quantparam() == nullptr)
        continue;

      if (_input_type == loco::DataType::S8 && _output_type == loco::DataType::U8)
      {
        // Check if this is not bias (bias is not requantized when int8 -> uint8)
        auto iw = get_input_weight_of_bias(circle_node);
        if (iw.first != nullptr && iw.second != nullptr)
          continue;

        // Check if this is not weights (weights are requantized in RequantizeWeights)
        if (!is_weights(circle_node))
        {
          auto quantparam = circle_node->quantparam();
          assert(quantparam->zerop.size() == 1);
          quantparam->zerop[0] += 128;
          circle_node->dtype(loco::DataType::U8);
        }
      }
    }
    return false;
  }
};

/**
 * @brief RequantizeWeights requantizes tensors for weights
 */
struct RequantizeWeights final : public luci::CircleNodeMutableVisitor<bool>
{
  RequantizeWeights(loco::DataType input, loco::DataType output)
      : _input_type(input), _output_type(output)
  {
  }

  loco::DataType _input_type;
  loco::DataType _output_type;

  // Requantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "RequantizeWeights visit node: " << node->name() << std::endl;
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->arg(i);
      auto circle_node = loco::must_cast<luci::CircleNode *>(input_node);

      if (circle_node->dtype() == _output_type) // already requantized
        continue;

      if (is_weights(circle_node))
      {
        auto circle_const = loco::must_cast<luci::CircleConst *>(circle_node);

        // Requantize i8 symmetric -> uint8 asymmetric
        if (_input_type == loco::DataType::S8 && _output_type == loco::DataType::U8)
        {
          wrequant_int8_to_uint8(circle_const);

          auto quantparam = circle_node->quantparam();
          assert(quantparam != nullptr);
          for (size_t i = 0; i < quantparam->zerop.size(); ++i)
          {
            assert(quantparam->zerop[i] == 0);
            quantparam->zerop[i] = 128;
          }
          circle_node->dtype(loco::DataType::U8);
        }
      }
    }
    return false;
  }
};

} // namespace

bool RequantizePass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "RequantizePass Start" << std::endl;

  // Requantize activation
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    RequantizeActivation rqa(_input_dtype, _output_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&rqa);
  }

  // Requantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    RequantizeWeights rqw(_input_dtype, _output_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&rqw);
  }

  // Update output dtype
  auto graph_outputs = g->outputs();
  for (auto node : loco::output_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleOutput *>(node);
    if (static_cast<luci::CircleNode *>(circle_node->from())->dtype() == _output_dtype)
    {
      circle_node->dtype(_output_dtype);
      auto graph_output = graph_outputs->at(circle_node->index());
      graph_output->dtype(_output_dtype);
    }
  }

  INFO(l) << "RequantizePass End" << std::endl;
  return false; // one time run
}

} // namespace luci
