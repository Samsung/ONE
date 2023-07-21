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

// TODO Remove unnecessary code
#if 0
// Check if the node is the bias of Conv2D, DepthwiseConv2D, or FullyConnected layer
bool is_bias(CircleConst *node)
{
  if (node == nullptr)
    return false;

  auto succs = loco::succs(node);
  if (succs.size() != 1) // assume bias is used by only one node
    return false;

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->bias() == node)
      return true;

    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->bias() == node)
      return true;

    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->bias() == node)
      return true;

    auto tconv = dynamic_cast<CircleTransposeConv *>(out);
    if (tconv != nullptr && tconv->bias() == node)
      return true;
  }
  return false;
}
#endif

void requant_nonconst_int8_to_uint8(CircleNode *circle_node)
{
  assert(circle_node->dtype() == loco::DataType::S8);

  auto quantparam = circle_node->quantparam();
  assert(quantparam != nullptr);
  for (size_t i = 0; i < quantparam->zerop.size(); ++i)
  {
    quantparam->zerop[i] += 128;
  }
  circle_node->dtype(loco::DataType::U8);
}

// Requantize CircleConst from symmetric int8 to asymmetric uint8
// Original values: -127 ~ 127
// After requantization: 1 ~ 255 (zp <- zp + 128)
void requant_const_int8_to_uint8(CircleConst *node)
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
    assert(1 <= requantized_values[i] && requantized_values[i] <= 255);
    node->at<loco::DataType::U8>(i) = requantized_values[i];
  }

  auto quantparam = node->quantparam();
  assert(quantparam != nullptr);
  for (size_t i = 0; i < quantparam->zerop.size(); ++i)
  {
    quantparam->zerop[i] += 128;
  }
}

#define RETURN_UNLESS(cond) \
  if (not(cond))            \
    return;

/**
 * @brief Requantize int8 quantized tensors to uint8 tensors
 */
struct RequantizeS8ToU8 final : public luci::CircleNodeMutableVisitor<void>
{
  // Requantize non-const tensors
  void visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "RequantizeS8ToU8 visit non-const node: " << node->name() << std::endl;

    // Ignore non-quantized tensors
    RETURN_UNLESS(node->quantparam() != nullptr);

    // Check dtype is int8
    RETURN_UNLESS(node->dtype() == loco::DataType::S8);

    requant_nonconst_int8_to_uint8(node);
  }

  // Requantize const tensors
  void visit(luci::CircleConst *node)
  {
    LOGGER(l);
    INFO(l) << "RequantizeS8ToU8 visit const node: " << node->name() << std::endl;

    // Ignore non-quantized tensors
    RETURN_UNLESS(node->quantparam() != nullptr);

    // Check dtype is int8
    RETURN_UNLESS(node->dtype() == loco::DataType::S8);

    requant_const_int8_to_uint8(node);
  }
};

#undef RETURN_UNLESS

// TODO Remove unnecessary code
#if 0
/**
 * @brief RequantizeNonConst requantizes tensors for activations
 */
struct RequantizeNonConst final : public luci::CircleNodeMutableVisitor<bool>
{
  RequantizeNonConst(loco::DataType input, loco::DataType output)
    : _input_type(input), _output_type(output)
  {
  }

  loco::DataType _input_type;
  loco::DataType _output_type;

  // Requantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "RequantizeNonConst visit node: " << node->name() << std::endl;
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->arg(i);
      auto circle_node = loco::must_cast<luci::CircleNode *>(input_node);

      // Check if this was quantized (only quantized tensors are requantized)
      if (circle_node->quantparam() == nullptr)
        continue;

      // Check if this is already requantized
      if (circle_node->dtype() == _output_type)
        continue;

      // Check if this is not const (only non-const is requantized in this function)
      auto circle_const = dynamic_cast<CircleConst *>(circle_node);
      if (circle_const != nullptr)
        continue;

      if (_input_type == loco::DataType::S8 && _output_type == loco::DataType::U8)
        requant_nonconst_int8_to_uint8(circle_node);
    }
    return false;
  }
};

/**
 * @brief RequantizeConst requantizes tensors for weights
 */
struct RequantizeConst final : public luci::CircleNodeMutableVisitor<bool>
{
  RequantizeConst(loco::DataType input, loco::DataType output)
    : _input_type(input), _output_type(output)
  {
  }

  loco::DataType _input_type;
  loco::DataType _output_type;

  // Requantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "RequantizeConst visit node: " << node->name() << std::endl;
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->arg(i);
      auto circle_node = loco::must_cast<luci::CircleNode *>(input_node);

      // Check if this was quantized (only quantized tensors are requantized)
      if (circle_node->quantparam() == nullptr)
        continue;

      // Check if this is already requantized
      if (circle_node->dtype() == _output_type)
        continue;

      // Check if this is const (only const is requantized in this function)
      auto circle_const = dynamic_cast<CircleConst *>(circle_node);
      if (circle_const == nullptr)
        continue;

      // Check if this is not bias
      // bias is not requantized when int8 -> uint8
      if (is_bias(circle_const))
        continue;

      if (_input_type == loco::DataType::S8 && _output_type == loco::DataType::U8)
        requant_const_int8_to_uint8(circle_const);
    }
    return false;
  }
};
#endif

} // namespace

bool RequantizePass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "RequantizePass Start" << std::endl;

  // Input: int8 model
  // Output: uint8 model
  if (_input_dtype == loco::DataType::S8 and _output_dtype == loco::DataType::U8)
  {
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      RequantizeS8ToU8 rq;
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      circle_node->accept(&rq);
    }
  }
  else
  {
    // Ignore other cases
    return false;
  }

// TODO Remove unnecessary code
#if 0
  // Requantize non-const (activations)
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    RequantizeNonConst rqnc(_input_dtype, _output_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&rqnc);
  }

  // Requantize const (including weights, constants)
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    RequantizeConst rqc(_input_dtype, _output_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&rqc);
  }
#endif

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
