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

#include "luci/Pass/FoldFullyConnectedPass.h"

#include "helpers/Compute.h"
#include "helpers/Shape.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>

#include <luci/Log.h>

#include <luci_compute/FullyConnected.h>

#include <cassert>

namespace luci
{

namespace
{

bool set_params(const luci::CircleFullyConnected *node, compute::FullyConnected &cfc)
{
  assert(node);

  LOGGER(l);

  // NOTE only support default for now
  if (node->weights_format() != luci::CircleFullyConnected::WeightsFormat::DEFAULT)
  {
    WARN(l) << "FoldFullyConnectedPass unsupported weights_format: "
            << uint32_t(node->weights_format());
    return false;
  }
  cfc.params().weights_format = compute::FullyConnectedWeightsFormat::kDefault;

  compute::FusedActFunc fac;
  if (!to_compute(node->fusedActivationFunction(), fac))
  {
    WARN(l) << "FoldFullyConnectedPass unsupported activation: "
            << uint32_t(node->fusedActivationFunction());
    return false;
  }
  cfc.fused_act_func(fac);

  return true;
}

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

/**
 * Fold FullyConnected with constant input and filter into a constant tensor
 *
 *    BEFORE
 *
 *    [CircleConst] [CircleConst]
 *               |   |
 *       [CircleFullyConnected]
 *
 *    AFTER
 *
 *           [CircleConst]
 */
bool fold_fully_connected(luci::CircleFullyConnected *node)
{
  RETURN_FALSE_UNLESS(node != nullptr);

  auto const input = dynamic_cast<luci::CircleConst *>(node->input());
  auto const weights = dynamic_cast<luci::CircleConst *>(node->weights());
  auto const bias = dynamic_cast<luci::CircleConst *>(node->bias());
  auto const no_bias = dynamic_cast<luci::CircleOutputExclude *>(node->bias());

  RETURN_FALSE_UNLESS(input != nullptr);
  RETURN_FALSE_UNLESS(weights != nullptr);
  RETURN_FALSE_UNLESS(bias != nullptr or no_bias != nullptr);

  RETURN_FALSE_UNLESS(input->dtype() == loco::DataType::FLOAT32);
  RETURN_FALSE_UNLESS(weights->dtype() == loco::DataType::FLOAT32);

  auto const input_data = &input->at<loco::DataType::FLOAT32>(0);
  auto const weights_data = &weights->at<loco::DataType::FLOAT32>(0);
  float *bias_data = nullptr;
  if (bias)
  {
    RETURN_FALSE_UNLESS(bias->dtype() == loco::DataType::FLOAT32);
    bias_data = &bias->at<loco::DataType::FLOAT32>(0);
  }

  auto static_shape = [](luci::CircleNode *node) {
    loco::TensorShape shape;
    if (not node)
      return shape;
    shape.rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
      shape.dim(i) = node->dim(i);
    return shape;
  };

  compute::FullyConnected comp_fc{};
  if (!set_params(node, comp_fc))
    return false;
  comp_fc.input(static_shape(input), input_data);
  comp_fc.weights(static_shape(weights), weights_data);
  comp_fc.bias(static_shape(bias), bias_data);

  comp_fc.keep_num_dims(node->keep_num_dims());

  if (!comp_fc.prepare())
    return false;

  const auto &output_shape = comp_fc.output_shape();
  assert(is_same_shape(node, output_shape));
  auto output_size = loco::element_count(&output_shape);

  auto constant = node->graph()->nodes()->create<luci::CircleConst>();
  {
    constant->dtype(node->dtype());
    constant->rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
      constant->dim(i).set(node->dim(i).value());
    constant->shape_status(luci::ShapeStatus::VALID);
    constant->size<loco::DataType::FLOAT32>(output_size);
    constant->name(node->name());
  }
  auto constant_data = &constant->at<loco::DataType::FLOAT32>(0);
  comp_fc.output(constant_data);
  comp_fc.compute();

  loco::replace(node).with(constant);

  return true;
}

} // namespace

/**
 * Constant Folding for FullyConnected Op
 **/
bool FoldFullyConnectedPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<CircleFullyConnected *>(node);

    if (fold_fully_connected(fc))
      changed = true;
  }

  return changed;
}

} // namespace luci

#undef RETURN_FALSE_UNLESS
