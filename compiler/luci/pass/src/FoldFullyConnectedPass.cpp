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

#if 0

namespace
{

bool set_kernel_parameters(tflite::FullyConnectedParams *params, luci::CircleFullyConnected *node)
{
  switch (node->fusedActivationFunction())
  {
    case luci::FusedActFunc::NONE:
    case luci::FusedActFunc::TANH:
      params->float_activation_min = std::numeric_limits<float>::lowest();
      params->float_activation_max = std::numeric_limits<float>::max();
      break;
    case luci::FusedActFunc::RELU:
      params->float_activation_min = 0;
      params->float_activation_max = std::numeric_limits<float>::max();
      break;
    case luci::FusedActFunc::RELU_N1_TO_1:
      params->float_activation_min = -1;
      params->float_activation_max = 1;
      break;
    case luci::FusedActFunc::RELU6:
      params->float_activation_min = 0;
      params->float_activation_max = 6;
      break;
    default:
    {
      LOGGER(l);
      WARN(l) << "Unsupported activation: " << uint32_t(node->fusedActivationFunction());
      return false;
    }
  }

  assert(node->weights_format() ==
         luci::CircleFullyConnected::WeightsFormat::DEFAULT); // FIX_CALLER_UNLESS
  params->weights_format = tflite::FullyConnectedWeightsFormat::kDefault;

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

  LOGGER(l);

  auto const input = dynamic_cast<luci::CircleConst *>(node->input());
  auto const weights = dynamic_cast<luci::CircleConst *>(node->weights());
  auto const bias = dynamic_cast<luci::CircleConst *>(node->bias());
  auto const no_bias = dynamic_cast<luci::CircleOutputExclude *>(node->bias());

  RETURN_FALSE_UNLESS(input != nullptr);
  RETURN_FALSE_UNLESS(weights != nullptr);
  RETURN_FALSE_UNLESS(node->weights_format() == luci::CircleFullyConnected::WeightsFormat::DEFAULT);
  RETURN_FALSE_UNLESS(bias != nullptr or no_bias != nullptr);

  RETURN_FALSE_UNLESS(input->dtype() == loco::DataType::FLOAT32);
  RETURN_FALSE_UNLESS(weights->dtype() == loco::DataType::FLOAT32);
  if (bias)
    RETURN_FALSE_UNLESS(bias->dtype() == loco::DataType::FLOAT32);

  auto const input_elems = input->size<loco::DataType::FLOAT32>();

  RETURN_FALSE_UNLESS(weights->rank() == 2);
  RETURN_FALSE_UNLESS(input_elems % weights->dim(1).value() == 0);
  auto const batch_size = input_elems / weights->dim(1).value();
  auto const num_units = weights->dim(0).value();

  if (bias)
    RETURN_FALSE_UNLESS(bias->size<loco::DataType::FLOAT32>() == num_units);

  tflite::FullyConnectedParams params{};
  if (!set_kernel_parameters(&params, node))
    return false; // Unsupported kernel parameter values

  std::vector<uint32_t> output_shape;
  if (node->keep_num_dims() == false)
  {
    output_shape.push_back(batch_size);
    output_shape.push_back(num_units);
  }
  else
  {
    output_shape.resize(input->rank());
    for (uint32_t i = 0; i < input->rank(); i++)
      output_shape[i] = input->dim(i).value();
    output_shape[input->rank() - 1] = num_units;
  }

  auto constant = node->graph()->nodes()->create<luci::CircleConst>();
  {
    constant->name(node->name());
    constant->dtype(node->dtype());
    constant->rank(node->rank());
    constant->shape_status(luci::ShapeStatus::VALID);
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < node->rank(); ++i)
    {
      constant->dim(i).set(node->dim(i).value());
      num_elem *= node->dim(i).value();
    }
    constant->size<loco::DataType::FLOAT32>(num_elem);
  }

  auto tensor_shape = [](luci::CircleNode *node) {
    if (node == nullptr)
      return tflite::RuntimeShape();

    tflite::RuntimeShape runtime_shape(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
      runtime_shape.SetDim(i, node->dim(i).value());
    return runtime_shape;
  };

  auto tensor_data = [](luci::CircleConst *node) -> float * {
    if (node == nullptr)
      return nullptr;

    return &node->at<loco::DataType::FLOAT32>(0);
  };

  tflite::reference_ops::FullyConnected(
    params, tensor_shape(input), tensor_data(input), tensor_shape(weights), tensor_data(weights),
    tensor_shape(bias), tensor_data(bias), tensor_shape(constant), tensor_data(constant));

  loco::replace(node).with(constant);

  return true;
}

} // namespace

#endif // 0

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

  auto output_shape = comp_fc.output_shape();
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
