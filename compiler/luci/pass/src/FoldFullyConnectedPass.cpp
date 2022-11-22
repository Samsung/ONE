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

#include <tensorflow/lite/kernels/internal/reference/fully_connected.h>

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>

#include <luci/Log.h>

#include <limits> // std::numeric_limits

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

namespace luci
{

/**
 * Constant Folding for FullyConnected Op
 **/
bool FoldFullyConnectedPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<CircleFullyConnected *>(node);

    if (fc == nullptr)
      continue;

    if (fc->dtype() != loco::DataType::FLOAT32)
      continue;

    changed = fold_fully_connected(fc);
  }

  return changed;
}

} // namespace luci

#undef RETURN_FALSE_UNLESS
