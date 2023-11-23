/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldDepthwiseConv2DPass.h"

#include "helpers/Compute.h"
#include "helpers/Shape.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>

#include <luci/Log.h>

#include <luci_compute/DepthwiseConv2D.h>

#include <cassert>

namespace luci
{

namespace
{

bool set_params(const luci::CircleDepthwiseConv2D *node, compute::DepthwiseConv2D &cdc)
{
  assert(node);

  LOGGER(l);

  auto &params = cdc.params();
  if (!to_compute(node->padding(), params.padding_type))
  {
    WARN(l) << "FoldDepthwiseConv2DPass unsupported padding: " << uint32_t(node->padding());
    return false;
  }

  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.depth_multiplier = node->depthMultiplier();

  compute::FusedActFunc fac;
  if (!to_compute(node->fusedActivationFunction(), fac))
  {
    WARN(l) << "FoldDepthwiseConv2DPass unsupported activation: "
            << uint32_t(node->fusedActivationFunction());
    return false;
  }
  cdc.fused_act_func(fac);

  return true;
}

/**
 * Fold DepthwiseConv2D with constant input and filter into a constant tensor
 *
 *    BEFORE
 *
 *    [CircleConst] [CircleConst]
 *               |   |
 *       [CircleDepthwiseConv2D]
 *
 *    AFTER
 *
 *           [CircleConst]
 */
bool fold_depthwise_conv_2d(luci::CircleDepthwiseConv2D *node)
{
  auto const input = dynamic_cast<luci::CircleConst *>(node->input());
  if (input == nullptr)
    return false; // Constant input is required for folding
  if (input->rank() != 4)
    return false;

  // filter format: [1, H, W, O]
  auto const filter = dynamic_cast<luci::CircleConst *>(node->filter());
  if (filter == nullptr)
    return false; // Constant filter is required for folding
  if (filter->rank() != 4)
    return false;
  if (filter->dim(0).value() != 1)
    return false; // Unsupported batch size

  // TODO support nullptr bias as it is optional
  auto const bias = dynamic_cast<luci::CircleConst *>(node->bias());
  if (bias == nullptr)
    return false;
  if (bias->rank() != 1)
    return false;

  auto static_shape = [](luci::CircleNode *node) {
    loco::TensorShape shape;
    shape.rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
      shape.dim(i) = node->dim(i);
    return shape;
  };

  auto const input_data = &input->at<loco::DataType::FLOAT32>(0);
  auto const filter_data = &filter->at<loco::DataType::FLOAT32>(0);
  auto const bias_data = &bias->at<loco::DataType::FLOAT32>(0);

  compute::DepthwiseConv2D comp_dwconv2d{};
  if (!set_params(node, comp_dwconv2d))
    return false;
  comp_dwconv2d.input(static_shape(input), input_data);
  comp_dwconv2d.filter(static_shape(filter), filter_data);
  comp_dwconv2d.bias(static_shape(bias), bias_data);

  if (!comp_dwconv2d.prepare())
    return false;

  auto output_shape = comp_dwconv2d.output_shape();
  assert(is_same_shape(node, output_shape));
  auto output_size = loco::element_count(&output_shape);

  // result folded constant node
  auto constant = node->graph()->nodes()->create<luci::CircleConst>();
  constant->dtype(node->dtype());
  constant->rank(node->rank());
  for (uint32_t i = 0; i < output_shape.rank(); ++i)
    constant->dim(i).set(output_shape.dim(i).value());
  constant->shape_status(luci::ShapeStatus::VALID);
  constant->size<loco::DataType::FLOAT32>(output_size);
  constant->name(node->name());

  auto constant_data = &constant->at<loco::DataType::FLOAT32>(0);
  comp_dwconv2d.output(constant_data);
  comp_dwconv2d.compute();

  loco::replace(node).with(constant);

  return true;
}

} // namespace

/**
 * Constant Folding for DepthwiseConv2D Op
 **/
bool FoldDepthwiseConv2DPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto depthwise_conv2d = dynamic_cast<CircleDepthwiseConv2D *>(node);

    if (depthwise_conv2d == nullptr)
      continue;

    switch (depthwise_conv2d->dtype())
    {
      case loco::DataType::FLOAT32:
        changed = fold_depthwise_conv_2d(depthwise_conv2d);
        break;
      default:
        break;
    }
  }

  return changed;
}

} // namespace luci
