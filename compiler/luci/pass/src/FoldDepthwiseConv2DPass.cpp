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

// TODO remove unused
#if 0

namespace
{

// TODO Share activation mix/max and compute_input/output code with luci-interpreter

bool compute_output(uint32_t *output_size, luci::Padding padding, int32_t image_size,
                    int32_t filter_size, int32_t stride, int32_t dilation_rate)
{
  auto const effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding)
  {
    case luci::Padding::SAME:
      *output_size = (image_size + stride - 1) / stride;
      return true;

    case luci::Padding::VALID:
      *output_size = (image_size + stride - effective_filter_size) / stride;
      return true;

    default:
    {
      LOGGER(l);
      WARN(l) << "Unsupported padding: " << uint32_t(padding);
      return false;
    }
  }
}

uint32_t compute_padding(int32_t stride, int32_t dilation_rate, int32_t in_size,
                         int32_t filter_size, int32_t out_size)
{
  auto const effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  auto const padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

bool set_kernel_parameters(tflite::DepthwiseParams *params, luci::CircleDepthwiseConv2D *node,
                           uint32_t padding_height, uint32_t padding_width)
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

  params->stride_height = node->stride()->h();
  params->stride_width = node->stride()->w();
  params->dilation_height_factor = node->dilation()->h();
  params->dilation_width_factor = node->dilation()->w();
  params->depth_multiplier = node->depthMultiplier();

  params->padding_values.height = padding_height;
  params->padding_values.width = padding_width;

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
  LOGGER(l);

  auto const input = dynamic_cast<luci::CircleConst *>(node->input());

  if (input == nullptr)
    return false; // Constant input is required for folding

  auto const filter = dynamic_cast<luci::CircleConst *>(node->filter());

  if (filter == nullptr)
    return false; // Constant filter is required for folding

  if (filter->dim(0).value() != 1)
    return false; // Unsupported batch size

  auto const bias = dynamic_cast<luci::CircleConst *>(node->bias());

  if (bias == nullptr)
    return false; // Constant bias is required for folding

  auto const input_batches = input->dim(0).value();
  auto const input_height = input->dim(1).value();
  auto const input_width = input->dim(2).value();
  auto const input_depth = input->dim(3).value();

  auto const filter_height = filter->dim(1).value();
  auto const filter_width = filter->dim(2).value();
  auto const filter_channels_out = filter->dim(3).value();

  if (filter_channels_out % input_depth != 0)
    return false; // Wrong input/output depth ratio

  if (node->depthMultiplier() != static_cast<int32_t>(filter_channels_out / input_depth))
    return false; // Wrong depth multiplier value

  if (bias->rank() != 1 || bias->dim(0).value() != filter_channels_out)
    return false; // Unsupported bias value

  uint32_t output_height = 0;
  uint32_t output_width = 0;

  if (!compute_output(&output_height, node->padding(), input_height, filter_height,
                      node->stride()->h(), node->dilation()->h()))
    return false; // Unsupported output parameters

  if (!compute_output(&output_width, node->padding(), input_width, filter_width,
                      node->stride()->w(), node->dilation()->w()))
    return false; // Unsupported output parameters

  auto const padding_height = compute_padding(node->stride()->h(), node->dilation()->h(),
                                              input_height, filter_height, output_height);
  auto const padding_width = compute_padding(node->stride()->w(), node->dilation()->w(),
                                             input_width, filter_width, output_width);

  tflite::DepthwiseParams params{};

  if (!set_kernel_parameters(&params, node, padding_height, padding_width))
    return false; // Unsupported kernel parameter values

  auto constant = node->graph()->nodes()->create<luci::CircleConst>();
  constant->name(node->name());
  constant->dtype(node->dtype());
  constant->rank(node->rank());
  constant->shape_status(luci::ShapeStatus::VALID);
  for (uint32_t i = 0; i < node->rank(); ++i)
    constant->dim(i).set(node->dim(i).value());

  constant->size<loco::DataType::FLOAT32>(input_batches * output_height * output_width *
                                          filter_channels_out);

  auto const input_data = &input->at<loco::DataType::FLOAT32>(0);
  auto const filter_data = &filter->at<loco::DataType::FLOAT32>(0);
  auto const bias_data = &bias->at<loco::DataType::FLOAT32>(0);
  auto const constant_data = &constant->at<loco::DataType::FLOAT32>(0);

  auto tensor_shape = [](luci::CircleNode *node) {
    tflite::RuntimeShape runtime_shape(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
      runtime_shape.SetDim(i, node->dim(i).value());
    return runtime_shape;
  };

  tflite::reference_ops::DepthwiseConv(params, tensor_shape(input), input_data,
                                       tensor_shape(filter), filter_data, tensor_shape(bias),
                                       bias_data, tensor_shape(constant), constant_data);

  loco::replace(node).with(constant);

  return true;
}

#endif // 0

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

  auto const filter = dynamic_cast<luci::CircleConst *>(node->filter());

  if (filter == nullptr)
    return false; // Constant filter is required for folding

  if (filter->dim(0).value() != 1)
    return false; // Unsupported batch size

  auto const bias = dynamic_cast<luci::CircleConst *>(node->bias());

  if (bias == nullptr)
    return false; // Constant bias is required for folding

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
