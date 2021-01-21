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

#include "luci/Pass/FuseBatchNormWithConvPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{
/**
 *  Fuse Mul-Add to Conv2D if possible.
 *
 *  NOTE TF's BatchNormalization is converted to Mul and Add.
 *
 *  BEFORE
 *                  |   [CircleConst]
 *                  |  / [CircleConst]
 *                  | / /
 *         [CircleConv2D] [CircleConst]
 *                  |    /
 *            [CircleMul] [CircleConst]
 *                  |    /
 *             [CircleAdd]
 *                  |
 *
 *  AFTER
 *                  |                  [CircleConst]
 *                  +--------------+  / [CircleConst]
 *                  |              | / /
 *                  |     [CircleConv2D] [CircleConst]
 *  [CircleConst]   |              |    /
 * [CircleConst] \  |         [CircleMul] [CircleConst]
 *              \ \ |              |     /
 *           [CircleConv2D]   [CircleAdd]
 *                  |
 */
bool fused_batch_norm_with_conv(luci::CircleAdd *add)
{
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *shift = nullptr;
  if (auto add_lhs = dynamic_cast<luci::CircleMul *>(add->x()))
  {
    mul = add_lhs;
    shift = dynamic_cast<luci::CircleConst *>(add->y());
  }
  else if (auto add_rhs = dynamic_cast<luci::CircleMul *>(add->y()))
  {
    mul = add_rhs;
    shift = dynamic_cast<luci::CircleConst *>(add->x());
  }

  // If CircleMul is not found or constant operand of CircleAdd is not found,
  // this pass cannot be applied.
  if (mul == nullptr || shift == nullptr)
    return false;

  // If FusedActivationFunction of mul is not none, this pass cannot be applied.
  if (mul->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  // To apply this pass, shape of shift should be [1, 1, 1, out_channel].
  if (shift->rank() != 4)
    return false;
  for (uint32_t i = 0; i < 3; ++i)
    if (shift->dim(i).value() != 1)
      return false;

  luci::CircleConv2D *conv = nullptr;
  luci::CircleConst *scale = nullptr;
  if (auto mul_lhs = dynamic_cast<luci::CircleConv2D *>(mul->x()))
  {
    conv = mul_lhs;
    scale = dynamic_cast<luci::CircleConst *>(mul->y());
  }
  else if (auto mul_rhs = dynamic_cast<luci::CircleConv2D *>(mul->y()))
  {
    conv = mul_rhs;
    scale = dynamic_cast<luci::CircleConst *>(mul->x());
  }

  // If CircleConv2D is not found or constant operand of CircleMul is not found,
  // this pass cannot be applied.
  if (conv == nullptr || scale == nullptr)
    return false;

  // To apply this pass, shape of scale should be [1, 1, 1, out_channel].
  if (scale->rank() != 4)
    return false;
  for (uint32_t i = 0; i < 3; ++i)
    if (scale->dim(i).value() != 1)
      return false;

  // If FusedActivationFunction of conv is not none, this pass cannot be applied.
  if (conv->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleConst *filter = dynamic_cast<luci::CircleConst *>(conv->filter());
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(conv->bias());

  // If filter or bias of conv is not const, this pass cannot be applied.
  if (filter == nullptr || bias == nullptr)
    return false;

  // If dtype of filter is different with scale and shift, multiplication may be impossible.
  if (filter->dtype() != scale->dtype())
    return false;
  if (filter->dtype() != shift->dtype())
    return false;

  // TODO Support more data type
  if (filter->dtype() != loco::DataType::FLOAT32)
    return false;

  // Output channel dimension should be same. If not, this pass cannot be applied.
  if (filter->dim(0).value() != scale->dim(3).value())
    return false;
  if (filter->dim(0).value() != shift->dim(3).value())
    return false;

  luci::CircleConv2D *fused_conv = add->graph()->nodes()->create<luci::CircleConv2D>();
  luci::CircleConst *fused_filter = add->graph()->nodes()->create<luci::CircleConst>();
  luci::CircleConst *fused_bias = add->graph()->nodes()->create<luci::CircleConst>();

  uint32_t filter_out_channel = filter->dim(0).value();
  uint32_t filter_height = filter->dim(1).value();
  uint32_t filter_width = filter->dim(2).value();
  uint32_t filter_in_channel = filter->dim(3).value();

  // Copy filter
  fused_filter->dtype(filter->dtype());
  fused_filter->size<loco::DataType::FLOAT32>(filter->size<loco::DataType::FLOAT32>());
  fused_filter->rank(4);
  fused_filter->dim(0).set(filter_out_channel);
  fused_filter->dim(1).set(filter_height);
  fused_filter->dim(2).set(filter_width);
  fused_filter->dim(3).set(filter_in_channel);
  fused_filter->shape_status(luci::ShapeStatus::VALID);

  // Fuse scale to new filter
  for (uint32_t c = 0; c < filter_out_channel; c++)
  {
    for (uint32_t h = 0; h < filter_height; h++)
    {
      for (uint32_t w = 0; w < filter_width; w++)
      {
        for (uint32_t b = 0; b < filter_in_channel; b++)
        {
          uint32_t offset = c * filter_height * filter_width * filter_in_channel +
                            h * filter_width * filter_in_channel + w * filter_in_channel + b;
          fused_filter->at<loco::DataType::FLOAT32>(offset) =
            filter->at<loco::DataType::FLOAT32>(offset) * scale->at<loco::DataType::FLOAT32>(c);
        }
      }
    }
  }

  // Copy bias
  assert(bias->rank() == 1);
  assert(bias->dim(0).value() == filter_out_channel);
  fused_bias->dtype(bias->dtype());
  fused_bias->size<loco::DataType::FLOAT32>(bias->size<loco::DataType::FLOAT32>());
  fused_bias->rank(1);
  fused_bias->dim(0).set(filter_out_channel);
  fused_bias->shape_status(luci::ShapeStatus::VALID);

  // Fuse scale and shift to bias
  for (uint32_t b = 0; b < filter_out_channel; ++b)
  {
    fused_bias->at<loco::DataType::FLOAT32>(b) =
      bias->at<loco::DataType::FLOAT32>(b) * scale->at<loco::DataType::FLOAT32>(b) +
      shift->at<loco::DataType::FLOAT32>(b);
  }

  // Set attributes of fused_conv
  fused_conv->input(conv->input());
  fused_conv->filter(fused_filter);
  fused_conv->bias(fused_bias);
  fused_conv->fusedActivationFunction(add->fusedActivationFunction());
  fused_conv->padding(conv->padding());
  fused_conv->stride()->h(conv->stride()->h());
  fused_conv->stride()->w(conv->stride()->w());
  fused_conv->dilation()->h(conv->dilation()->h());
  fused_conv->dilation()->w(conv->dilation()->w());

  replace(add).with(fused_conv);

  return true;
}

} // namespace

namespace luci
{

bool FuseBatchNormWithConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
    {
      if (fused_batch_norm_with_conv(add))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
