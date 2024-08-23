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

#include "FusePreScalePass.h"
#include "Support.Cast.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

using namespace fme_apply;

namespace
{

// Fuse CircleCustom(PreScale) + Op
struct FusePreScale final : public luci::CircleNodeMutableVisitor<bool>
{
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleConv2D *node)
  {
    auto pre_scale = to_pre_scale(node->input());
    if (not pre_scale)
      return false;

    auto param = loco::must_cast<luci::CircleConst *>(pre_scale->inputs(1)); // FIX_PreScale_UNLESS
    auto filter = loco::must_cast<luci::CircleConst *>(node->filter());

    uint32_t filter_o = filter->dim(0).value();
    uint32_t filter_h = filter->dim(1).value();
    uint32_t filter_w = filter->dim(2).value();
    uint32_t filter_i = filter->dim(3).value();

    if (filter_i != param->size<loco::DataType::FLOAT32>())
    {
      throw std::runtime_error(
        "Mismatch between scale size and filter input channel size: " + std::to_string(filter_i) +
        " != " + std::to_string(param->size<loco::DataType::FLOAT32>()));
    }

    auto fused_filter = luci::clone(filter);
    fused_filter->name(filter->name() + "_fused");
    add_origin(fused_filter, luci::get_origin(filter));

    // Multiply param to weights
    for (uint32_t c = 0; c < filter_o; c++)
    {
      for (uint32_t h = 0; h < filter_h; h++)
      {
        for (uint32_t w = 0; w < filter_w; w++)
        {
          for (uint32_t b = 0; b < filter_i; b++)
          {
            uint32_t offset =
              c * filter_h * filter_w * filter_i + h * filter_w * filter_i + w * filter_i + b;
            float scale = param->at<loco::DataType::FLOAT32>(b);
            assert(scale > 0.0); // Defensive guard

            fused_filter->at<loco::DataType::FLOAT32>(offset) =
              fused_filter->at<loco::DataType::FLOAT32>(offset) * scale;
          }
        }
      }
    }

    node->input(pre_scale->inputs(0));
    node->filter(fused_filter);

    return true;
  }

  bool visit(luci::CircleTransposeConv *node)
  {
    auto pre_scale = to_pre_scale(node->outBackprop());
    if (not pre_scale)
      return false;

    auto param = loco::must_cast<luci::CircleConst *>(pre_scale->inputs(1)); // FIX_PreScale_UNLESS
    auto filter = loco::must_cast<luci::CircleConst *>(node->filter());

    uint32_t filter_o = filter->dim(0).value();
    uint32_t filter_h = filter->dim(1).value();
    uint32_t filter_w = filter->dim(2).value();
    uint32_t filter_i = filter->dim(3).value();

    if (filter_i != param->size<loco::DataType::FLOAT32>())
    {
      throw std::runtime_error(
        "Mismatch between scale size and filter input channel size: " + std::to_string(filter_i) +
        " != " + std::to_string(param->size<loco::DataType::FLOAT32>()));
    }

    auto fused_filter = luci::clone(filter);
    fused_filter->name(filter->name() + "_fused");
    add_origin(fused_filter, luci::get_origin(filter));

    // Multiply param to weights
    for (uint32_t c = 0; c < filter_o; c++)
    {
      for (uint32_t h = 0; h < filter_h; h++)
      {
        for (uint32_t w = 0; w < filter_w; w++)
        {
          for (uint32_t b = 0; b < filter_i; b++)
          {
            uint32_t offset =
              c * filter_h * filter_w * filter_i + h * filter_w * filter_i + w * filter_i + b;
            float scale = param->at<loco::DataType::FLOAT32>(b);
            assert(scale > 0.0); // Defensive guard

            fused_filter->at<loco::DataType::FLOAT32>(offset) =
              fused_filter->at<loco::DataType::FLOAT32>(offset) * scale;
          }
        }
      }
    }

    node->outBackprop(pre_scale->inputs(0));
    node->filter(fused_filter);

    return true;
  }
};

} // namespace

namespace fme_apply
{

bool FusePreScalePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    FusePreScale fps;
    auto cnode = loco::must_cast<luci::CircleNode *>(node);
    if (cnode->accept(&fps))
      changed = true;
  }
  return changed;
}

} // namespace fme_apply
