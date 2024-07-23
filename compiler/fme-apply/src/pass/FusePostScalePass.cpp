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

#include "FusePostScalePass.h"
#include "Support.Cast.h"
#include "RandomString.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

using namespace fme_apply;

namespace
{

// Fuse Op + CircleCustom(PostScale)
struct FusePostScale final : public luci::CircleNodeMutableVisitor<bool>
{
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleDepthwiseConv2D *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE and
        node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return false;

    bool changed = false;
    for (auto succ : loco::succs(node))
    {
      auto post_scale = to_post_scale(succ);
      if (not post_scale)
        continue;

      auto param =
        loco::must_cast<luci::CircleConst *>(post_scale->inputs(1)); // FIX_PostScale_UNLESS
      auto filter = loco::must_cast<luci::CircleConst *>(node->filter());
      auto bias = loco::must_cast<luci::CircleConst *>(node->bias());

      uint32_t filter_i = filter->dim(0).value();
      uint32_t filter_h = filter->dim(1).value();
      uint32_t filter_w = filter->dim(2).value();
      uint32_t filter_o = filter->dim(3).value();

      if (filter_o != param->size<loco::DataType::FLOAT32>())
      {
        assert(false); // FIX_PostScale_Unless
        return false;
      }

      auto cloned_dconv = luci::clone_node(node, node->graph());
      assert(cloned_dconv != nullptr); // FIX_CALLER_UNLESS
      auto fused_dconv = loco::must_cast<luci::CircleDepthwiseConv2D *>(cloned_dconv);
      auto fused_filter = luci::clone(filter);
      auto fused_bias = luci::clone(bias);

      // Add random string to make unique name
      fused_dconv->name(node->name() + "_postscale_" + random_str());
      fused_filter->name(filter->name() + "_postscale_" + random_str());
      fused_bias->name(bias->name() + "_postscale_" + random_str());

      add_origin(fused_dconv, luci::get_origin(node));
      add_origin(fused_filter, luci::get_origin(filter));
      add_origin(fused_bias, luci::get_origin(bias));

      // Multiply param to weights
      for (uint32_t b = 0; b < filter_i; b++)
      {
        for (uint32_t h = 0; h < filter_h; h++)
        {
          for (uint32_t w = 0; w < filter_w; w++)
          {
            for (uint32_t c = 0; c < filter_o; c++)
            {
              uint32_t offset =
                b * filter_h * filter_w * filter_o + h * filter_w * filter_o + w * filter_o + c;
              float scale = param->at<loco::DataType::FLOAT32>(c);
              assert(scale > 0.0); // Defensive guard

              fused_filter->at<loco::DataType::FLOAT32>(offset) =
                fused_filter->at<loco::DataType::FLOAT32>(offset) * scale;
            }
          }
        }
      }

      // Multiply param to bias
      for (uint32_t c = 0; c < filter_o; ++c)
      {
        float scale = param->at<loco::DataType::FLOAT32>(c);
        fused_bias->at<loco::DataType::FLOAT32>(c) =
          fused_bias->at<loco::DataType::FLOAT32>(c) * scale;
      }

      fused_dconv->input(node->input());
      fused_dconv->filter(fused_filter);
      fused_dconv->bias(fused_bias);

      loco::replace(post_scale).with(fused_dconv);
      changed = true;
    }

    return changed;
  }

  bool visit(luci::CircleConv2D *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE and
        node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return false;

    bool changed = false;

    for (auto succ : loco::succs(node))
    {
      auto post_scale = to_post_scale(succ);
      if (not post_scale)
        continue;

      auto param =
        loco::must_cast<luci::CircleConst *>(post_scale->inputs(1)); // FIX_PostScale_UNLESS
      auto filter = loco::must_cast<luci::CircleConst *>(node->filter());
      auto bias = loco::must_cast<luci::CircleConst *>(node->bias());

      uint32_t filter_o = filter->dim(0).value();
      uint32_t filter_h = filter->dim(1).value();
      uint32_t filter_w = filter->dim(2).value();
      uint32_t filter_i = filter->dim(3).value();

      if (filter_o != param->size<loco::DataType::FLOAT32>())
      {
        assert(false); // FIX_PostScale_Unless
        return false;
      }

      auto cloned_conv = luci::clone_node(node, node->graph());
      assert(cloned_conv != nullptr); // FIX_CALLER_UNLESS
      auto fused_conv = loco::must_cast<luci::CircleConv2D *>(cloned_conv);
      auto fused_filter = luci::clone(filter);
      auto fused_bias = luci::clone(bias);

      fused_conv->name(node->name() + "_postscale_" + random_str());
      fused_filter->name(filter->name() + "_postscale_" + random_str());
      fused_bias->name(bias->name() + "_postscale_" + random_str());

      add_origin(fused_conv, luci::get_origin(node));
      add_origin(fused_filter, luci::get_origin(filter));
      add_origin(fused_bias, luci::get_origin(bias));

      // Multiply param to weights
      for (uint32_t c = 0; c < filter_o; c++)
      {
        float scale = param->at<loco::DataType::FLOAT32>(c);
        assert(scale > 0.0); // Defensive guard

        for (uint32_t h = 0; h < filter_h; h++)
        {
          for (uint32_t w = 0; w < filter_w; w++)
          {
            for (uint32_t b = 0; b < filter_i; b++)
            {
              uint32_t offset =
                c * filter_h * filter_w * filter_i + h * filter_w * filter_i + w * filter_i + b;

              fused_filter->at<loco::DataType::FLOAT32>(offset) =
                fused_filter->at<loco::DataType::FLOAT32>(offset) * scale;
            }
          }
        }
      }

      // Multiply param to bias
      for (uint32_t c = 0; c < filter_o; ++c)
      {
        float scale = param->at<loco::DataType::FLOAT32>(c);
        fused_bias->at<loco::DataType::FLOAT32>(c) =
          fused_bias->at<loco::DataType::FLOAT32>(c) * scale;
      }

      fused_conv->input(node->input());
      fused_conv->filter(fused_filter);
      fused_conv->bias(fused_bias);

      loco::replace(post_scale).with(fused_conv);
      changed = true;
    }

    return changed;
  }

  bool visit(luci::CircleTransposeConv *node)
  {
    bool changed = false;

    for (auto succ : loco::succs(node))
    {
      auto post_scale = to_post_scale(succ);
      if (not post_scale)
        continue;

      auto param =
        loco::must_cast<luci::CircleConst *>(post_scale->inputs(1)); // FIX_PostScale_UNLESS
      auto filter = loco::must_cast<luci::CircleConst *>(node->filter());
      auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
      // Non-const bias is not supported
      if (not bias)
      {
        if (nullptr == dynamic_cast<luci::CircleOutputExclude *>(node->bias()))
          continue;
      }

      uint32_t filter_o = filter->dim(0).value();
      uint32_t filter_h = filter->dim(1).value();
      uint32_t filter_w = filter->dim(2).value();
      uint32_t filter_i = filter->dim(3).value();

      if (filter_o != param->size<loco::DataType::FLOAT32>())
      {
        assert(false); // FIX_PostScale_Unless
        return false;
      }

      auto cloned_tconv = luci::clone_node(node, node->graph());
      assert(cloned_tconv != nullptr); // FIX_CALLER_UNLESS
      auto fused_tconv = loco::must_cast<luci::CircleTransposeConv *>(cloned_tconv);
      auto fused_filter = luci::clone(filter);

      fused_tconv->name(node->name() + "_postscale_" + random_str());
      fused_filter->name(filter->name() + "_postscale_" + random_str());

      add_origin(fused_tconv, luci::get_origin(node));
      add_origin(fused_filter, luci::get_origin(filter));

      // Multiply param to weights
      for (uint32_t c = 0; c < filter_o; c++)
      {
        float scale = param->at<loco::DataType::FLOAT32>(c);
        assert(scale > 0.0); // Defensive guard

        for (uint32_t h = 0; h < filter_h; h++)
        {
          for (uint32_t w = 0; w < filter_w; w++)
          {
            for (uint32_t b = 0; b < filter_i; b++)
            {
              uint32_t offset =
                c * filter_h * filter_w * filter_i + h * filter_w * filter_i + w * filter_i + b;

              fused_filter->at<loco::DataType::FLOAT32>(offset) =
                fused_filter->at<loco::DataType::FLOAT32>(offset) * scale;
            }
          }
        }
      }

      if (bias)
      {
        auto fused_bias = luci::clone(bias);
        fused_bias->name(bias->name() + "_postscale_" + random_str());
        add_origin(fused_bias, luci::get_origin(bias));

        // Multiply param to bias
        for (uint32_t c = 0; c < filter_o; ++c)
        {
          float scale = param->at<loco::DataType::FLOAT32>(c);
          fused_bias->at<loco::DataType::FLOAT32>(c) =
            fused_bias->at<loco::DataType::FLOAT32>(c) * scale;
        }
        fused_tconv->bias(fused_bias);
      }

      fused_tconv->outBackprop(node->outBackprop());
      fused_tconv->filter(fused_filter);
      fused_tconv->inputSizes(node->inputSizes());

      loco::replace(post_scale).with(fused_tconv);
      changed = true;
    }

    return changed;
  }
};

} // namespace

namespace fme_apply
{

bool FusePostScalePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    FusePostScale fps;
    auto cnode = loco::must_cast<luci::CircleNode *>(node);
    if (cnode->accept(&fps))
      changed = true;
  }
  return changed;
}

} // namespace fme_apply
