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

#include "FusePostShiftPass.h"
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

// Fuse Op + CircleCustom(PostShift)
struct FusePostShift final : public luci::CircleNodeMutableVisitor<bool>
{
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleConv2D *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE)
      return false;

    bool changed = false;
    for (auto succ : loco::succs(node))
    {
      auto post_shift = to_post_shift(succ);
      if (not post_shift)
        continue;

      auto param =
        loco::must_cast<luci::CircleConst *>(post_shift->inputs(1)); // FIX_PostShift_UNLESS
      auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
      if (not bias)
        continue;

      uint32_t channel_size = bias->size<loco::DataType::FLOAT32>();
      if (channel_size != param->size<loco::DataType::FLOAT32>())
      {
        assert(false); // FIX_PostScale_Unless
        return false;
      }

      auto cloned_conv = luci::clone_node(node, node->graph());
      assert(cloned_conv != nullptr); // FIX_CALLER_UNLESS
      auto fused_conv = loco::must_cast<luci::CircleConv2D *>(cloned_conv);
      auto fused_bias = luci::clone(bias);

      fused_conv->name(node->name() + "_" + random_str());
      fused_bias->name(bias->name() + "_" + random_str());

      add_origin(fused_conv, luci::get_origin(node));
      add_origin(fused_bias, luci::get_origin(bias));

      // Add param to bias
      for (uint32_t c = 0; c < channel_size; ++c)
      {
        float shift = param->at<loco::DataType::FLOAT32>(c);
        fused_bias->at<loco::DataType::FLOAT32>(c) =
          fused_bias->at<loco::DataType::FLOAT32>(c) + shift;
      }

      fused_conv->input(node->input());
      fused_conv->filter(node->filter());
      fused_conv->bias(fused_bias);

      loco::replace(post_shift).with(fused_conv);
      changed = true;
    }

    return changed;
  }

  bool visit(luci::CircleDepthwiseConv2D *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE)
      return false;

    bool changed = false;
    for (auto succ : loco::succs(node))
    {
      auto post_shift = to_post_shift(succ);
      if (not post_shift)
        continue;

      auto param =
        loco::must_cast<luci::CircleConst *>(post_shift->inputs(1)); // FIX_PostShift_UNLESS
      auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
      if (not bias)
        continue;

      uint32_t channel_size = bias->size<loco::DataType::FLOAT32>();
      if (channel_size != param->size<loco::DataType::FLOAT32>())
      {
        assert(false); // FIX_PostScale_Unless
        return false;
      }

      auto cloned_dconv = luci::clone_node(node, node->graph());
      assert(cloned_dconv != nullptr); // FIX_CALLER_UNLESS
      auto fused_dconv = loco::must_cast<luci::CircleDepthwiseConv2D *>(cloned_dconv);
      auto fused_bias = luci::clone(bias);

      fused_dconv->name(node->name() + "_" + random_str());
      fused_bias->name(bias->name() + "_" + random_str());

      add_origin(fused_dconv, luci::get_origin(node));
      add_origin(fused_bias, luci::get_origin(bias));

      // Add param to bias
      for (uint32_t c = 0; c < channel_size; ++c)
      {
        float shift = param->at<loco::DataType::FLOAT32>(c);
        fused_bias->at<loco::DataType::FLOAT32>(c) =
          fused_bias->at<loco::DataType::FLOAT32>(c) + shift;
      }

      fused_dconv->input(node->input());
      fused_dconv->filter(node->filter());
      fused_dconv->bias(fused_bias);

      loco::replace(post_shift).with(fused_dconv);
      changed = true;
    }

    return changed;
  }

  bool visit(luci::CircleTransposeConv *node)
  {
    bool changed = false;
    for (auto succ : loco::succs(node))
    {
      auto post_shift = to_post_shift(succ);
      if (not post_shift)
        continue;

      auto param =
        loco::must_cast<luci::CircleConst *>(post_shift->inputs(1)); // FIX_PostShift_UNLESS

      // TConv has bias. Update bias.
      if (auto bias = dynamic_cast<luci::CircleConst *>(node->bias()))
      {
        uint32_t channel_size = bias->size<loco::DataType::FLOAT32>();
        if (channel_size != param->size<loco::DataType::FLOAT32>())
        {
          assert(false); // FIX_PostScale_Unless
          return false;
        }

        auto cloned_tconv = luci::clone_node(node, node->graph());
        assert(cloned_tconv != nullptr); // FIX_CALLER_UNLESS
        auto fused_tconv = loco::must_cast<luci::CircleTransposeConv *>(cloned_tconv);
        auto fused_bias = luci::clone(bias);

        fused_tconv->name(node->name() + "_" + random_str());
        fused_bias->name(bias->name() + "_" + random_str());

        add_origin(fused_tconv, luci::get_origin(node));
        add_origin(fused_bias, luci::get_origin(bias));

        // Add param to bias
        for (uint32_t c = 0; c < channel_size; ++c)
        {
          float shift = param->at<loco::DataType::FLOAT32>(c);
          fused_bias->at<loco::DataType::FLOAT32>(c) =
            fused_bias->at<loco::DataType::FLOAT32>(c) + shift;
        }

        fused_tconv->inputSizes(node->inputSizes());
        fused_tconv->outBackprop(node->outBackprop());
        fused_tconv->filter(node->filter());
        fused_tconv->bias(fused_bias);

        loco::replace(post_shift).with(fused_tconv);
        changed = true;
        continue;
      }

      // TConv has no bias. Just use param
      if (auto bias = dynamic_cast<luci::CircleOutputExclude *>(node->bias()))
      {
        auto cloned_tconv = luci::clone_node(node, node->graph());
        assert(cloned_tconv != nullptr); // FIX_CALLER_UNLESS
        auto fused_tconv = loco::must_cast<luci::CircleTransposeConv *>(cloned_tconv);

        fused_tconv->inputSizes(node->inputSizes());
        fused_tconv->outBackprop(node->outBackprop());
        fused_tconv->filter(node->filter());
        fused_tconv->bias(param);

        loco::replace(post_shift).with(fused_tconv);
        changed = true;
        continue;
      }
    }

    return changed;
  }
};

} // namespace

namespace fme_apply
{

bool FusePostShiftPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    FusePostShift fps;
    auto cnode = loco::must_cast<luci::CircleNode *>(node);
    if (cnode->accept(&fps))
      changed = true;
  }

  return changed;
}

} // namespace fme_apply
