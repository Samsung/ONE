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

  bool visit(luci::CircleConv2D *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE and
        node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return false;

    luci::CircleCustom *post_scale = nullptr;
    bool multi_custom = false;
    auto succs = loco::succs(node);
    for (const auto succ : succs)
    {
      post_scale = to_scale(succ);
      if (multi_custom && post_scale)
      {
        throw std::runtime_error("Do not allow multiple scales.");
      }
      if (post_scale)
      {
        multi_custom = true;
      }
    }
    if (not post_scale)
      return false;

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
      throw std::runtime_error(
        "Mismatch between scale size and filter output channel size: " + std::to_string(filter_o) +
        " != " + std::to_string(param->size<loco::DataType::FLOAT32>()));
    }

    auto cloned_conv = luci::clone_node(node, node->graph());
    assert(cloned_conv != nullptr); // FIX_CALLER_UNLESS
    auto fused_conv = loco::must_cast<luci::CircleConv2D *>(cloned_conv);
    auto fused_filter = luci::clone(filter);
    auto fused_bias = luci::clone(bias);

    fused_conv->name(node->name() + "_fused_" + random_str());
    fused_filter->name(filter->name() + "_fused_" + random_str());
    fused_bias->name(bias->name() + "_fused_" + random_str());

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

    return true;
  }

  bool visit(luci::CircleDepthwiseConv2D *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE and
        node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return false;

    luci::CircleCustom *post_scale = nullptr;
    bool multi_custom = false;
    auto succs = loco::succs(node);
    for (const auto succ : succs)
    {
      post_scale = to_scale(succ);
      if (multi_custom && post_scale)
      {
        throw std::runtime_error("Do not allow multiple scales.");
      }
      if (post_scale)
      {
        multi_custom = true;
      }
    }
    if (not post_scale)
      return false;

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

    return true;
  }

  bool visit(luci::CircleFullyConnected *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE and
        node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return false;

    luci::CircleCustom *post_scale = nullptr;
    bool multi_custom = false;
    auto succs = loco::succs(node);
    for (const auto succ : succs)
    {
      post_scale = to_scale(succ);
      if (multi_custom && post_scale)
      {
        throw std::runtime_error("Do not allow multiple scales.");
      }
      if (post_scale)
      {
        multi_custom = true;
      }
    }
    if (not post_scale)
      return false;

    auto param =
      loco::must_cast<luci::CircleConst *>(post_scale->inputs(1)); // FIX_PostScale_UNLESS
    auto filter = loco::must_cast<luci::CircleConst *>(node->weights());
    luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());

    uint32_t filter_o = filter->dim(0).value();
    uint32_t filter_i = filter->dim(1).value();

    const auto param_size = param->size<loco::DataType::FLOAT32>();
    if (filter_o != param_size)
    {
      throw std::runtime_error("Mismatch between scale size and filter output channel size: " +
                               std::to_string(filter_o) + " != " + std::to_string(param_size));
    }
    if (bias)
    {
      const auto bias_size = bias->size<loco::DataType::FLOAT32>();
      if (bias_size != param_size)
      {
        throw std::runtime_error("Mismatch between scale size and bias size: " +
                                 std::to_string(bias_size) + " != " + std::to_string(param_size));
      }
    }

    auto cloned_fc = luci::clone_node(node, node->graph());
    assert(cloned_fc != nullptr); // FIX_CALLER_UNLESS
    auto fused_fc = loco::must_cast<luci::CircleFullyConnected *>(cloned_fc);
    auto fused_filter = luci::clone(filter);

    fused_fc->name(node->name() + "_fused_" + random_str());
    fused_filter->name(filter->name() + "_fused_" + random_str());

    add_origin(fused_fc, luci::get_origin(node));
    add_origin(fused_filter, luci::get_origin(filter));

    luci::CircleConst *fused_bias = nullptr;
    if (bias)
    {
      fused_bias = luci::clone(bias);
      fused_bias->name(bias->name() + "_fused_" + random_str());
      add_origin(fused_bias, luci::get_origin(bias));
    }

    // Multiply param to weights
    for (uint32_t o = 0; o < filter_o; o++)
    {
      float scale = param->at<loco::DataType::FLOAT32>(o);
      assert(scale > 0.0); // Defensive guard
      for (uint32_t i = 0; i < filter_i; i++)
      {
        uint32_t offset = o * filter_i + i;

        fused_filter->at<loco::DataType::FLOAT32>(offset) =
          fused_filter->at<loco::DataType::FLOAT32>(offset) * scale;
      }
    }

    fused_fc->input(node->input());
    fused_fc->weights(fused_filter);
    fused_fc->bias(node->bias());

    if (bias)
    {
      // Multiply param to bias
      for (uint32_t c = 0; c < filter_o; ++c)
      {
        float scale = param->at<loco::DataType::FLOAT32>(c);
        fused_bias->at<loco::DataType::FLOAT32>(c) =
          fused_bias->at<loco::DataType::FLOAT32>(c) * scale;
      }
      fused_fc->bias(fused_bias);
    }

    loco::replace(post_scale).with(fused_fc);

    return true;
  }

  bool visit(luci::CircleInstanceNorm *node)
  {
    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE and
        node->fusedActivationFunction() != luci::FusedActFunc::RELU)
      return false;

    luci::CircleCustom *post_scale = nullptr;
    bool multi_custom = false;
    auto succs = loco::succs(node);
    for (const auto succ : succs)
    {
      post_scale = to_scale(succ);
      if (multi_custom && post_scale)
      {
        throw std::runtime_error("Do not allow multiple scales.");
      }
      if (post_scale)
      {
        multi_custom = true;
      }
    }
    if (not post_scale)
      return false;

    auto param =
      loco::must_cast<luci::CircleConst *>(post_scale->inputs(1)); // FIX_PostScale_UNLESS
    auto gamma = loco::must_cast<luci::CircleConst *>(node->gamma());
    auto beta = loco::must_cast<luci::CircleConst *>(node->beta());

    const auto gamma_size = gamma->size<loco::DataType::FLOAT32>();
    const auto beta_size = beta->size<loco::DataType::FLOAT32>();
    assert(gamma_size == beta_size);
    const auto param_size = param->size<loco::DataType::FLOAT32>();
    if (gamma_size != param_size)
    {
      throw std::runtime_error("Mismatch between scale size and gamma size: " +
                               std::to_string(gamma_size) + " != " + std::to_string(param_size));
    }

    auto cloned_instnorm = luci::clone_node(node, node->graph());
    assert(cloned_instnorm != nullptr); // FIX_CALLER_UNLESS
    auto fused_instnorm = loco::must_cast<luci::CircleInstanceNorm *>(cloned_instnorm);
    auto fused_gamma = luci::clone(gamma);
    auto fused_beta = luci::clone(beta);

    fused_instnorm->name(node->name() + "_fused_" + random_str());
    fused_gamma->name(gamma->name() + "_fused_" + random_str());
    fused_beta->name(beta->name() + "_fused_" + random_str());

    add_origin(fused_instnorm, luci::get_origin(node));
    add_origin(fused_gamma, luci::get_origin(gamma));
    add_origin(fused_beta, luci::get_origin(beta));

    // Multiply param to gamma and beta
    for (uint32_t idx = 0; idx < gamma_size; idx++)
    {
      float scale = param->at<loco::DataType::FLOAT32>(idx);
      assert(scale > 0.0); // Defensive guard

      fused_gamma->at<loco::DataType::FLOAT32>(idx) =
        fused_gamma->at<loco::DataType::FLOAT32>(idx) * scale;
      fused_beta->at<loco::DataType::FLOAT32>(idx) =
        fused_beta->at<loco::DataType::FLOAT32>(idx) * scale;
    }

    fused_instnorm->input(node->input());
    fused_instnorm->gamma(fused_gamma);
    fused_instnorm->beta(fused_beta);

    loco::replace(post_scale).with(fused_instnorm);

    return true;
  }

  bool visit(luci::CircleTransposeConv *node)
  {
    luci::CircleCustom *post_scale = nullptr;
    bool multi_custom = false;
    auto succs = loco::succs(node);
    for (const auto succ : succs)
    {
      post_scale = to_scale(succ);
      if (multi_custom && post_scale)
      {
        throw std::runtime_error("Do not allow multiple scales.");
      }
      if (post_scale)
      {
        multi_custom = true;
      }
    }
    if (not post_scale)
      return false;

    auto param =
      loco::must_cast<luci::CircleConst *>(post_scale->inputs(1)); // FIX_PostScale_UNLESS
    auto filter = loco::must_cast<luci::CircleConst *>(node->filter());
    auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
    // Non-const bias is not supported
    if (not bias)
    {
      if (nullptr == dynamic_cast<luci::CircleOutputExclude *>(node->bias()))
        return false;
    }

    uint32_t filter_o = filter->dim(0).value();
    uint32_t filter_h = filter->dim(1).value();
    uint32_t filter_w = filter->dim(2).value();
    uint32_t filter_i = filter->dim(3).value();

    if (filter_o != param->size<loco::DataType::FLOAT32>())
    {
      throw std::runtime_error(
        "Mismatch between scale size and filter output channel size: " + std::to_string(filter_o) +
        " != " + std::to_string(param->size<loco::DataType::FLOAT32>()));
    }

    auto cloned_tconv = luci::clone_node(node, node->graph());
    assert(cloned_tconv != nullptr); // FIX_CALLER_UNLESS
    auto fused_tconv = loco::must_cast<luci::CircleTransposeConv *>(cloned_tconv);
    auto fused_filter = luci::clone(filter);

    fused_tconv->name(node->name() + "_fused_" + random_str());
    fused_filter->name(filter->name() + "_fused_" + random_str());

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
      fused_bias->name(bias->name() + "_fused_" + random_str());
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

    return true;
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
