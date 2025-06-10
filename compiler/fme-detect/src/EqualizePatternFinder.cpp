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

#include "EqualizePatternFinder.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <vector>

using namespace fme_detect;

namespace
{

struct Fusability
{
  bool pre_scale = false;
  bool post_scale = false;
  bool pre_shift = false;
  bool post_shift = false;
};

struct GetFusability final : public luci::CircleNodeMutableVisitor<Fusability>
{
  Fusability visit(luci::CircleNode *node) { return Fusability(); }

  Fusability visit(luci::CircleConv2D *node)
  {
    Fusability f;
    {
      f.pre_scale = true;
      if (node->fusedActivationFunction() == luci::FusedActFunc::NONE)
      {
        f.post_scale = true;
        f.post_shift = true;
      }
      // Negative scale is not fusable across ReLU, but fme-detect does not
      // know the scale value. So, we assume that the scale is positive.
      // NOTE If a pattern has negative scales, fm-equalize rejects the pattern
      else if (node->fusedActivationFunction() == luci::FusedActFunc::RELU)
      {
        f.post_scale = true;
      }
    }
    return f;
  }

  Fusability visit(luci::CircleTransposeConv *node)
  {
    Fusability f;
    {
      f.pre_scale = true;
      f.post_scale = true;
      f.post_shift = true;
      // NOTE CircleTransposeConv does not support fused activation function
    }
    return f;
  }

  Fusability visit(luci::CircleDepthwiseConv2D *node)
  {
    Fusability f;
    {
      f.pre_scale = true;
      // PreShift would be fused with DConv when DConv
      // does not use padding
      // TODO Check the above condition and enable the below line
      // f.pre_shift = true;

      if (node->fusedActivationFunction() == luci::FusedActFunc::NONE)
      {
        f.post_scale = true;
        f.post_shift = true;
      }
      // Negative scale is not fusable across ReLU, but fme-detect does not
      // know the scale value. So, we assume that the scale is positive.
      // NOTE If a pattern has negative scales, fm-equalize rejects the pattern
      else if (node->fusedActivationFunction() == luci::FusedActFunc::RELU)
      {
        f.post_scale = true;
      }
    }
    return f;
  }

  Fusability visit(luci::CircleInstanceNorm *node)
  {
    Fusability f;
    {
      f.pre_shift = true;
      if (node->fusedActivationFunction() == luci::FusedActFunc::NONE)
      {
        f.post_scale = true;
        f.post_shift = true;
      }
      // Negative scale is not fusable across ReLU, but fme-detect does not
      // know the scale value. So, we assume that the scale is positive.
      // NOTE If a pattern has negative scales, fm-equalize rejects the pattern
      else if (node->fusedActivationFunction() == luci::FusedActFunc::RELU)
      {
        f.post_scale = true;
      }
    }
    return f;
  }

  Fusability visit(luci::CircleFullyConnected *node)
  {
    Fusability f;
    {
      f.pre_scale = true;
      if (node->fusedActivationFunction() == luci::FusedActFunc::NONE)
      {
        f.post_scale = true;
        f.post_shift = true;
      }
      // Negative scale is not fusable across ReLU, but fme-detect does not
      // know the scale value. So, we assume that the scale is positive.
      // NOTE If a pattern has negative scales, fm-equalize rejects the pattern
      else if (node->fusedActivationFunction() == luci::FusedActFunc::RELU)
      {
        f.post_scale = true;
      }
    }
    return f;
  }
};

Fusability fusability(luci::CircleNode *node)
{
  if (node == nullptr)
    throw std::invalid_argument("node");

  GetFusability gf;
  return node->accept(&gf);
}

struct Forwardable
{
  bool scale_forwardable = false;
  bool shift_forwardable = false;
};

// Return Forwardable of node
// Note that the degree of effect may vary from layer to layer.
Forwardable forwardable(luci::CircleNode *node)
{
  if (node == nullptr)
    throw std::invalid_argument("node");

  switch (node->opcode())
  {
    case luci::CircleOpcode::PAD:
      return {true, false};
    case luci::CircleOpcode::MAX_POOL_2D:
      return {true, true};
    case luci::CircleOpcode::RELU:
      return {true, false};
    case luci::CircleOpcode::LEAKY_RELU:
      return {true, false};
    case luci::CircleOpcode::GELU:
      return {false, false};
    default:
      return {false, false};
  }
}

// Save matching EqualizePattern to res
void match(luci::CircleNode *front, std::vector<EqualizePattern> &res)
{
  if (front == nullptr)
    throw std::invalid_argument("front");

  auto front_fusability = fusability(front);
  auto succs = loco::succs(front);
  // TODO Support multiple successors.
  if (succs.size() != 1)
    return;
  for (auto succ : succs)
  {
    // Check succ fusability
    auto back = loco::must_cast<luci::CircleNode *>(succ);
    auto back_fusability = fusability(back);

    // If 'back' is not fusable with PreScale, we check if PreScale
    // can forward across 'back'
    // TODO Generalize this code to support multiple forwardable Ops
    if (not back_fusability.pre_scale)
    {
      auto f = forwardable(back);
      if (f.scale_forwardable)
      {
        auto back_succs = loco::succs(back);
        // Only support single successor for simplicity
        if (back_succs.size() != 1)
          continue;
        back = loco::must_cast<luci::CircleNode *>(*back_succs.begin());
        back_fusability = fusability(back);
        if (not back_fusability.pre_scale)
        {
          f = forwardable(back);
          if (f.scale_forwardable)
          {
            back_succs = loco::succs(back);
            if (back_succs.size() != 1)
              continue;
            back = loco::must_cast<luci::CircleNode *>(*back_succs.begin());
            back_fusability = fusability(back);
          }
        }
      }
    }

    if (front_fusability.post_scale and back_fusability.pre_scale)
    {
      res.emplace_back(front->name(), back->name(), EqualizePattern::Type::ScaleOnly);
    }

    // TODO Let's consider "shift" when it is necessary.
#if 0
    // Create EqualizePattern based on fusability
    // ScaleShift
    //   front: fusable_post_shift and fusable_post_scale
    //   back: fusable_pre_shift and fusable_post_shift
    if (front_fusability.post_shift and front_fusability.post_scale and
        back_fusability.pre_shift and back_fusability.pre_scale)
    {
      res.emplace_back(front->name(), back->name(), EqualizePattern::Type::ScaleShift);
    }
    // ShiftOnly
    //   front: fusable_post_shift
    //   back: fusable_pre_shift
    else if (front_fusability.post_shift and back_fusability.pre_shift)
    {
      res.emplace_back(front->name(), back->name(), EqualizePattern::Type::ShiftOnly);
    }
    // ScaleOnly
    //   front: fusable_post_scale
    //   back: fusable_pre_scale
    else if (front_fusability.post_scale and back_fusability.pre_scale)
    {
      res.emplace_back(front->name(), back->name(), EqualizePattern::Type::ScaleOnly);
    }
#endif
  }
}

} // namespace

namespace fme_detect
{

std::vector<EqualizePattern> EqualizePatternFinder::find(loco::Graph *g) const
{
  if (g == nullptr)
    throw std::invalid_argument("g");

  std::vector<EqualizePattern> res;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    const auto cnode = loco::must_cast<luci::CircleNode *>(node);
    if (not _ctx._allow_dup_op)
    {
      // We create new Ops (front, back) for each matched pattern, so duplicate
      // Ops can be created. This condition prevents pattern matching for a node
      // (front) with multiple successors (backs) to avoid duplicate computation.
      // TODO Duplicate Ops can be removed if all (pred-succ) pairs are matched
      // as the same pattern. For example, if (pred-succ1) is ScaleOnly and
      // (pred-succ2) is ScaleOnly, pred does not need to be duplicated.
      if (loco::succs(cnode).size() != 1)
        continue;
    }

    ::match(cnode, res);
  }

  return res;
}

} // namespace fme_detect
