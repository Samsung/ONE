/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseSiblingsPass.h"

#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <loco/Service/ShapeInference.h>

#define RETURN_UNLESS(cond) \
  if (not(cond))            \
    return false;

#define CONTINUE_UNLESS(cond) \
  if (not(cond))              \
    continue;

namespace
{

/**
 * This pass fuses sibling operations into one operation, if they are compatible.
 * For now, this pass only works with convolutions, but the same type of optimization can be applied
 * to many operations. More formally, the pass transforms the following pattern
 *
 *                 [In]
 *                   |
 *                   V
 *     +----------- ifm -----------+
 *     |             |             |
 *     v             V             V
 *   Conv2D        Conv2D       Other op(e.g. Pad)
 *
 * Into
 *
 *                 [In]
 *                   |
 *                   V
 *                  ifm -----------+
 *                   |             |
 *                   V             V
 *                 Conv2D       Other op(e.g. Pad)
 *                   |
 *                   V
 *    +----------- SplitV -----------+
 *    |                              |
 *    V                              V
 * SplitVOut                       SplitVOut
 */

loco::Node *parent(loco::Node *node)
{
  loco::Node *p = nullptr;
  for (auto candidate : loco::preds(node))
  {
    if (not dynamic_cast<luci::CircleConst *>(candidate))
    {
      // multiple-input nodes are not supported
      if (p != nullptr)
        return {};

      p = candidate;
    }
  }

  return p;
}

std::vector<loco::Node *> siblings(loco::Node *node)
{
  LOGGER(l);

  auto p = parent(node);

  if (not p)
    return {};

  std::vector<loco::Node *> result;
  for (auto sibling : loco::succs(p))
  {
    if (node == sibling)
    {
      continue;
    }
    result.emplace_back(sibling);
  }

  return result;
}
class FuseSiblings final : public luci::CircleNodeMutableVisitor<bool>
{
  bool visit(luci::CircleConv2D *node) final
  {
    auto const F32 = loco::DataType::FLOAT32;

    auto l_filter = dynamic_cast<luci::CircleConst *>(node->filter());
    RETURN_UNLESS(l_filter);
    RETURN_UNLESS(l_filter->dtype() == F32);
    RETURN_UNLESS(l_filter->shape_status() == luci::ShapeStatus::VALID);
    RETURN_UNLESS(l_filter->rank() == 4);
    std::vector<uint32_t> lw_shape = {l_filter->dim(0).value(), l_filter->dim(1).value(),
                                      l_filter->dim(2).value(), l_filter->dim(3).value()};

    auto prnt = parent(node);
    RETURN_UNLESS(prnt);

    luci::CircleConv2D *r_conv = nullptr;
    for (auto sibling : siblings(node))
    {
      r_conv = dynamic_cast<luci::CircleConv2D *>(sibling);
      CONTINUE_UNLESS(r_conv);

      auto r_filter = dynamic_cast<luci::CircleConst *>(r_conv->filter());
      CONTINUE_UNLESS(r_filter);
      CONTINUE_UNLESS(r_filter->dtype() == F32);
      CONTINUE_UNLESS(r_filter->shape_status() == luci::ShapeStatus::VALID);
      CONTINUE_UNLESS(r_filter->rank() == 4);

      std::vector<uint32_t> rw_shape = {r_filter->dim(0).value(), r_filter->dim(1).value(),
                                        r_filter->dim(2).value(), r_filter->dim(3).value()};

      // Assume KHWD filter layout
      CONTINUE_UNLESS((lw_shape.size() == rw_shape.size()) && (lw_shape[1] == rw_shape[1]) &&
                      (lw_shape[2] == rw_shape[2]) && (lw_shape[3] == rw_shape[3]));

      auto l_pad = node->padding();
      auto r_pad = r_conv->padding();
      CONTINUE_UNLESS(l_pad == r_pad);

      auto l_stride = node->stride();
      auto r_stride = r_conv->stride();
      CONTINUE_UNLESS((l_stride->h() == r_stride->h()) && (l_stride->w() == r_stride->w()));

      CONTINUE_UNLESS(node->fusedActivationFunction() == r_conv->fusedActivationFunction());
    }

    if (not r_conv)
    {
      return false;
    }

    auto fused_filter = node->graph()->nodes()->create<luci::CircleConst>();
    {
      auto r_filter = dynamic_cast<luci::CircleConst *>(r_conv->filter());
      RETURN_UNLESS(r_filter);

      fused_filter->name(l_filter->name() + "/" + r_filter->name());

      std::vector<uint32_t> rw_shape = {r_filter->dim(0).value(), r_filter->dim(1).value(),
                                        r_filter->dim(2).value(), r_filter->dim(3).value()};

      fused_filter->rank(4);
      fused_filter->shape_status(luci::ShapeStatus::VALID);
      fused_filter->shape({lw_shape[0] + rw_shape[0], lw_shape[1], lw_shape[2], lw_shape[3]});

      fused_filter->dtype(F32);

      fused_filter->size<F32>(l_filter->size<F32>() + r_filter->size<F32>());

      uint32_t fused_offset = 0;

      for (uint32_t l_offset = 0; l_offset < l_filter->size<F32>(); ++l_offset)
      {
        fused_filter->at<F32>(fused_offset++) = l_filter->at<F32>(l_offset);
      }

      for (uint32_t r_offset = 0; r_offset < r_filter->size<F32>(); ++r_offset)
      {
        fused_filter->at<F32>(fused_offset++) = r_filter->at<F32>(r_offset);
      }
    }

    uint32_t const LK = l_filter->dim(0).value();
    uint32_t const RK = fused_filter->dim(0).value() - LK;

    auto l_bias = dynamic_cast<luci::CircleConst *>(node->bias());
    RETURN_UNLESS(l_bias);

    auto r_bias = dynamic_cast<luci::CircleConst *>(r_conv->bias());
    RETURN_UNLESS(r_bias);

    auto fused_bias = node->graph()->nodes()->create<luci::CircleConst>();
    {
      fused_bias->name(l_bias->name() + "+" + r_bias->name());

      fused_bias->rank(1);
      fused_bias->shape_status(luci::ShapeStatus::VALID);
      fused_bias->shape({fused_filter->dim(0).value()});

      fused_bias->dtype(F32);

      fused_bias->size<F32>(fused_filter->dim(0).value());

      uint32_t fused_offset = 0;
      for (uint32_t l_offset = 0; l_offset < LK; ++l_offset)
      {
        fused_bias->at<F32>(fused_offset++) = l_bias ? l_bias->at<F32>(l_offset) : 0;
      }

      for (uint32_t r_offset = 0; r_offset < RK; ++r_offset)
      {
        fused_bias->at<F32>(fused_offset++) = r_bias ? r_bias->at<F32>(r_offset) : 0;
      }
    }

    auto fused_conv = node->graph()->nodes()->create<luci::CircleConv2D>();
    {
      fused_conv->name(node->name() + "/" + r_conv->name());
      luci::add_origin(fused_conv,
                       luci::composite_origin({luci::get_origin(node), luci::get_origin(r_conv)}));
      fused_conv->rank(4);
      fused_conv->shape_status(luci::ShapeStatus::VALID);
      fused_conv->shape(
        {node->dim(0).value(), node->dim(1).value(), node->dim(2).value(), LK + RK});

      fused_conv->dtype(F32);

      fused_conv->input(prnt);
      fused_conv->filter(fused_filter);
      fused_conv->bias(fused_bias);

      fused_conv->padding(node->padding());
      fused_conv->stride()->h(node->stride()->h());
      fused_conv->stride()->w(node->stride()->w());

      fused_conv->fusedActivationFunction(node->fusedActivationFunction());
    }

    auto splitv = node->graph()->nodes()->create<luci::CircleSplitV>();
    {
      splitv->name(node->name() + "/" + r_conv->name() + "/split");
      luci::add_origin(splitv,
                       luci::composite_origin({luci::get_origin(node), luci::get_origin(r_conv)}));
      splitv->input(fused_conv);
      splitv->num_split(2);

      auto split_dim = node->graph()->nodes()->create<luci::CircleConst>();
      split_dim->dtype(loco::DataType::S32);
      split_dim->rank(0);
      split_dim->size<loco::DataType::S32>(1);
      split_dim->scalar<loco::DataType::S32>() = 3;
      split_dim->shape_status(luci::ShapeStatus::VALID);
      split_dim->name(splitv->name() + "/split_dim");

      splitv->split_dim(split_dim);

      auto size_splits = node->graph()->nodes()->create<luci::CircleConst>();
      size_splits->dtype(loco::DataType::S32);
      size_splits->rank(1);
      size_splits->size<loco::DataType::S32>(2);
      size_splits->shape({2});
      size_splits->at<loco::DataType::S32>(0) = node->dim(3).value();
      size_splits->at<loco::DataType::S32>(1) = r_conv->dim(3).value();
      assert(size_splits->at<loco::DataType::S32>(0) + size_splits->at<loco::DataType::S32>(1) ==
             fused_filter->dim(0).value());
      size_splits->name(splitv->name() + "/size_splits");

      splitv->size_splits(size_splits);
    }

    auto l_splitv = node->graph()->nodes()->create<luci::CircleSplitVOut>();
    {
      l_splitv->name(node->name() + "/" + r_conv->name() + "/split_left");
      luci::add_origin(l_splitv, luci::get_origin(node));
      l_splitv->input(splitv);
      l_splitv->index(0);
      l_splitv->rank(4);
      l_splitv->shape(
        {node->dim(0).value(), node->dim(1).value(), node->dim(2).value(), node->dim(3).value()});
      l_splitv->shape_status(luci::ShapeStatus::VALID);
    }

    auto r_splitv = node->graph()->nodes()->create<luci::CircleSplitVOut>();
    {
      r_splitv->name(node->name() + "/" + r_conv->name() + "/split_right");
      luci::add_origin(r_splitv, luci::get_origin(r_conv));
      r_splitv->input(splitv);
      r_splitv->index(1);
      r_splitv->rank(4);
      r_splitv->shape({r_conv->dim(0).value(), r_conv->dim(1).value(), r_conv->dim(2).value(),
                       r_conv->dim(3).value()});
      r_splitv->shape_status(luci::ShapeStatus::VALID);
    }

    loco::replace(node).with(l_splitv);
    loco::replace(r_conv).with(r_splitv);
    node->drop();
    r_conv->drop();

    return true;
  }

  bool visit(luci::CircleNode *) final { return false; }
};
} // namespace

namespace luci
{

bool FuseSiblingsPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "FuseSiblingsPass Start" << std::endl;

  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto conv = dynamic_cast<luci::CircleConv2D *>(node);
    CONTINUE_UNLESS(conv);

    FuseSiblings visitor;
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    if (circle_node->accept(&visitor))
    {
      changed = true;
    }
    else
    {
      continue;
    }
  }

  INFO(l) << "FuseSiblingsPass End" << std::endl;
  return changed;
}

} // namespace luci
