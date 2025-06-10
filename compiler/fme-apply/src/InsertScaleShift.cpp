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

#include <algorithm>
#include <cassert>
#include <cmath>

#include "InsertScaleShift.h"
#include "EqualizePattern.h"
#include "Support.Misc.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>

using namespace fme_apply;

#define THROW_UNLESS(COND, MSG) \
  if (not(COND))                \
    throw std::runtime_error(MSG);

namespace
{

std::vector<float> reciprocal(const std::vector<float> &val)
{
  std::vector<float> res(val.size());
  for (uint32_t i = 0; i < res.size(); i++)
  {
    if (val[i] == 0.0)
      throw std::runtime_error("Scale is zero. Divide-by-zero error.");

    res[i] = 1.0 / val[i];
  }
  return res;
}

bool calculate_smooth_quant_scale(luci::CircleNode *node, EqualizePattern *p)
{
  if (p->scale.size() != 0)
  {
    throw std::runtime_error("scale should be empty at this moment.");
  }

  luci::CircleConst *weight = nullptr;
  switch (node->opcode())
  {
    case luci::CircleOpcode::CONV_2D:
    {
      auto conv = loco::must_cast<luci::CircleConv2D *>(node);
      weight = dynamic_cast<luci::CircleConst *>(conv->filter());
      break;
    }
    case luci::CircleOpcode::DEPTHWISE_CONV_2D:
    {
      auto conv = loco::must_cast<luci::CircleDepthwiseConv2D *>(node);
      weight = dynamic_cast<luci::CircleConst *>(conv->filter());
      break;
    }
    case luci::CircleOpcode::FULLY_CONNECTED:
    {
      auto fc = loco::must_cast<luci::CircleFullyConnected *>(node);
      weight = dynamic_cast<luci::CircleConst *>(fc->weights());
      break;
    }
    case luci::CircleOpcode::TRANSPOSE_CONV:
    {
      auto conv = loco::must_cast<luci::CircleTransposeConv *>(node);
      weight = dynamic_cast<luci::CircleConst *>(conv->filter());
      break;
    }
    default:
    {
      throw std::runtime_error("(calculate_smooth_quant_scale) NYI operator: " + node->name());
    }
  }

  if (not weight)
    return false;
  if (weight->dtype() != loco::DataType::FLOAT32)
    return false;

  auto act_scale = p->act_scale;
  switch (node->opcode())
  {
    case luci::CircleOpcode::CONV_2D:
    case luci::CircleOpcode::DEPTHWISE_CONV_2D:
    case luci::CircleOpcode::TRANSPOSE_CONV:
    {
      auto weight_rank = weight->rank();
      if (weight_rank != 4)
        return false;

      if (act_scale.size() != weight->dim(3).value())
      {
        throw std::runtime_error(
          "Mismatch between 'act_scale' size and 'filter' input channel size" +
          std::to_string(act_scale.size()) + " != " + std::to_string(weight->dim(3).value()));
      }

      // Find filter max along with In-channel dimension
      auto weight_size = weight->size<loco::DataType::FLOAT32>();
      const auto weight_O = weight->dim(0).value();
      const auto weight_H = weight->dim(1).value();
      const auto weight_W = weight->dim(2).value();
      const auto weight_I = weight->dim(3).value();
      const auto norm_dim = weight_O * weight_H * weight_W;
      std::vector<float> weight_max(weight_I, std::numeric_limits<float>::min());
      uint32_t cur = 0;
      for (uint32_t i = 0; i < weight_I; i++)
      {
        cur = i;
        for (uint32_t j = 0; j < norm_dim; j++)
        {
          weight_max.at(i) =
            std::max(weight_max.at(i), std::abs(weight->at<loco::DataType::FLOAT32>(cur)));
          cur += weight_I;
        }
      }
      // Check if it properly iterates the filter.
      assert(cur - weight_I == weight_size - 1);

      // TODO parameterize "alpha"
      auto alpha = 0.5f;
      assert(p->act_scale.size() == weight_max.size());
      for (uint32_t i = 0; i < p->act_scale.size(); i++)
      {
        p->scale.push_back(std::max(
          std::pow(p->act_scale.at(i), alpha) / std::pow(weight_max.at(i), (1 - alpha)), 1e-5f));
      }
      break;
    }
    case luci::CircleOpcode::FULLY_CONNECTED:
    {
      auto weight_rank = weight->rank();
      if (weight_rank != 2)
        return false;

      if (act_scale.size() != weight->dim(1).value())
      {
        throw std::runtime_error(
          "Mismatch between 'act_scale' size and 'filter' input channel size" +
          std::to_string(act_scale.size()) + " != " + std::to_string(weight->dim(1).value()));
      }

      // Find filter max along with In-channel dimension
      auto weight_size = weight->size<loco::DataType::FLOAT32>();
      const auto weight_O = weight->dim(0).value();
      const auto weight_I = weight->dim(1).value();
      std::vector<float> weight_max(weight_I, std::numeric_limits<float>::min());
      uint32_t cur = 0;
      for (uint32_t i = 0; i < weight_I; i++)
      {
        cur = i;
        for (uint32_t j = 0; j < weight_O; j++)
        {
          weight_max.at(i) =
            std::max(weight_max.at(i), std::abs(weight->at<loco::DataType::FLOAT32>(cur)));
          cur += weight_I;
        }
      }
      // Check if it properly iterates the filter.
      assert(cur - weight_I == weight_size - 1);

      // TODO parameterize "alpha"
      auto alpha = 0.5f;
      assert(p->act_scale.size() == weight_max.size());
      for (uint32_t i = 0; i < p->act_scale.size(); i++)
      {
        p->scale.push_back(std::max(
          std::pow(p->act_scale.at(i), alpha) / std::pow(weight_max.at(i), (1 - alpha)), 1e-5f));
      }
      break;
    }
    default:
    {
      throw std::runtime_error("(calculate_smooth_quant_scale) NYI operator: " + node->name());
    }
  }

  return true;
}

struct InsertScaleShiftVisitor final : public luci::CircleNodeMutableVisitor<void>
{
  InsertScaleShiftVisitor(EqualizePattern *p) : _pattern(p)
  {
    // DO NOTHING
  }

private:
  EqualizePattern *_pattern = nullptr;

private:
  void insert_scale_before(luci::CircleNode *node, const std::vector<float> &scale)
  {
    assert(node);
    auto previous_node = loco::must_cast<luci::CircleNode *>(get_input(node));

    // Create const for scale
    auto param = node->graph()->nodes()->create<luci::CircleConst>();
    {
      param->name(node->name() + "_scale_const");
      param->dtype(loco::DataType::FLOAT32);
      param->rank(1);
      param->dim(0).set(scale.size());
      param->size<loco::DataType::FLOAT32>(scale.size());
      for (uint32_t i = 0; i < scale.size(); i++)
      {
        param->at<loco::DataType::FLOAT32>(i) = scale.at(i);
      }
      param->shape_status(luci::ShapeStatus::VALID);
      luci::add_origin(param, luci::get_origin(node));
    }

    // Create Scale Op
    auto ss = node->graph()->nodes()->create<luci::CircleCustom>(2, 1);
    {
      ss->name(node->name() + "_scale");
      ss->dtype(loco::DataType::FLOAT32);
      ss->rank(previous_node->rank());
      for (uint32_t i = 0; i < previous_node->rank(); i++)
        ss->dim(i).set(previous_node->dim(i).value());
      ss->custom_code("scale");

      ss->inputs(0, previous_node);
      ss->inputs(1, param);

      ss->shape_status(luci::ShapeStatus::VALID);
      luci::add_origin(ss, luci::get_origin(node));
    }
    set_input(node, ss);
  }

  void insert_scale_after(luci::CircleNode *node, const std::vector<float> &scale)
  {
    if (not node)
    {
      throw std::runtime_error("(insert_scale_after) Invalid node.");
    }

    // Create const for scale
    auto param = node->graph()->nodes()->create<luci::CircleConst>();
    {
      param->name(node->name() + "_scale_const");
      param->dtype(loco::DataType::FLOAT32);
      param->rank(1);
      param->dim(0).set(scale.size());
      param->size<loco::DataType::FLOAT32>(scale.size());
      for (uint32_t i = 0; i < scale.size(); i++)
      {
        param->at<loco::DataType::FLOAT32>(i) = scale.at(i);
      }
      param->shape_status(luci::ShapeStatus::VALID);
      luci::add_origin(param, luci::get_origin(node));
    }

    // NOTE. Get a node user before setting CircleCustom("scale")'s input.
    auto uses = loco::succs(node);
    if (uses.size() != 1)
    {
      throw std::runtime_error("Not support multiple output nodes.");
    }

    // Create Scale Op
    auto ss = node->graph()->nodes()->create<luci::CircleCustom>(2, 1);
    {
      ss->name(node->name() + "_scale");
      ss->dtype(loco::DataType::FLOAT32);
      ss->rank(node->rank());
      for (uint32_t i = 0; i < node->rank(); i++)
        ss->dim(i).set(node->dim(i).value());
      ss->custom_code("scale");

      ss->inputs(0, node);
      ss->inputs(1, param);

      ss->shape_status(luci::ShapeStatus::VALID);
      luci::add_origin(ss, luci::get_origin(node));
    }

    set_input(loco::must_cast<luci::CircleNode *>(*uses.begin()), ss);
  }

private:
  void insert(luci::CircleNode *node)
  {
    assert(_pattern->scale.size() ==
           0); // scale for smooth qaunt is not calculated yet at this moment.
    auto valid = ::calculate_smooth_quant_scale(node, _pattern);
    auto back_node = node;
    // Find front node.
    const auto support_depth = 3;
    auto front_node = find_arg_with_name(node, _pattern->front, support_depth);
    if (not front_node)
    {
      throw std::runtime_error("Cannot find front node: " + _pattern->front);
    }
    insert_scale_after(front_node, reciprocal(_pattern->scale));
    insert_scale_before(back_node, _pattern->scale);
  }

private:
  void visit(luci::CircleOutput *) {}

  void visit(luci::CircleNode *node)
  {
    throw std::runtime_error("(InsertScaleShiftVisitor) NYI operator: " + node->name());
  }

  void visit(luci::CircleConv2D *node) { insert(node); }

  void visit(luci::CircleDepthwiseConv2D *node) { insert(node); }

  void visit(luci::CircleFullyConnected *node) { insert(node); }

  void visit(luci::CircleTransposeConv *node) { insert(node); }
};

} // namespace

namespace fme_apply
{

void InsertScaleShift::run(loco::Graph *g)
{
  // Create a map for pattern matching
  // { back(string) -> EqualizationPattern*}
  // This assumes that each EqualizePattern has a unique 'back'
  std::map<std::string, EqualizePattern *> pattern_by_back;
  {
    for (auto &pattern : _patterns)
    {
      auto back = pattern.back;
      pattern_by_back[back] = &pattern;
    }
  }

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    auto cnode = loco::must_cast<luci::CircleNode *>(node);

    if (pattern_by_back.find(cnode->name()) == pattern_by_back.end())
      continue;

    auto pattern = pattern_by_back.at(cnode->name());
    InsertScaleShiftVisitor issv(pattern);
    cnode->accept(&issv);
  }
}

} // namespace fme_apply

#undef THROW_UNLESS
