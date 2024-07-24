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

#include "InsertScaleShift.h"
#include "EqualizePattern.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>

using namespace fme_apply;

#define THROW_UNLESS(COND, MSG) \
  if (not(COND))                \
    throw std::runtime_error(MSG);

namespace
{

/**
 * Create Scale/Shift Op
 *
 * @param node input of the created Scale/Shift Op
 * @param val value of Scale/Shift parameter (channel-wise const)
 * @param code One of PreScale/PostScale/PreShift/PostShift
 * @return CircleCustom node with code
 */
luci::CircleCustom *create_scale_or_shift(luci::CircleNode *node, const std::vector<float> &val,
                                          const std::string &code)
{
  assert(node); // FIX_CALLER_UNLESS

  THROW_UNLESS(node->rank() == 4, "Node rank is not four");

  // Assume NHWC (index of C: 3)
  auto channel_size = node->dim(3).value();
  THROW_UNLESS(val.size() == channel_size, "Channel and scale/shift size mismatch");

  // Create parameter
  auto param = node->graph()->nodes()->create<luci::CircleConst>();
  {
    param->name(node->name() + "_" + code + "_param");
    param->dtype(loco::DataType::FLOAT32);
    param->rank(1);
    param->dim(0).set(channel_size);
    param->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      param->at<loco::DataType::FLOAT32>(i) = val[i];
    }
    param->shape_status(luci::ShapeStatus::VALID);
    luci::add_origin(param, luci::get_origin(node));
  }

  // Create Scale/Shift Op
  auto ss = node->graph()->nodes()->create<luci::CircleCustom>(2, 1);
  {
    ss->name(node->name() + "_" + code);
    ss->dtype(loco::DataType::FLOAT32);
    ss->rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); i++)
      ss->dim(i).set(node->dim(i).value());
    ss->custom_code(code);

    ss->inputs(0, node);
    ss->inputs(1, param);

    ss->shape_status(luci::ShapeStatus::VALID);
    luci::add_origin(ss, luci::get_origin(node));
  }

  return ss;
}

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

std::vector<float> minus(const std::vector<float> &val)
{
  std::vector<float> res(val.size());
  for (uint32_t i = 0; i < res.size(); i++)
  {
    res[i] = -val[i];
  }
  return res;
}

// Create PreScale and insert it after node
luci::CircleCustom *create_pre_scale(luci::CircleNode *node, const EqualizePattern *p)
{
  auto param = reciprocal(p->scale);

  return create_scale_or_shift(node, param, "PreScale");
}

// Create PostScale and insert it after node
luci::CircleCustom *create_post_scale(luci::CircleNode *node, const EqualizePattern *p)
{
  return create_scale_or_shift(node, p->scale, "PostScale");
}

// Create PreShift and insert it after node
luci::CircleCustom *create_pre_shift(luci::CircleNode *node, const EqualizePattern *p)
{
  auto param = minus(p->shift);

  return create_scale_or_shift(node, param, "PreShift");
}

// Create PostShift and insert it after node
luci::CircleCustom *create_post_shift(luci::CircleNode *node, const EqualizePattern *p)
{
  return create_scale_or_shift(node, p->shift, "PostShift");
}

struct InsertScaleShiftVisitor final : public luci::CircleNodeMutableVisitor<void>
{
  InsertScaleShiftVisitor(const EqualizePattern *p) : _pattern(p)
  {
    // DO NOTHING
  }

private:
  const EqualizePattern *_pattern = nullptr;

private:
  // Generate scale/shift Ops and return the last one
  // lnode: 'front' of EqualizePattern
  // Never return nullptr
  luci::CircleCustom *gen_scale_shift(loco::Node *lnode) const
  {
    auto node = loco::must_cast<luci::CircleNode *>(lnode);

    assert(node->name() == _pattern->front); // FIX_CALLER_UNLESS

    luci::CircleCustom *bottom = nullptr;

    switch (_pattern->type)
    {
      case EqualizePattern::Type::ScaleOnly:
      {
        auto post_scale = create_post_scale(node, _pattern);
        bottom = create_pre_scale(post_scale, _pattern);
        break;
      }
      case EqualizePattern::Type::ShiftOnly:
      {
        auto post_shift = create_post_shift(node, _pattern);
        bottom = create_pre_shift(post_shift, _pattern);
        break;
      }
      case EqualizePattern::Type::ScaleShift:
      {
        auto post_scale = create_post_scale(node, _pattern);
        auto post_shift = create_post_shift(post_scale, _pattern);
        auto pre_shift = create_pre_shift(post_shift, _pattern);
        bottom = create_pre_scale(pre_shift, _pattern);
        break;
      }
      default:
        throw std::runtime_error("Unsupported EqualizePattern type");
    }

    assert(bottom != nullptr); // FIX_ME_UNLESS

    return bottom;
  }

  void visit(luci::CircleOutput *) {}

  void visit(luci::CircleNode *node) { throw std::runtime_error("NYI operator: " + node->name()); }

  void visit(luci::CircleConv2D *node)
  {
    auto bottom = gen_scale_shift(node->input());
    node->input(bottom);
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    auto bottom = gen_scale_shift(node->input());
    node->input(bottom);
  }

  void visit(luci::CircleTransposeConv *node)
  {
    auto bottom = gen_scale_shift(node->outBackprop());
    node->outBackprop(bottom);
  }

  void visit(luci::CircleInstanceNorm *node)
  {
    auto bottom = gen_scale_shift(node->input());
    node->input(bottom);
  }
  void visit(luci::CirclePad *node)
  {
    auto bottom = gen_scale_shift(node->input());
    node->input(bottom);
  }
  void visit(luci::CircleSlice *node)
  {
    auto bottom = gen_scale_shift(node->input());
    node->input(bottom);
  }
};

} // namespace

namespace fme_apply
{

void InsertScaleShift::run(loco::Graph *g)
{
  // Create a map for pattern matching
  // { back(string) -> EqualizationPattern*}
  // This assumes that each EqualizePattern has a unique 'back'
  std::map<std::string, const EqualizePattern *> pattern_by_back;
  {
    for (const auto &pattern : _patterns)
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
