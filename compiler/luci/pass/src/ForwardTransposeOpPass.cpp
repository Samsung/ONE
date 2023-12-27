/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ForwardTransposeOpPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Service/CircleNodeClone.h>

using namespace luci;

namespace
{

// Create new Transpose Op including perm
// Return nullptr if failed
CircleTranspose *create_cloned_transpose(CircleTranspose *transpose)
{
  assert(transpose != nullptr); // FIX_CALLER_UNLESS

  auto perm = dynamic_cast<CircleConst *>(transpose->perm());
  if (not perm)
    return nullptr;

  CircleConst *cloned_perm = clone(perm);
  if (cloned_perm == nullptr)
    return nullptr;

  cloned_perm->name(perm->name() + "_C");
  luci::add_origin(cloned_perm, luci::get_origin(perm));

  auto cloned_node = clone_node(transpose, transpose->graph());
  if (cloned_node == nullptr)
    return nullptr;

  auto new_transpose = loco::must_cast<luci::CircleTranspose *>(cloned_node);
  new_transpose->perm(cloned_perm);
  new_transpose->name(transpose->name() + "_C");
  luci::add_origin(new_transpose, luci::get_origin(transpose));

  return new_transpose;
}

uint32_t cal_offset(const std::vector<uint32_t> &shape, const std::vector<uint32_t> &indices)
{
  assert(shape.size() == indices.size()); // FIX_CALLER_UNLESS

  uint32_t offset = 0;
  for (uint32_t i = 0; i < indices.size(); i++)
  {
    uint32_t index = indices[i];
    for (uint32_t j = shape.size() - 1; j > i; j--)
    {
      index *= shape[j];
    }
    offset += index;
  }
  return offset;
}

// Return reverse-transpose of 'node'
// i.e., Transpose(return value) = node
CircleConst *reverse_transposed(CircleConst *node, std::vector<uint32_t> &t)
{
  assert(node->rank() == t.size()); // FIX_CALLER_UNLESS
  assert(node->rank() == 4);        // FIX_CALLER_UNLESS

  std::vector<uint32_t> orig_shape(node->rank());
  std::vector<uint32_t> new_shape(node->rank());

  for (uint32_t i = 0; i < node->rank(); i++)
  {
    assert(t[i] < node->rank()); // FIX_CALLER_UNLESS

    orig_shape[i] = node->dim(i).value();
    new_shape[t[i]] = node->dim(i).value();
  }

  auto clone_const = clone(node);
  for (uint32_t i = 0; i < node->rank(); i++)
    clone_const->dim(i).set(new_shape[i]);

  clone_const->name(clone_const->name() + "_r_transposed");
  add_origin(clone_const, luci::get_origin(node));

  for (uint32_t n = 0; n < clone_const->dim(0).value(); n++)
  {
    for (uint32_t h = 0; h < clone_const->dim(1).value(); h++)
    {
      for (uint32_t w = 0; w < clone_const->dim(2).value(); w++)
      {
        for (uint32_t c = 0; c < clone_const->dim(3).value(); c++)
        {
          std::vector<uint32_t> new_indices{n, h, w, c};
          std::vector<uint32_t> orig_indices{new_indices[t[0]], new_indices[t[1]],
                                             new_indices[t[2]], new_indices[t[3]]};

          const auto data = node->at<loco::DataType::FLOAT32>(cal_offset(orig_shape, orig_indices));
          clone_const->at<loco::DataType::FLOAT32>(cal_offset(new_shape, new_indices)) = data;
        }
      }
    }
  }

  return clone_const;
}

bool check_rank_four(const CircleConst *c) { return c->rank() == 4; }

// Return true if below conditions are met
// 1. t->perm() is CircleConst
// 2. t->perm() is S32
bool check_perm(const CircleTranspose *t)
{
  auto perm = dynamic_cast<CircleConst *>(t->perm());
  if (not perm)
    return false;

  switch (perm->dtype())
  {
    case loco::DataType::S32:
      for (uint32_t i = 0; i < perm->size<loco::DataType::S32>(); i++)
      {
        auto data = perm->at<loco::DataType::S32>(i);
        // TODO Support not normalized index
        if (data < 0 or data >= static_cast<int32_t>(t->rank()))
          return false;
      }
      break;
    // TODO Support S64 data type
    default:
      return false;
  }

  return true;
}

#define RETURN_FALSE_UNLESS(COND) \
  if (not(COND))                  \
    return false;

// Elementwise Binary Operator with const
class EBOWithConstPattern final : public CircleNodeMutableVisitor<bool>
{
private:
  template <typename CIRCLE_OP_PTR> bool has_pattern(CIRCLE_OP_PTR node)
  {
    if (auto x = dynamic_cast<luci::CircleConst *>(node->x()))
    {
      if (auto y = dynamic_cast<luci::CircleTranspose *>(node->y()))
      {
        RETURN_FALSE_UNLESS(check_rank_four(x));
        RETURN_FALSE_UNLESS(check_perm(y));

        auto new_const = gen_new_const(y, x);
        assert(new_const); // FIX_ME_UNLESS

        auto new_transpose = create_cloned_transpose(y);
        assert(new_transpose); // FIX_ME_UNLESS

        // Reconnect network
        node->x(new_const);
        node->y(y->a());
        loco::replace(node).with(new_transpose);
        new_transpose->a(node);

        // Do shape inference for this node again.
        node->shape_status(luci::ShapeStatus::UNDEFINED);

        return true;
      }
    }

    if (auto y = dynamic_cast<luci::CircleConst *>(node->y()))
    {
      if (auto x = dynamic_cast<luci::CircleTranspose *>(node->x()))
      {
        RETURN_FALSE_UNLESS(check_rank_four(y));
        RETURN_FALSE_UNLESS(check_perm(x));

        auto new_const = gen_new_const(x, y);
        assert(new_const); // FIX_ME_UNLESS

        auto new_transpose = create_cloned_transpose(x);
        assert(new_transpose); // FIX_ME_UNLESS

        // Reconnect network
        node->y(new_const);
        node->x(x->a());
        loco::replace(node).with(new_transpose);
        new_transpose->a(node);

        // Do shape inference for this node again.
        node->shape_status(luci::ShapeStatus::UNDEFINED);

        return true;
      }
    }

    return false;
  }

public:
  // Default
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleAdd *node) { return has_pattern(node); }

  bool visit(luci::CircleMul *node) { return has_pattern(node); }

private:
  // Return a new const node after Tranpose Op is forwarded
  // Return nullptr if unsupported cases
  CircleConst *gen_new_const(CircleTranspose *t, CircleConst *c)
  {
    const auto perm = dynamic_cast<CircleConst *>(t->perm());

    // Only support constant perm
    if (not perm)
      return nullptr;

    std::vector<uint32_t> perm_data;
    switch (perm->dtype())
    {
      case loco::DataType::S32:
        for (uint32_t i = 0; i < perm->size<loco::DataType::S32>(); i++)
        {
          auto data = perm->at<loco::DataType::S32>(i);
          assert(data >= 0 and data < static_cast<int32_t>(t->rank()));
          perm_data.emplace_back(static_cast<uint32_t>(data));
        }
        break;
      // TODO Support S64 data type
      default:
        return nullptr;
    }

    assert(perm_data.size() == t->rank()); // FIX_CALLER_UNLESS

    return reverse_transposed(c, perm_data);
  }
};

// Elementwise Unary Operator
class EwUnaryPattern final : public CircleNodeMutableVisitor<bool>
{
private:
  // input is 'x'
  template <typename CIRCLE_OP_PTR> bool has_pattern_x(CIRCLE_OP_PTR node)
  {
    if (auto x = dynamic_cast<luci::CircleTranspose *>(node->x()))
    {
      RETURN_FALSE_UNLESS(check_perm(x));

      auto new_transpose = create_cloned_transpose(x);
      assert(new_transpose); // FIX_ME_UNLESS

      // Reconnect network
      node->x(x->a());
      loco::replace(node).with(new_transpose);
      new_transpose->a(node);

      // Do shape inference for this node again.
      node->shape_status(luci::ShapeStatus::UNDEFINED);

      return true;
    }

    return false;
  }

public:
  // Default
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleAbs *node) { return has_pattern_x(node); }

  bool visit(luci::CircleLogistic *node) { return has_pattern_x(node); }
};

} // namespace

namespace luci
{

/**
 * BEFORE
 *                       |
 *                  [CircleNode]  [CircleConst]
 *                       |       /
 *              [CircleTranspose] [CircleConst]
 *                /      |       /
 *     [CircleNode]  [(BinaryOp)]
 *          |            |     \
 *          |            |      [CircleNode]
 *          |            |           |
 *
 *   BinaryOp: CircleAdd, CircleMul, ...
 *
 *                       |
 *                  [CircleNode]  [CircleConst]
 *                       |       /
 *              [CircleTranspose]
 *                /      |
 *     [CircleNode]  [(UnaryOp)]
 *          |            |     \
 *          |            |      [CircleNode]
 *          |            |           |
 *
 *   UnaryOp: CircleAbs, ...
 *
 * AFTER
 *                       |
 *   [CircleConst]  [CircleNode]  [CircleConst(updated)]
 *         |       /     |       /
 *  [CircleTranspose] [(BinaryOp)] [CircleConst]
 *         |             |        /
 *   [CircleNode] [CircleTranspose]
 *         |             |      \
 *         |             |       [CircleNode]
 *         |             |            |
 *
 *                       |
 *   [CircleConst]  [CircleNode]
 *         |       /     |
 *  [CircleTranspose] [(UnaryOp)] [CircleConst]
 *         |             |        /
 *   [CircleNode] [CircleTranspose]
 *         |             |      \
 *         |             |       [CircleNode]
 *         |             |            |
 *
 *   Note: new [CircleTranspose] is added after [(BinaryOp)]
 */
bool ForwardTransposeOpPass::run(loco::Graph *g)
{
  bool changed = false;
  EBOWithConstPattern eboc;
  EwUnaryPattern ewu;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->accept(&eboc))
      changed = true;
    else if (circle_node->accept(&ewu))
      changed = true;
  }
  return changed;
}

#undef RETURN_FALSE_UNLESS

} // namespace luci
