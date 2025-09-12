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

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Service/CircleNodeClone.h>

using namespace luci;

#define RETURN_FALSE_UNLESS(COND) \
  if (not(COND))                  \
    return false;

namespace
{

// Return true if below conditions are met
// 1. t->perm() is CircleConst
// 2. t->perm() is S32
bool check_perm(const CircleTranspose *t)
{
  auto perm = dynamic_cast<CircleConst *>(t->perm());
  RETURN_FALSE_UNLESS(perm);

  switch (perm->dtype())
  {
    case loco::DataType::S32:
      for (uint32_t i = 0; i < perm->size<loco::DataType::S32>(); i++)
      {
        auto data = perm->at<loco::DataType::S32>(i);
        // TODO Support not normalized index
        RETURN_FALSE_UNLESS(data >= 0 and data < static_cast<int32_t>(t->rank()));
      }
      break;
    // TODO Support S64 data type
    default:
      return false;
  }

  return true;
}

// Return vector of int32_t from CircleConst node
// Return empty vector if not supported
std::vector<int32_t> get_perm_data(const CircleConst *node)
{
  assert(node); // FIX_CALLER_UNLESS
  std::vector<int32_t> perm_data;
  switch (node->dtype())
  {
    case loco::DataType::S32:
      for (uint32_t i = 0; i < node->size<loco::DataType::S32>(); i++)
      {
        auto data = node->at<loco::DataType::S32>(i);

        // Unsupported
        if (data < 0 or data >= static_cast<int32_t>(node->size<loco::DataType::S32>()))
          return {};

        perm_data.emplace_back(data);
      }
      break;
    // TODO Support S64 data type
    default:
      break;
  }

  return perm_data;
}

// Return true if below conditions are met
// 1. lhs->perm() and rhs->perm() are CircleConst
// 2. Both perm's values are the same
bool check_same_perm(const CircleTranspose *lhs, const CircleTranspose *rhs)
{
  auto lhs_perm = dynamic_cast<CircleConst *>(lhs->perm());
  RETURN_FALSE_UNLESS(lhs_perm);

  auto rhs_perm = dynamic_cast<CircleConst *>(rhs->perm());
  RETURN_FALSE_UNLESS(rhs_perm);

  std::vector<int32_t> lhs_perm_data = get_perm_data(lhs_perm);
  RETURN_FALSE_UNLESS(not lhs_perm_data.empty());

  std::vector<int32_t> rhs_perm_data = get_perm_data(rhs_perm);
  RETURN_FALSE_UNLESS(not rhs_perm_data.empty());

  RETURN_FALSE_UNLESS(lhs_perm_data == rhs_perm_data);

  return true;
}

// Create new Transpose Op including perm
// Never return nullptr
CircleTranspose *create_cloned_transpose(CircleTranspose *transpose)
{
  assert(transpose != nullptr);  // FIX_CALLER_UNLESS
  assert(check_perm(transpose)); // FIX_CALLER_UNLESS

  auto perm = luci::must_cast<CircleConst *>(transpose->perm());
  assert(perm); // FIX_CALLER_UNLESS

  CircleConst *cloned_perm = clone(perm);
  assert(cloned_perm); // FIX_ME_UNLESS

  cloned_perm->name(perm->name() + "_C");
  luci::add_origin(cloned_perm, luci::get_origin(perm));

  auto cloned_node = clone_node(transpose, transpose->graph());
  assert(cloned_node); // FIX_ME_UNLESS

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

bool has_single_element(const luci::CircleConst *node)
{
  bool has_single_elem = false;
  switch (node->dtype())
  {
    case loco::DataType::FLOAT32:
      has_single_elem = node->size<loco::DataType::FLOAT32>() == 1;
      break;
    default:
      // NYI
      break;
  }

  if (has_single_elem)
  {
    for (uint32_t i = 0; i < node->rank(); i++)
      assert(node->dim(i).value() == 1); // FIX_ME_UNLESS
  }

  return has_single_elem;
}

// Elementwise Binary Operator with const
class EBOWithConstPattern final : public CircleNodeMutableVisitor<bool>
{
private:
  template <typename CIRCLE_OP_PTR> bool has_commutative_xy(CIRCLE_OP_PTR node)
  {
    luci::CircleTranspose *transpose = nullptr;
    luci::CircleConst *const_value = nullptr;

    RETURN_FALSE_UNLESS(luci::fill(&transpose, &const_value).with_commutative_args_of(node));

    if (has_single_element(const_value))
    {
      RETURN_FALSE_UNLESS(check_perm(transpose));
      auto new_transpose = create_cloned_transpose(transpose);
      assert(new_transpose); // FIX_ME_UNLESS

      if (node->x() == const_value)
      {
        node->y(transpose->a());
      }
      else
      {
        assert(node->y() == const_value);
        node->x(transpose->a());
      }
      loco::replace(node).with(new_transpose);
      new_transpose->a(node);

      // Do shape inference for this node again.
      node->shape_status(luci::ShapeStatus::UNDEFINED);

      return true;
    }
    else if (const_value->rank() == transpose->rank())
    {
      // Only support rank 4 for now
      RETURN_FALSE_UNLESS(check_rank_four(const_value));
      RETURN_FALSE_UNLESS(check_perm(transpose));

      auto new_const = gen_new_const(transpose, const_value);
      assert(new_const); // FIX_ME_UNLESS

      auto new_transpose = create_cloned_transpose(transpose);
      assert(new_transpose); // FIX_ME_UNLESS

      // Reconnect network
      if (node->x() == const_value)
      {
        node->x(new_const);
        node->y(transpose->a());
      }
      else
      {
        node->x(transpose->a());
        node->y(new_const);
      }

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

  bool visit(luci::CircleAdd *node) { return has_commutative_xy(node); }

  bool visit(luci::CircleMul *node) { return has_commutative_xy(node); }

private:
  // Return a new const node after Tranpose Op is forwarded
  // Never return nullptr
  CircleConst *gen_new_const(CircleTranspose *t, CircleConst *c)
  {
    assert(t);             // FIX_CALLER_UNLESS
    assert(c);             // FIX_CALLER_UNLESS
    assert(check_perm(t)); // FIX_CALLER_UNLESS

    const auto perm = luci::must_cast<CircleConst *>(t->perm());
    assert(perm); // FIX_ME_UNLESS

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
        // Unreachable. FIX_check_perm_UNLESS
        throw std::runtime_error("Unsupported dtype");
    }

    assert(perm_data.size() == t->rank()); // FIX_CALLER_UNLESS

    return reverse_transposed(c, perm_data);
  }
};

// Elementwise Binary Operator (no const input)
class EBOPattern final : public CircleNodeMutableVisitor<bool>
{
private:
  template <typename CIRCLE_OP_PTR> bool has_transpose_xy(CIRCLE_OP_PTR node)
  {
    luci::CircleTranspose *lhs = nullptr;
    luci::CircleTranspose *rhs = nullptr;

    RETURN_FALSE_UNLESS(luci::fill(&lhs, &rhs).with_args_of(node));

    // Check lhs's perm == rhs's perm
    RETURN_FALSE_UNLESS(check_same_perm(lhs, rhs));

    // Create cloned transpose
    auto new_transpose = create_cloned_transpose(lhs);
    assert(new_transpose); // FIX_ME_UNLESS

    // Reconnect network
    node->x(lhs->a());
    node->y(rhs->a());

    loco::replace(node).with(new_transpose);
    new_transpose->a(node);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    return true;
  }

public:
  // Default
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleAdd *node) { return has_transpose_xy(node); }

  bool visit(luci::CircleMul *node) { return has_transpose_xy(node); }
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

  // input is 'features'
  template <typename CIRCLE_OP_PTR> bool has_pattern_features(CIRCLE_OP_PTR node)
  {
    if (auto tr = dynamic_cast<luci::CircleTranspose *>(node->features()))
    {
      RETURN_FALSE_UNLESS(check_perm(tr));

      auto new_transpose = create_cloned_transpose(tr);
      assert(new_transpose); // FIX_ME_UNLESS

      // Reconnect network
      node->features(tr->a());
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

  bool visit(luci::CircleRelu6 *node) { return has_pattern_features(node); }
};

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 *                 [CircleNode]     [CircleNode]
 *                       |              |
 *              [CircleTranspose] [CircleTranspose]
 *                       |       /
 *                    [(BinaryOp)]
 *                         |
 *
 *   BinaryOp: CircleAdd, CircleMul, ...
 *
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
 *
 *              [CircleNode] [CircleNode]
 *                       |       /
 *                    [(BinaryOp)]
 *                         |
 *                 [CircleTranspose]
 *                         |
 *
 *   BinaryOp: CircleAdd, CircleMul, ...
 *
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

  // TODO Revisit pattern interface
  EBOPattern ebo;
  EBOWithConstPattern eboc;
  EwUnaryPattern ewu;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->accept(&eboc))
      changed = true;
    else if (circle_node->accept(&ewu))
      changed = true;
    else if (circle_node->accept(&ebo))
      changed = true;
  }
  return changed;
}

#undef RETURN_FALSE_UNLESS

} // namespace luci
