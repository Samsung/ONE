/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ForwardReshapeToUnaryOpPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleShapeInference.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Service/CircleNodeClone.h>

namespace
{

luci::CircleReshape *as_reshape(loco::Node *node)
{
  return dynamic_cast<luci::CircleReshape *>(node);
}

luci::CircleConst *clone_shape(luci::CircleReshape *reshape)
{
  const auto shape = dynamic_cast<luci::CircleConst *>(reshape->shape());
  // only support CircleConst for now
  if (shape == nullptr)
    return nullptr;

  // NOTE tflite and circle only supports S32
  // TODO just check with assert() after import handles this
  auto dtype = shape->dtype();
  if (dtype != loco::DataType::S32)
    return nullptr;

  return luci::clone(shape);
}

void copy_shape(luci::CircleReshape *reshape, luci::CircleReshape *new_reshape)
{
  auto ns_rank = reshape->newShape()->rank();
  new_reshape->newShape()->rank(ns_rank);
  for (uint32_t r = 0; r < ns_rank; ++r)
    new_reshape->newShape()->dim(r) = reshape->newShape()->dim(r);
}

luci::CircleReshape *create_cloned_reshape(luci::CircleReshape *reshape)
{
  assert(reshape != nullptr); // FIX_CALLER_UNLESS

  luci::CircleConst *cloned_shape = clone_shape(reshape);
  if (cloned_shape == nullptr)
    return nullptr;

  auto cloned_node = luci::clone_node(reshape, reshape->graph());
  if (cloned_node == nullptr)
    return nullptr;

  auto new_reshape = loco::must_cast<luci::CircleReshape *>(cloned_node);
  new_reshape->shape(cloned_shape);
  new_reshape->name(reshape->name() + "_C");
  luci::add_origin(new_reshape, luci::get_origin(reshape));

  return new_reshape;
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleMean *mean, uint32_t axis)
{
  assert(reshape != nullptr); // FIX_CALLER_UNLESS
  assert(mean != nullptr);    // FIX_CALLER_UNLESS

  auto new_reshape = create_cloned_reshape(reshape);
  if (not new_reshape)
    return false;

  // reconnect network
  loco::replace(mean).with(new_reshape);
  mean->input(reshape->tensor());
  new_reshape->tensor(mean);

  // Change const shape axis value
  auto *shape_reshape = loco::must_cast<luci::CircleConst *>(new_reshape->shape());
  assert(shape_reshape->dtype() == loco::DataType::S32);     // FIX_CALLER_UNLESS
  assert(axis < shape_reshape->size<loco::DataType::S32>()); // FIX_CALLER_UNLESS
  shape_reshape->at<loco::DataType::S32>(axis) = 1;

  // Do shape inference for this node again.
  mean->shape_status(luci::ShapeStatus::UNDEFINED);
  reshape->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleAbs *abs)
{
  assert(reshape != nullptr); // FIX_CALLER_UNLESS
  assert(abs != nullptr);     // FIX_CALLER_UNLESS

  auto new_reshape = create_cloned_reshape(reshape);
  if (not new_reshape)
    return false;

  // reconnect network
  loco::replace(abs).with(new_reshape);
  abs->x(reshape->tensor());
  new_reshape->tensor(abs);

  // Do shape inference for this node again.
  abs->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleNeg *neg)
{
  assert(reshape != nullptr);
  assert(neg != nullptr);

  luci::CircleConst *cloned_shape = clone_shape(reshape);
  if (cloned_shape == nullptr)
    return false;

  auto name = reshape->name();
  assert(name.length() > 0);
  loco::Graph *graph = neg->graph();
  // create reshape placed after neg
  luci::CircleReshape *new_reshape = graph->nodes()->create<luci::CircleReshape>();
  copy_shape(reshape, new_reshape);
  new_reshape->shape(cloned_shape);
  new_reshape->name(name + "_C");
  luci::add_origin(new_reshape, luci::get_origin(reshape));

  // reconnect network
  loco::replace(neg).with(new_reshape);
  neg->x(reshape->tensor());
  new_reshape->tensor(neg);

  // Do shape inference for this node again.
  neg->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleLogistic *logit)
{
  assert(reshape != nullptr); // FIX_CALLER_UNLESS
  assert(logit != nullptr);   // FIX_CALLER_UNLESS

  auto new_reshape = create_cloned_reshape(reshape);
  if (not new_reshape)
    return false;

  // reconnect network
  loco::replace(logit).with(new_reshape);
  logit->x(reshape->tensor());
  new_reshape->tensor(logit);

  // Do shape inference for this node again.
  logit->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleMul *div,
                     luci::CircleConst *const_value)
{
  assert(reshape != nullptr); // FIX_CALLER_UNLESS
  assert(div != nullptr);     // FIX_CALLER_UNLESS

  auto new_reshape = create_cloned_reshape(reshape);
  if (not new_reshape)
    return false;

  // reconnect network
  loco::replace(div).with(new_reshape);
  if (div->x() == const_value)
  {
    div->y(reshape->tensor());
  }
  else
  {
    assert(div->y() == const_value);
    div->x(reshape->tensor());
  }
  new_reshape->tensor(div);

  // Do shape inference for this node again.
  div->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleDiv *div,
                     luci::CircleConst *const_value)
{
  assert(reshape != nullptr); // FIX_CALLER_UNLESS
  assert(div != nullptr);     // FIX_CALLER_UNLESS

  auto new_reshape = create_cloned_reshape(reshape);
  if (not new_reshape)
    return false;

  // reconnect network
  loco::replace(div).with(new_reshape);
  if (div->x() == const_value)
  {
    div->y(reshape->tensor());
  }
  else
  {
    assert(div->y() == const_value);
    div->x(reshape->tensor());
  }
  new_reshape->tensor(div);

  // Do shape inference for this node again.
  div->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

class ForwardReshape final : public luci::CircleNodeMutableVisitor<bool>
{
protected:
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "ForwardReshape: Unsupported operator: " << node->name() << std::endl;
    return false;
  }

  /**
   * Graph example:
   *
   *  BEFORE
   *               [Input]
   *              (3, 4, 4)                 [Shape_Const = (1, -1, 4)]
   *                  |                     |
   *              [Reshape] ----------------
   *              (1, 12, 4)
   *                  |
   *        [Mean, keep_dims = true]
   *              (1, 12, 1)
   *                  |
   *               [Output]
   *
   *  AFTER
   *               [Input]
   *              (3, 4, 4)
   *                  |
   *         [Mean, keep_dims = true]
   *              (3, 4, 1)                 [Shape_Const = (1, -1, 1)]
   *                  |                     |
   *              [Reshape]-----------------
   *              (1, 12, 1)
   *                  |
   *              [Output]
   *
   */
  bool visit(luci::CircleMean *node)
  {
    luci::CircleReshape *reshape = nullptr;
    luci::CircleConst *axis = nullptr;

    reshape = dynamic_cast<luci::CircleReshape *>(node->input());
    axis = dynamic_cast<luci::CircleConst *>(node->reduction_indices());

    if (reshape == nullptr or axis == nullptr)
      return false;

    if (axis->dtype() != loco::DataType::S32)
      return false;

    // Should be scalar
    if (axis->size<loco::DataType::S32>() != 1)
      return false;

    // axis value
    auto axis_value = axis->at<loco::DataType::S32>(0);

    if (axis_value < 0)
      axis_value += static_cast<int32_t>(reshape->rank());

    assert(axis_value >= 0);

    if (node->keep_dims() != true)
      return false;

    auto reshape_input = loco::must_cast<luci::CircleNode *>(reshape->tensor());

    // reshape shouldn't change rank
    if (reshape_input->rank() != reshape->rank())
      return false;

    assert(reshape_input->rank() > static_cast<uint32_t>(axis_value));

    for (int32_t i = 0; i <= axis_value; ++i)
    {
      if (not reshape_input->dim(i).known() or
          reshape_input->dim(i).value() != reshape->dim(i).value())
        return false;
    }

    return forward_reshape(reshape, node, axis_value);
  }

  bool visit(luci::CircleAbs *node)
  {
    auto reshape = as_reshape(node->x());
    if (reshape == nullptr)
      return false;
    return forward_reshape(reshape, node);
  }

  bool visit(luci::CircleNeg *node)
  {
    auto reshape = as_reshape(node->x());
    if (reshape == nullptr)
      return false;
    return forward_reshape(reshape, node);
  }

  bool visit(luci::CircleLogistic *node)
  {
    auto reshape = as_reshape(node->x());
    if (reshape == nullptr)
      return false;

    return forward_reshape(reshape, node);
  }

  bool visit(luci::CircleDiv *node)
  {
    luci::CircleReshape *reshape = nullptr;
    luci::CircleConst *const_value = nullptr;

    if (not luci::fill(&reshape, &const_value).with_commutative_args_of(node))
      return false;

    if (const_value->dtype() != loco::DataType::FLOAT32)
      return false;

    // Should be scalar
    if (const_value->size<loco::DataType::FLOAT32>() != 1)
      return false;

    return forward_reshape(reshape, node, const_value);
  }

  bool visit(luci::CircleMul *node)
  {
    luci::CircleReshape *reshape = nullptr;
    luci::CircleConst *const_value = nullptr;

    if (not luci::fill(&reshape, &const_value).with_commutative_args_of(node))
      return false;

    if (const_value->dtype() != loco::DataType::FLOAT32)
      return false;

    // Should be scalar
    if (const_value->size<loco::DataType::FLOAT32>() != 1)
      return false;

    return forward_reshape(reshape, node, const_value);
  }

  // TODO add more unary operators
};

} // namespace

namespace luci
{

/**
 * BEFORE
 *                       |
 *                  [CircleNode]  [CircleConst]
 *                       |       /
 *                 [CircleReshape]
 *                /      |
 *     [CircleNode]  [(UnaryOp)]
 *          |            |     \
 *          |            |      [CircleNode]
 *          |            |           |
 *
 *   UnaryOp: CircleNeg, ...
 *   Note: Binary Op (Div, Mul) can also be considered as a unary operation
 *         if one of its inputs is a constant.
 *         For CircleMean in which the axis is a scalar
 *         constant and reshape Op does not change the axis on which the mean is
 *         taken, the Reshape Op can be forwarded.
 *
 * AFTER
 *                       |
 *   [CircleConst]  [CircleNode]
 *         |       /     |
 *  [CircleReshape] [(UnaryOp)] [CircleConst]
 *         |             |      /
 *   [CircleNode] [CircleReshape]
 *         |             |      \
 *         |             |       [CircleNode]
 *         |             |            |
 *
 *   Note: new [CircleReshape] after [(UnaryOp)] added
 */
bool ForwardReshapeToUnaryOpPass::run(loco::Graph *g)
{
  bool changed = false;
  ForwardReshape forward;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->accept(&forward))
      changed = true;
  }
  return changed;
}

} // namespace luci
