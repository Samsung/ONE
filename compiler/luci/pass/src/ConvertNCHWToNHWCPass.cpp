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

#include "luci/Pass/ConvertNCHWToNHWCPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <loco/Service/ShapeInference.h>

#include <iostream>

namespace
{

luci::CircleTranspose *create_4d_transpose(luci::CircleNode *node, std::vector<int32_t> indices)
{
  assert(indices.size() == 4);

  auto perm = node->graph()->nodes()->create<luci::CircleConst>();
  perm->dtype(loco::DataType::S32);
  perm->size<loco::DataType::S32>(4);
  perm->rank(1);
  perm->dim(0) = 4;
  for (uint32_t i = 0; i < 4; i++)
    perm->at<loco::DataType::S32>(i) = indices[i];
  perm->shape_status(luci::ShapeStatus::VALID);

  auto trans = node->graph()->nodes()->create<luci::CircleTranspose>();
  trans->perm(perm);

  return trans;
}

luci::CircleTranspose *create_post_transpose(luci::CircleNode *node)
{
  return create_4d_transpose(node, {0, 3, 1, 2});
}

luci::CircleTranspose *create_pre_transpose(luci::CircleNode *node)
{
  return create_4d_transpose(node, {0, 2, 3, 1});
}

uint32_t cal_offset(loco::TensorShape &dimension, uint32_t *indices)
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
             dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

luci::CircleConst *create_NHWC_paddings(luci::CircleConst *paddings)
{
  // Assumption: paddings shape is (4,2) (it must have been checked by is_NCHW)
  assert(paddings->rank() == 2);
  assert(paddings->dim(0).value() == 4);
  assert(paddings->dim(1).value() == 2);

  auto nhwc_paddings = paddings->graph()->nodes()->create<luci::CircleConst>();
  nhwc_paddings->dtype(loco::DataType::S32);
  nhwc_paddings->shape({4, 2});
  nhwc_paddings->shape_status(luci::ShapeStatus::VALID);
  nhwc_paddings->size<loco::DataType::S32>(4 * 2);

  for (uint32_t dim = 0; dim < 4; dim++)
  {
    for (uint32_t i = 0; i < 2; i++)
    {
      int32_t data = 0;

      if (dim == 1)
      {
        // get third dimension (H in NCHW)
        data = paddings->at<loco::DataType::S32>(2 * 2 + i);
      }
      else if (dim == 2)
      {
        // get fourth dimension (W in NCHW)
        data = paddings->at<loco::DataType::S32>(3 * 2 + i);
      }

      nhwc_paddings->at<loco::DataType::S32>(dim * 2 + i) = data;
    }
  }
  return nhwc_paddings;
}

luci::CircleConst *create_NHWC_from_NCHW(luci::CircleConst *constant)
{
  // Assumption: constant is NCHW (it must have been checked by is_NCHW)
  assert(constant->rank() == 4);

  // TODO: Support non-float types
  if (constant->dtype() != loco::DataType::FLOAT32)
    throw std::runtime_error("ConvertNCHWToNHWCPass only supports FP32 model for now.");

  loco::TensorShape nchw_dimension{constant->dim(0), constant->dim(1), constant->dim(2),
                                   constant->dim(3)};
  loco::TensorShape nhwc_dimension{constant->dim(0), constant->dim(2), constant->dim(3),
                                   constant->dim(1)};

  auto nhwc_const = constant->graph()->nodes()->create<luci::CircleConst>();
  nhwc_const->dtype(loco::DataType::FLOAT32);
  nhwc_const->rank(4);
  nhwc_const->dim(0).set(constant->dim(0).value());
  nhwc_const->dim(1).set(constant->dim(2).value());
  nhwc_const->dim(2).set(constant->dim(3).value());
  nhwc_const->dim(3).set(constant->dim(1).value());
  nhwc_const->shape_status(luci::ShapeStatus::VALID);
  nhwc_const->size<loco::DataType::FLOAT32>(constant->size<loco::DataType::FLOAT32>());

  for (uint32_t n = 0; n < nchw_dimension.dim(0).value(); n++)
  {
    for (uint32_t c = 0; c < nchw_dimension.dim(1).value(); c++)
    {
      for (uint32_t h = 0; h < nchw_dimension.dim(2).value(); h++)
      {
        for (uint32_t w = 0; w < nchw_dimension.dim(3).value(); w++)
        {
          uint32_t nchw_indices[4] = {n, c, h, w};
          uint32_t nhwc_indices[4] = {n, h, w, c};
          auto data =
              constant->at<loco::DataType::FLOAT32>(cal_offset(nchw_dimension, nchw_indices));
          nhwc_const->at<loco::DataType::FLOAT32>(cal_offset(nhwc_dimension, nhwc_indices)) = data;
        }
      }
    }
  }
  return nhwc_const;
}

// We assume PAD is NCHW if,
// 1. paddings shape is (4, 2)
// 2. paddings value is [[0, 0], [0, 0], [h_t, h_b], [w_t, w_b]]]
bool is_NCHW(const luci::CirclePad *node, luci::CircleNode *&pred_node,
             luci::CircleConst *&paddings)
{
  pred_node = loco::must_cast<luci::CircleNode *>(node->input());
  paddings = loco::must_cast<luci::CircleConst *>(node->paddings());

  const auto paddings_rank = paddings->rank();
  if (paddings_rank != 2)
    return false;

  if (paddings->dim(0).value() != 4 || paddings->dim(1).value() != 2)
    return false;

  // We only check the first two dimensions,
  for (uint32_t dim = 0; dim < 2; dim++)
  {
    for (uint32_t i = 0; i < 2; i++)
    {
      auto data = paddings->at<loco::DataType::S32>(dim * 2 + i);
      if (data != 0)
        return false;
    }
  }

  return true;
}

// We assume MUL is NCHW if,
// Input shape: (N, C, H, W)
// Output shape: (N, C, H, W)
// 1. Const shape is (1, C, 1, 1)
// 2. Input, Output, Const have the same C.
bool is_NCHW_with_const(const luci::CircleMul *node, luci::CircleNode *&pred_node,
                        luci::CircleConst *&gamma)
{
  auto x = dynamic_cast<luci::CircleConst *>(node->x());
  auto y = dynamic_cast<luci::CircleConst *>(node->y());

  luci::CircleNode *pred = nullptr;
  luci::CircleConst *constant = nullptr;

  if (x != nullptr && y == nullptr)
  {
    pred = loco::must_cast<luci::CircleNode *>(node->y());
    constant = x;
  }
  else if (x == nullptr && y != nullptr)
  {
    pred = loco::must_cast<luci::CircleNode *>(node->x());
    constant = y;
  }
  else
  {
    // Ignore if MUL does not have a constant input.
    return false;
  }

  const auto const_rank = constant->rank();
  if (const_rank != 4)
    return false;

  for (uint32_t i = 0; i < const_rank; i++)
  {
    if (i == 1)
      continue;

    if (constant->dim(i).value() != 1)
      return false;
  }

  const auto const_cdim = constant->dim(1);
  const auto input_cdim = pred->dim(1);
  const auto output_cdim = node->dim(1);

  if (const_cdim == input_cdim && input_cdim == output_cdim)
  {
    pred_node = pred;
    gamma = constant;
    return true;
  }
  else
    return false;
}

// We assume ADD with const input is NCHW if,
// Input shape: (N, C, H, W)
// Output shape: (N, C, H, W)
// 1. Const shape is (1, C, 1, 1)
// 2. Input, Output, Const have the same C.
bool is_NCHW_with_const(const luci::CircleAdd *node, luci::CircleNode *&pred_node,
                        luci::CircleConst *&beta)
{
  auto x = dynamic_cast<luci::CircleConst *>(node->x());
  auto y = dynamic_cast<luci::CircleConst *>(node->y());

  if (x != nullptr && y == nullptr)
  {
    pred_node = loco::must_cast<luci::CircleNode *>(node->y());
    beta = x;
  }
  else if (x == nullptr && y != nullptr)
  {
    pred_node = loco::must_cast<luci::CircleNode *>(node->x());
    beta = y;
  }
  else
  {
    // Ignore if ADD does not have a beta input.
    return false;
  }

  const auto const_rank = beta->rank();
  if (const_rank != 4)
    return false;

  for (uint32_t i = 0; i < const_rank; i++)
  {
    if (i == 1)
      continue;

    if (beta->dim(i).value() != 1)
      return false;
  }

  const auto const_cdim = beta->dim(1);
  const auto input_cdim = pred_node->dim(1);
  const auto output_cdim = node->dim(1);

  if (const_cdim == input_cdim && input_cdim == output_cdim)
    return true;
  else
    return false;
}

class ConvertNCHWToNHWC final : public luci::CircleNodeMutableVisitor<bool>
{
  // Default
  // TODO: Throw an exception in this function
  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleInput *node)
  {
    const auto n = node->dim(0).value();
    const auto c = node->dim(1).value();
    const auto h = node->dim(2).value();
    const auto w = node->dim(3).value();

    node->dim(1).set(h);
    node->dim(2).set(w);
    node->dim(3).set(c);

    node->shape_status(luci::ShapeStatus::VALID);

    // Do shape inference again to update annotated shape.
    loco::shape_erase(node);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    // Update graph input
    auto graph_inputs = node->graph()->inputs();
    auto graph_input = graph_inputs->at(node->index());
    graph_input->shape({n, h, w, c});

    return true;
  }

  bool visit(luci::CircleOutput *node)
  {
    // Do shape inference again to update annotated shape.
    loco::shape_erase(node);

    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(node->from());

    node->from(pre_trans);

    // Update graph output
    const auto n = node->dim(0).value();
    const auto c = node->dim(1).value();
    const auto h = node->dim(2).value();
    const auto w = node->dim(3).value();

    auto graph_outputs = node->graph()->outputs();
    auto graph_output = graph_outputs->at(node->index());
    graph_output->shape({n, h, w, c});

    return true;
  }

  bool visit(luci::CircleAdd *node)
  {
    luci::CircleNode *pred_node = nullptr;
    luci::CircleConst *beta = nullptr;

    if (is_NCHW_with_const(node, pred_node, beta))
    {
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);
      node->x(pre_trans);

      auto nhwc_const = create_NHWC_from_NCHW(beta);
      node->y(nhwc_const);
    }
    else if (beta == nullptr)
    {
      // Both inputs are not constant.
      // In this case, we cannot distinguish NCHW from NHWC,
      // so just insert Transpose Ops.
      auto pre_trans_x = create_pre_transpose(node);
      pre_trans_x->a(node->x());
      node->x(pre_trans_x);

      auto pre_trans_y = create_pre_transpose(node);
      pre_trans_y->a(node->y());
      node->y(pre_trans_y);
    }
    else
    {
      return false;
    }

    // Make loco do shape inference for this node again.
    loco::shape_erase(node);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);
    return true;
  }

  bool visit(luci::CircleMul *node)
  {
    luci::CircleNode *pred_node = nullptr;
    luci::CircleConst *gamma = nullptr;

    if (is_NCHW_with_const(node, pred_node, gamma))
    {
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);
      node->x(pre_trans);

      auto nhwc_const = create_NHWC_from_NCHW(gamma);
      node->y(nhwc_const);
    }
    else
    {
      // TODO : Implement this case.
      throw std::runtime_error("Not yet implemented. Both inputs of MUL are non-const.");
    }

    // Do shape inference for this node again.
    loco::shape_erase(node);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  bool visit(luci::CirclePad *node)
  {
    luci::CircleNode *pred_node = nullptr;
    luci::CircleConst *paddings = nullptr;

    if (!is_NCHW(node, pred_node, paddings))
      return false;

    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(pred_node);
    node->input(pre_trans);

    auto nhwc_paddings = create_NHWC_paddings(paddings);
    node->paddings(nhwc_paddings);

    // Do shape inference for this node again.
    loco::shape_erase(node);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }
};

} // namespace

namespace luci
{

using DataFormat = ConvertNCHWToNHWCPass::DataFormat;

bool ConvertNCHWToNHWCPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "ConvertNCHWToNHWCPass Start" << std::endl;

  // Collect target operators
  // This is done once
  if (_op_status.empty())
  {
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      switch (circle_node->opcode())
      {
        // List of supported Ops
        case luci::CircleOpcode::CIRCLEINPUT:
        case luci::CircleOpcode::CIRCLEOUTPUT:
        case luci::CircleOpcode::ADD:
        case luci::CircleOpcode::MUL:
        case luci::CircleOpcode::PAD:
          if (_target_ops.insert(node).second)
            _op_status.emplace(node, DataFormat::NCHW);
          break;
        default:
          break;
      }
    }
  }

  bool changed = false;
  for (auto node : _target_ops)
  {
    if (_op_status.at(node) == DataFormat::NHWC)
      continue;
    else
    {
      ConvertNCHWToNHWC converter;
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      assert(circle_node->rank() == 4);
      if (circle_node->accept(&converter))
      {
        _op_status.at(node) = DataFormat::NHWC;
        changed = true;
        break;
      }
      else
      {
        throw std::runtime_error("Unsupported pattern. " + circle_node->name() +
                                 " node does not have a proper NCHW format.");
      }
    }
  }

  INFO(l) << "ConvertNCHWToNHWCPass End" << std::endl;
  return changed;
}

} // namespace luci
