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

namespace
{

enum class DataFormat
{
  NCHW,
  NHWC
};

/**
 * @brief Set annotation for DataFormat (NCHW, NHWC)
 *
 * @note DataFormatAnnotation will live longer than this Pass (until the
 *       annotated loco::Node is erased). So, do not use large data in the
 *       annotation to avoid excessive memory usage.
 */
class DataFormatAnnotation final : public loco::NodeAnnotation
{
public:
  DataFormatAnnotation(const DataFormat &format) : _format{format}
  {
    // DO NOTHING
  }

public:
  const DataFormat &format(void) const { return _format; }

private:
  DataFormat _format;
};

void set_data_format(loco::Node *node, const DataFormat &format)
{
  assert(node->annot<DataFormatAnnotation>() == nullptr);
  node->annot(std::make_unique<DataFormatAnnotation>(format));
}

DataFormat get_data_format(loco::Node *node)
{
  assert(node->annot<DataFormatAnnotation>() != nullptr);
  return node->annot<DataFormatAnnotation>()->format();
}

bool has_data_format(loco::Node *node) { return node->annot<DataFormatAnnotation>() != nullptr; }

luci::CircleTranspose *create_4d_transpose(luci::CircleNode *node,
                                           const std::vector<int32_t> indices)
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

uint32_t cal_offset(const loco::TensorShape &dimension, const uint32_t *indices)
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
           dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

luci::CircleConst *create_NHWC_from_NCHW(luci::CircleConst *constant)
{
  LOGGER(l);
  assert(constant->rank() == 4);

  // TODO: Support non-float types
  if (constant->dtype() != loco::DataType::FLOAT32)
  {
    INFO(l) << "Non-float type constant: " << constant->name() << std::endl;
    return nullptr;
  }

  loco::TensorShape nchw_dimension{constant->dim(0), constant->dim(1), constant->dim(2),
                                   constant->dim(3)};
  loco::TensorShape nhwc_dimension{constant->dim(0), constant->dim(2), constant->dim(3),
                                   constant->dim(1)};

  auto nhwc_const = constant->graph()->nodes()->create<luci::CircleConst>();
  nhwc_const->dtype(constant->dtype());
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
    // Ignore if ADD does not have a constant input.
    return false;
  }

  const auto const_rank = beta->rank();
  if (const_rank != 4)
    return false;

  // Check the shape is (1, C, 1, 1)
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

  // Check Input, Output, Const have the same channel size
  if (const_cdim == input_cdim && input_cdim == output_cdim)
    return true;
  else
    return false;
}

class ConvertNCHWToNHWC final : public luci::CircleNodeMutableVisitor<bool>
{
  // Default
  bool visit(luci::CircleNode *node)
  {
    throw std::runtime_error(node->name() + " is an unsupported operator.");
  }

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

    loco::shape_erase(node);

    // Insert post-tranpose
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
    // Insert pre-transpose
    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(node->from());

    node->from(pre_trans);

    loco::shape_erase(node);

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

      auto nhwc_const = create_NHWC_from_NCHW(beta);
      if (nhwc_const == nullptr)
        return false;

      node->x(pre_trans);
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
};

} // namespace

namespace luci
{

bool ConvertNCHWToNHWCPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "ConvertNCHWToNHWCPass Start" << std::endl;

  // Annotate NCHW operators
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    switch (circle_node->opcode())
    {
      // List of supported Ops
      case luci::CircleOpcode::CIRCLEINPUT:
      case luci::CircleOpcode::CIRCLEOUTPUT:
      case luci::CircleOpcode::ADD:
        if (!has_data_format(node))
        {
          set_data_format(node, DataFormat::NCHW);
        }
        break;
      default:
        break;
    }
  }

  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (!has_data_format(node))
    {
      // Unsupported Op
      continue;
    }
    else if (get_data_format(node) == DataFormat::NHWC)
    {
      // Already converted to NHWC
      continue;
    }
    else
    {
      ConvertNCHWToNHWC converter;
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      assert(circle_node->rank() == 4);
      if (circle_node->accept(&converter))
      {
        set_data_format(node, DataFormat::NHWC);
        changed = true;
        break;
      }
      else
      {
        continue;
      }
    }
  }

  INFO(l) << "ConvertNCHWToNHWCPass End" << std::endl;
  return changed;
}

} // namespace luci
