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
#include "CircleOptimizerUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <functional>

namespace
{

// Return true if from can be broadcasted to to
// to's shape is [N, C, H, W]
bool broadcastable(const luci::CircleConst *from, const luci::CircleNode *to)
{
  assert(to->rank() == 4); // FIX_CALLER_UNLESS

  const auto from_rank = from->rank();
  if (from_rank > 4)
    return false;

  // Scalar is always broadcastable
  if (from_rank == 0)
    return true;

  for (uint32_t i = 1; i <= from_rank; i++)
  {
    auto to_index = 4 - i;
    auto from_index = from_rank - i;

    if (from->dim(from_index).value() != to->dim(to_index).value() and
        from->dim(from_index).value() != 1)
      return false;
  }

  return true;
}

// Return node with rank 4
// node should have rank less than or equal to 4
// 1 is inserted to the front of shape if rank is less than 4
// For example, [2] -> [1, 1, 1, 2]
luci::CircleConst *expand_to_rank_4(luci::CircleConst *node)
{
  auto original_rank = node->rank();

  assert(original_rank <= 4); // FIX_CALLER_UNLESS

  if (original_rank == 4)
    return node;

  std::vector<uint32_t> original_shape;
  for (uint32_t i = 0; i < original_rank; i++)
  {
    original_shape.emplace_back(node->dim(i).value());
  }

  auto cloned = luci::clone(node);
  cloned->name(cloned->name() + "_rank4");

  cloned->rank(4);
  for (uint32_t i = 0; i < (4 - original_rank); i++)
    cloned->dim(i) = 1;

  for (uint32_t i = 0; i < original_rank; i++)
    cloned->dim(i + (4 - original_rank)) = original_shape.at(i);

  return cloned;
}

bool is_output(const loco::Node *node)
{
  auto cnode = loco::must_cast<const luci::CircleNode *>(node);
  auto opcode = cnode->opcode();
  if (opcode == luci::CircleOpcode::CIRCLEOUTPUT ||
      opcode == luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
    return true;

  return false;
}

bool is_same_shape(const luci::CircleNode *node, const std::vector<loco::Dimension> &shape)
{
  if (not node)
    return false;

  if (shape.size() != node->rank())
    return false;

  for (uint32_t i = 0; i < shape.size(); i++)
  {
    if (not(node->dim(i) == shape[i]))
      return false;
  }
  return true;
}

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
  node->annot(std::make_unique<DataFormatAnnotation>(format));
}

DataFormat get_data_format(loco::Node *node)
{
  assert(node->annot<DataFormatAnnotation>() != nullptr);
  return node->annot<DataFormatAnnotation>()->format();
}

bool has_data_format(loco::Node *node) { return node->annot<DataFormatAnnotation>() != nullptr; }

bool check_4d_transpose(loco::Node *node, const std::vector<int32_t> indices)
{
  assert(indices.size() == 4);

  auto trans = dynamic_cast<luci::CircleTranspose *>(node);
  if (not trans)
    return false;

  if (not trans->perm())
    return false;

  auto perm = dynamic_cast<luci::CircleConst *>(trans->perm());
  // Only const perm is supported
  if (not perm)
    return false;

  if (perm->dtype() != loco::DataType::S32)
    return false;

  if (perm->size<loco::DataType::S32>() != 4)
    return false;

  for (uint32_t i = 0; i < 4; i++)
  {
    if (perm->at<loco::DataType::S32>(i) != indices[i])
      return false;
  }

  return true;
}

luci::CircleTranspose *create_4d_transpose(luci::CircleNode *node,
                                           const std::vector<int32_t> indices)
{
  assert(indices.size() == 4);

  auto name = node->name();
  assert(name.length() > 0);

  auto perm = node->graph()->nodes()->create<luci::CircleConst>();
  perm->dtype(loco::DataType::S32);
  perm->size<loco::DataType::S32>(4);
  perm->rank(1);
  perm->dim(0) = 4;
  for (uint32_t i = 0; i < 4; i++)
    perm->at<loco::DataType::S32>(i) = indices[i];
  perm->shape_status(luci::ShapeStatus::VALID);

  auto make_string = [](const std::vector<int32_t> &nums) {
    std::string str;
    for (auto num : nums)
    {
      if (str.length() > 0)
        str += ".";
      str += std::to_string(num);
    }
    return str;
  };

  auto str_indices = make_string(indices);

  perm->name(name + "/Transpose_" + str_indices + "/perm");

  auto trans = node->graph()->nodes()->create<luci::CircleTranspose>();
  trans->perm(perm);
  trans->name(name + "/Transpose_" + str_indices);
  luci::add_origin(trans, luci::get_origin(node));

  return trans;
}

luci::CircleTranspose *create_Nd_transpose(luci::CircleNode *node,
                                           const std::vector<int32_t> indices)
{
  auto name = node->name();
  assert(name.length() > 0);

  auto perm = node->graph()->nodes()->create<luci::CircleConst>();
  perm->dtype(loco::DataType::S32);
  perm->size<loco::DataType::S32>(indices.size());
  perm->rank(1);
  perm->dim(0) = indices.size();
  for (uint32_t i = 0; i < indices.size(); i++)
    perm->at<loco::DataType::S32>(i) = indices[i];
  perm->shape_status(luci::ShapeStatus::VALID);

  auto make_string = [](const std::vector<int32_t> &nums) {
    std::string str;
    for (auto num : nums)
    {
      if (str.length() > 0)
        str += ".";
      str += std::to_string(num);
    }
    return str;
  };

  auto str_indices = make_string(indices);

  perm->name(name + "/Transpose_" + str_indices + "/perm");

  auto trans = node->graph()->nodes()->create<luci::CircleTranspose>();
  trans->perm(perm);
  trans->name(name + "/Transpose_" + str_indices);
  luci::add_origin(trans, luci::get_origin(node));

  return trans;
}

int32_t nchw_axis_to_nhwc(int32_t axis)
{
  uint32_t pos_axis = axis >= 0 ? static_cast<uint32_t>(axis) : static_cast<uint32_t>(axis + 4);
  static const uint32_t to_nhwc[4] = {0, 3, 1, 2};
  if (pos_axis > 3)
    throw std::runtime_error("Concat axis must be in range [-4, 4)");
  return to_nhwc[pos_axis];
}

luci::CircleTranspose *create_post_transpose(luci::CircleNode *node)
{
  return create_4d_transpose(node, {0, 3, 1, 2});
}

luci::CircleTranspose *create_pre_transpose(luci::CircleNode *node)
{
  return create_4d_transpose(node, {0, 2, 3, 1});
}

bool check_4d_reshape(loco::Node *node, const std::vector<int32_t> indices)
{
  assert(indices.size() == 4); // FIX_CALLER_UNLESS

  auto reshape = dynamic_cast<luci::CircleReshape *>(node);
  if (not reshape)
    return false;

  if (reshape->rank() != 4)
    return false;

  auto input = loco::must_cast<luci::CircleNode *>(reshape->tensor());
  if (input->shape_status() != luci::ShapeStatus::VALID)
    return false;

  if (input->rank() != 4)
    return false;

  if (reshape->shape_status() != luci::ShapeStatus::VALID)
    return false;

  if (!(input->dim(0) == reshape->dim(indices[0])) ||
      !(input->dim(1) == reshape->dim(indices[1])) ||
      !(input->dim(2) == reshape->dim(indices[2])) || !(input->dim(3) == reshape->dim(indices[3])))
    return false;

  return true;
}

// Check if Reshape that converts NCHW -> NHWC
bool is_pre_reshape(loco::Node *node) { return check_4d_reshape(node, {0, 3, 1, 2}); }

// Check if Reshape that converts NHWC -> NCHW
bool is_post_reshape(loco::Node *node) { return check_4d_reshape(node, {0, 2, 3, 1}); }

bool is_post_transpose(loco::Node *node) { return check_4d_transpose(node, {0, 3, 1, 2}); }

bool is_pre_transpose(loco::Node *node) { return check_4d_transpose(node, {0, 2, 3, 1}); }

uint32_t cal_offset(const loco::TensorShape &dimension, const uint32_t *indices)
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
           dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

luci::CircleConst *create_NHWC_paddings(luci::CircleConst *paddings)
{
  // paddings shape is (4,2) (it was checked by is_NCHW)
  assert(paddings != nullptr);
  assert(paddings->rank() == 2);
  assert(paddings->dim(0).value() == 4);
  assert(paddings->dim(1).value() == 2);

  // paddings for idx 0~3 are 0 (checked by is_NCHW)
  assert(paddings->at<loco::DataType::S32>(0) == 0);
  assert(paddings->at<loco::DataType::S32>(1) == 0);
  assert(paddings->at<loco::DataType::S32>(2) == 0);
  assert(paddings->at<loco::DataType::S32>(3) == 0);

  auto name = paddings->name();
  assert(name.length() > 0);

  auto nhwc_paddings = paddings->graph()->nodes()->create<luci::CircleConst>();
  nhwc_paddings->dtype(loco::DataType::S32);
  nhwc_paddings->shape({4, 2});
  nhwc_paddings->shape_status(luci::ShapeStatus::VALID);
  nhwc_paddings->size<loco::DataType::S32>(4 * 2);
  nhwc_paddings->name(name + "_NHWC");

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

luci::CircleConst *create_NHWC_rindices(luci::CircleConst *rindices)
{
  assert(rindices != nullptr); // FIX_CALLER_UNLESS

  if (rindices->dtype() != loco::DataType::S32)
    return nullptr;

  auto nhwc_rindices = luci::clone(rindices);
  auto name = rindices->name();
  assert(name.length() > 0); // FIX_CALLER_UNLESS
  nhwc_rindices->name(name + "_NHWC");

  auto size = nhwc_rindices->size<loco::DataType::S32>();
  for (uint32_t i = 0; i < size; i++)
  {
    nhwc_rindices->at<loco::DataType::S32>(i) =
      nchw_axis_to_nhwc(rindices->at<loco::DataType::S32>(i));
  }

  return nhwc_rindices;
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

  auto name = constant->name();
  assert(name.length() > 0);

  auto nhwc_const = constant->graph()->nodes()->create<luci::CircleConst>();
  nhwc_const->dtype(constant->dtype());
  nhwc_const->rank(4);
  nhwc_const->dim(0).set(constant->dim(0).value());
  nhwc_const->dim(1).set(constant->dim(2).value());
  nhwc_const->dim(2).set(constant->dim(3).value());
  nhwc_const->dim(3).set(constant->dim(1).value());
  nhwc_const->shape_status(luci::ShapeStatus::VALID);
  nhwc_const->size<loco::DataType::FLOAT32>(constant->size<loco::DataType::FLOAT32>());
  nhwc_const->name(name + "_NHWC");

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

// NOTE Following conditions can be extended later
//
// Find PAD with an NCHW pattern described below
//   - Paddings shape : [4, 2]
//   - Paddings value : [[0, 0], [0, 0], [h_t, h_b], [w_t, w_b]]]
bool is_NCHW(const luci::CirclePad *node)
{
  const auto paddings = dynamic_cast<luci::CircleConst *>(node->paddings());
  // Non-const paddings is not supported
  if (paddings == nullptr)
    return false;

  if (paddings->rank() != 2)
    return false;

  if (paddings->dim(0).value() != 4 || paddings->dim(1).value() != 2)
    return false;

  // Only check the first two dimensions
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

// NOTE Copied from is_NCHW(CirclePad)
bool is_NCHW(const luci::CirclePadV2 *node)
{
  const auto paddings = dynamic_cast<luci::CircleConst *>(node->paddings());
  // Non-const paddings is not supported
  if (paddings == nullptr)
    return false;

  if (paddings->rank() != 2)
    return false;

  if (paddings->dim(0).value() != 4 || paddings->dim(1).value() != 2)
    return false;

  // Only check the first two dimensions
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

bool is_const(const loco::Node *node)
{
  if (not dynamic_cast<const luci::CircleConst *>(node))
    return false;

  return true;
}

bool is_scalar_const(const loco::Node *node)
{
  auto const_node = dynamic_cast<const luci::CircleConst *>(node);
  if (not const_node)
    return false;

  const auto const_rank = const_node->rank();
  // shape of scalar
  // 1. rank = 0
  // 2. rank = 1, dimension = 1
  if (const_rank == 0)
    return true;

  if (const_rank == 1 && const_node->dim(0).value() == 1)
    return true;

  return false;
}

// NOTE Following conditions can be extended later
//
// Find MUL with an NCHW pattern described below
//   - Input (non-constant) shape : [N, C, H, W]
//   - Input (constant) shape : broadcastable to [N, C, H, W]
//   - Output shape : [N, C, H, W]
bool is_NCHW_with_const(const luci::CircleMul *node, luci::CircleNode *&pred_node,
                        luci::CircleConst *&multiplier)
{
  auto x = dynamic_cast<luci::CircleConst *>(node->x());
  auto y = dynamic_cast<luci::CircleConst *>(node->y());

  if (x != nullptr && y == nullptr)
  {
    pred_node = loco::must_cast<luci::CircleNode *>(node->y());
    multiplier = x;
  }
  else if (x == nullptr && y != nullptr)
  {
    pred_node = loco::must_cast<luci::CircleNode *>(node->x());
    multiplier = y;
  }
  else
  {
    // Ignore if MUL does not have a multiplier input.
    return false;
  }

  if (pred_node->rank() != 4)
    return false;

  if (not broadcastable(multiplier, node))
    return false;

  multiplier = expand_to_rank_4(multiplier);

  return true;
}

// We assume ADD with const input is NCHW if,
// Input shape: (N, C, H, W)
// Output shape: (N, C, H, W)
// 1. Const shape is (1, C, 1, 1), (N, C, H, W) or a scalar (1)
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

  if (pred_node->rank() != 4)
    return false;

  if (not broadcastable(beta, node))
    return false;

  beta = expand_to_rank_4(beta);

  return true;
}

// We assume SUB with const input is NCHW if,
// Input shape: (N, C, H, W)
// Output shape: (N, C, H, W)
// 1. Const shape is (1, C, 1, 1), (N, C, H, W) or a scalar (1)
// 2. Input, Output, Const have the same C.
bool is_NCHW_with_const(const luci::CircleSub *node, const luci::CircleNode *pred_node,
                        const luci::CircleConst *subtract)
{
  assert(pred_node != nullptr);
  assert(subtract != nullptr);

  if (pred_node->rank() != 4)
    return false;

  const auto const_rank = subtract->rank();
  // Support Rank 4 or scalar (rank 0 or 1)
  if (const_rank != 4 && const_rank != 0 && const_rank != 1)
    return false;

  const auto input_cdim = pred_node->dim(1);
  const auto output_cdim = node->dim(1);

  if (const_rank == 4)
  {
    bool supported_shape = false;

    // Check subtract is (1, C, 1, 1)
    if (is_same_shape(subtract, {1, node->dim(1), 1, 1}))
      supported_shape = true;

    // Check subtract is (N, C, H, W)
    if (is_same_shape(subtract, {node->dim(0), node->dim(1), node->dim(2), node->dim(3)}))
      supported_shape = true;

    return supported_shape;
  }
  if (input_cdim == output_cdim)
    return true;
  else
    return false;
}

template <class T> bool convert_unary_features(T *node)
{
  const auto pred_node = loco::must_cast<luci::CircleNode *>(node->features());
  auto pre_trans = create_pre_transpose(node);
  pre_trans->a(pred_node);
  node->features(pre_trans);

  // Do shape inference for this node again.
  node->shape_status(luci::ShapeStatus::UNDEFINED);

  auto post_trans = create_post_transpose(node);
  loco::replace(node).with(post_trans);

  post_trans->a(node);

  return true;
}

template <class T> bool convert_unary_x(T *node)
{
  const auto pred_node = loco::must_cast<luci::CircleNode *>(node->x());
  auto pre_trans = create_pre_transpose(node);
  pre_trans->a(pred_node);
  node->x(pre_trans);

  // Do shape inference for this node again.
  node->shape_status(luci::ShapeStatus::UNDEFINED);

  auto post_trans = create_post_transpose(node);
  loco::replace(node).with(post_trans);

  post_trans->a(node);

  return true;
}

template <class T> bool convert_unary_logits(T *node)
{
  const auto pred_node = loco::must_cast<luci::CircleNode *>(node->logits());
  auto pre_trans = create_pre_transpose(node);
  pre_trans->a(pred_node);
  node->logits(pre_trans);

  // Do shape inference for this node again.
  node->shape_status(luci::ShapeStatus::UNDEFINED);

  auto post_trans = create_post_transpose(node);
  loco::replace(node).with(post_trans);

  post_trans->a(node);

  return true;
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
    const auto n = node->dim(0);
    const auto c = node->dim(1);
    const auto h = node->dim(2);
    const auto w = node->dim(3);

    node->dim(1) = h;
    node->dim(2) = w;
    node->dim(3) = c;

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

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

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

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
      assert(beta->rank() == 4); // FIX is_NCHW_with_const unless
      auto nhwc_const = create_NHWC_from_NCHW(beta);
      if (nhwc_const == nullptr)
        return false;
      node->y(nhwc_const);

      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);
      node->x(pre_trans);
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

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);
    return true;
  }

  bool visit(luci::CircleConcatenation *node)
  {
    const auto num_values = node->numValues();
    for (uint32_t i = 0; i < num_values; i++)
    {
      auto pred_node = loco::must_cast<luci::CircleNode *>(node->values(i));
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);
      node->values(i, pre_trans);
    }

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    node->axis(nchw_axis_to_nhwc(node->axis()));

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  bool visit(luci::CircleElu *node) { return convert_unary_features<luci::CircleElu>(node); }

  bool visit(luci::CircleGelu *node) { return convert_unary_features<luci::CircleGelu>(node); }

  bool visit(luci::CircleLeakyRelu *node)
  {
    return convert_unary_features<luci::CircleLeakyRelu>(node);
  }

  bool visit(luci::CircleLogistic *node) { return convert_unary_x<luci::CircleLogistic>(node); }

  bool visit(luci::CircleMaximum *node)
  {
    if ((not is_const(node->x())) and is_scalar_const(node->y()))
    {
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(node->x());
      node->x(pre_trans);
    }
    else if (is_scalar_const(node->x()) and (not is_const(node->y())))
    {
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(node->y());
      node->y(pre_trans);
    }
    else if ((not is_const(node->x())) and (not is_const(node->y())))
    {
      auto pre_trans_x = create_pre_transpose(node);
      pre_trans_x->a(node->x());
      node->x(pre_trans_x);

      auto pre_trans_y = create_pre_transpose(node);
      pre_trans_y->a(node->y());
      node->y(pre_trans_y);
    }
    else
    {
      // TODO support other cases
      return false;
    }

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);
    return true;
  }

  bool visit(luci::CircleMean *node)
  {
    auto input = loco::must_cast<luci::CircleNode *>(node->input());
    if (input->rank() != 4)
      return false;

    auto rindices = dynamic_cast<luci::CircleConst *>(node->reduction_indices());
    if (not rindices)
      return false;

    auto nhwc_rindices = create_NHWC_rindices(rindices);
    if (not nhwc_rindices)
      return false;

    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(input);
    node->input(pre_trans);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    node->reduction_indices(nhwc_rindices);

    if (node->keep_dims())
    {
      auto post_trans = create_post_transpose(node);
      loco::replace(node).with(post_trans);

      post_trans->a(node);

      return true;
    }

    // node->keep_dims() == false
    // 1D output never needs a transpose
    if (node->rank() <= 1)
      return true;

    std::vector<bool> reduced_dims_nhwc(4, false);
    uint32_t num_reduced_indices = nhwc_rindices->size<loco::DataType::S32>();

    for (uint32_t ri = 0; ri < num_reduced_indices; ++ri)
    {
      reduced_dims_nhwc[nhwc_rindices->at<loco::DataType::S32>(ri)] = true;
    }

    // if channel dimension has been reduced, we don't need a transpose
    if (reduced_dims_nhwc[3])
      return true;

    // likewise, if both space dimensions are reduced, no transpose is needed
    if (reduced_dims_nhwc[1] && reduced_dims_nhwc[2])
      return true;

    std::vector<int32_t> post_trans_ind;
    // case 1: only N is reduced
    if (num_reduced_indices == 1 && reduced_dims_nhwc[0])
      post_trans_ind = {2, 0, 1};

    // case 2: only H or W is reduced
    if (num_reduced_indices == 1 && (reduced_dims_nhwc[1] || reduced_dims_nhwc[2]))
      post_trans_ind = {0, 2, 1};

    // case 3: N and either H or W are reduced
    if (num_reduced_indices == 2)
      post_trans_ind = {1, 0};

    auto post_trans = create_Nd_transpose(node, post_trans_ind);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  bool visit(luci::CircleMinimum *node)
  {
    if ((not is_const(node->x())) and is_scalar_const(node->y()))
    {
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(node->x());
      node->x(pre_trans);
    }
    else if (is_scalar_const(node->x()) and (not is_const(node->y())))
    {
      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(node->y());
      node->y(pre_trans);
    }
    else
    {
      // TODO support other cases
      return false;
    }

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);
    return true;
  }

  bool visit(luci::CircleMul *node)
  {
    LOGGER(l);

    luci::CircleNode *pred_node = nullptr;
    luci::CircleConst *multiplier = nullptr;

    if (is_NCHW_with_const(node, pred_node, multiplier))
    {
      assert(multiplier->rank() == 4); // FIX is_NCHW_with_const unless
      auto nhwc_const = create_NHWC_from_NCHW(multiplier);
      if (nhwc_const == nullptr)
        return false;
      node->y(nhwc_const);

      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);
      node->x(pre_trans);
    }
    else if (multiplier == nullptr)
    {
      // Only support for input rank 4
      auto input_x = loco::must_cast<luci::CircleNode *>(node->x());
      if (input_x->rank() != 4)
        return false;
      auto input_y = loco::must_cast<luci::CircleNode *>(node->y());
      if (input_y->rank() != 4)
        return false;

      auto pre_trans_x = create_pre_transpose(node);
      pre_trans_x->a(input_x);
      node->x(pre_trans_x);

      auto pre_trans_y = create_pre_transpose(node);
      pre_trans_y->a(input_y);
      node->y(pre_trans_y);
    }
    else
    {
      return false;
    }

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);
    return true;
  }

  bool visit(luci::CircleNeg *node) { return convert_unary_x<luci::CircleNeg>(node); }

  bool visit(luci::CirclePad *node)
  {
    if (!is_NCHW(node))
      return false;

    const auto pred_node = loco::must_cast<luci::CircleNode *>(node->input());
    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(pred_node);
    node->input(pre_trans);

    auto nchw_paddings = loco::must_cast<luci::CircleConst *>(node->paddings());
    const auto nhwc_paddings = create_NHWC_paddings(nchw_paddings);
    node->paddings(nhwc_paddings);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  bool visit(luci::CirclePadV2 *node)
  {
    if (!is_NCHW(node))
      return false;

    const auto pred_node = loco::must_cast<luci::CircleNode *>(node->input());
    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(pred_node);
    node->input(pre_trans);

    auto nchw_paddings = loco::must_cast<luci::CircleConst *>(node->paddings());
    const auto nhwc_paddings = create_NHWC_paddings(nchw_paddings);
    node->paddings(nhwc_paddings);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  // TODO Reduce duplicate code with CircleMean
  bool visit(luci::CircleReduceMax *node)
  {
    auto input = loco::must_cast<luci::CircleNode *>(node->input());
    if (input->rank() != 4)
      return false;

    auto rindices = dynamic_cast<luci::CircleConst *>(node->reduction_indices());
    if (not rindices)
      return false;

    auto nhwc_rindices = create_NHWC_rindices(rindices);
    if (not nhwc_rindices)
      return false;

    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(input);
    node->input(pre_trans);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    node->reduction_indices(nhwc_rindices);

    if (node->keep_dims())
    {
      auto post_trans = create_post_transpose(node);
      loco::replace(node).with(post_trans);

      post_trans->a(node);

      return true;
    }

    // The below codes handle the cases where node->keep_dims() == false
    // 1D output never needs a transpose
    if (node->rank() <= 1)
      return true;

    std::vector<bool> reduced_dims_nhwc(4, false);
    uint32_t num_reduced_indices = nhwc_rindices->size<loco::DataType::S32>();

    for (uint32_t ri = 0; ri < num_reduced_indices; ++ri)
    {
      reduced_dims_nhwc[nhwc_rindices->at<loco::DataType::S32>(ri)] = true;
    }

    // if channel dimension has been reduced, we don't need a transpose
    if (reduced_dims_nhwc[3])
      return true;

    // likewise, if both space dimensions are reduced, no transpose is needed
    if (reduced_dims_nhwc[1] && reduced_dims_nhwc[2])
      return true;

    std::vector<int32_t> post_trans_ind;
    // case 1: only N is reduced
    if (num_reduced_indices == 1 && reduced_dims_nhwc[0])
      post_trans_ind = {2, 0, 1};

    // case 2: only H or W is reduced
    if (num_reduced_indices == 1 && (reduced_dims_nhwc[1] || reduced_dims_nhwc[2]))
      post_trans_ind = {0, 2, 1};

    // case 3: N and either H or W are reduced
    if (num_reduced_indices == 2)
      post_trans_ind = {1, 0};

    auto post_trans = create_Nd_transpose(node, post_trans_ind);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  // TODO Reduce duplicate codes with CircleReduceMax
  bool visit(luci::CircleReduceMin *node)
  {
    auto input = loco::must_cast<luci::CircleNode *>(node->input());
    if (input->rank() != 4)
      return false;

    auto rindices = dynamic_cast<luci::CircleConst *>(node->reduction_indices());
    if (not rindices)
      return false;

    auto nhwc_rindices = create_NHWC_rindices(rindices);
    if (not nhwc_rindices)
      return false;

    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(input);
    node->input(pre_trans);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    node->reduction_indices(nhwc_rindices);

    if (node->keep_dims())
    {
      auto post_trans = create_post_transpose(node);
      loco::replace(node).with(post_trans);

      post_trans->a(node);

      return true;
    }

    // The below codes handle the cases where node->keep_dims() == false
    // 1D output never needs a transpose
    if (node->rank() <= 1)
      return true;

    std::vector<bool> reduced_dims_nhwc(4, false);
    uint32_t num_reduced_indices = nhwc_rindices->size<loco::DataType::S32>();

    for (uint32_t ri = 0; ri < num_reduced_indices; ++ri)
    {
      reduced_dims_nhwc[nhwc_rindices->at<loco::DataType::S32>(ri)] = true;
    }

    // if channel dimension has been reduced, we don't need a transpose
    if (reduced_dims_nhwc[3])
      return true;

    // likewise, if both space dimensions are reduced, no transpose is needed
    if (reduced_dims_nhwc[1] && reduced_dims_nhwc[2])
      return true;

    std::vector<int32_t> post_trans_ind;
    // case 1: only N is reduced
    if (num_reduced_indices == 1 && reduced_dims_nhwc[0])
      post_trans_ind = {2, 0, 1};

    // case 2: only H or W is reduced
    if (num_reduced_indices == 1 && (reduced_dims_nhwc[1] || reduced_dims_nhwc[2]))
      post_trans_ind = {0, 2, 1};

    // case 3: N and either H or W are reduced
    if (num_reduced_indices == 2)
      post_trans_ind = {1, 0};

    auto post_trans = create_Nd_transpose(node, post_trans_ind);
    loco::replace(node).with(post_trans);

    post_trans->a(node);

    return true;
  }

  bool visit(luci::CircleRelu *node) { return convert_unary_features<luci::CircleRelu>(node); }

  bool visit(luci::CircleRelu6 *node) { return convert_unary_features<luci::CircleRelu6>(node); }

  bool visit(luci::CircleRsqrt *node) { return convert_unary_x<luci::CircleRsqrt>(node); }

  bool visit(luci::CircleSplitV *node)
  {
    // Change split dimension
    auto axis = dynamic_cast<luci::CircleConst *>(node->split_dim());
    if (not axis)
      return false;

    if (axis->dtype() != loco::DataType::S32)
      return false;

    if (axis->size<loco::DataType::S32>() != 1)
      return false;

    axis->at<loco::DataType::S32>(0) = nchw_axis_to_nhwc(axis->at<loco::DataType::S32>(0));

    // Insert pre-transpose
    const auto pred_node = loco::must_cast<luci::CircleNode *>(node->input());
    auto pre_trans = create_pre_transpose(node);
    pre_trans->a(pred_node);
    node->input(pre_trans);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    // Insert post-transposes
    for (auto succ : loco::succs(node))
    {
      auto svo = loco::must_cast<luci::CircleSplitVOut *>(succ);

      auto post_trans = create_post_transpose(svo);
      loco::replace(svo).with(post_trans);
      post_trans->a(svo);
    }

    return true;
  }

  bool visit(luci::CircleSquaredDifference *node)
  {
    // TODO support CircleConst input
    if (dynamic_cast<luci::CircleConst *>(node->x()) != nullptr)
      return false;
    if (dynamic_cast<luci::CircleConst *>(node->y()) != nullptr)
      return false;

    auto input_x = loco::must_cast<luci::CircleNode *>(node->x());
    if (input_x->rank() != 4)
      return false;
    auto input_y = loco::must_cast<luci::CircleNode *>(node->y());
    if (input_y->rank() != 4)
      return false;

    auto pre_trans_x = create_pre_transpose(node);
    pre_trans_x->a(input_x);
    node->x(pre_trans_x);

    auto pre_trans_y = create_pre_transpose(node);
    pre_trans_y->a(input_y);
    node->y(pre_trans_y);

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

    auto post_trans = create_post_transpose(node);
    loco::replace(node).with(post_trans);

    post_trans->a(node);
    return true;
  }

  bool visit(luci::CircleSub *node)
  {
    luci::CircleNode *pred_node = nullptr;
    luci::CircleConst *subtract = nullptr;

    auto const_x = dynamic_cast<luci::CircleConst *>(node->x());
    auto const_y = dynamic_cast<luci::CircleConst *>(node->y());

    if (const_x != nullptr && const_y == nullptr)
    {
      // case of subtract - pred_node
      pred_node = loco::must_cast<luci::CircleNode *>(node->y());
      subtract = const_x;

      if (!is_NCHW_with_const(node, pred_node, subtract))
        return false;

      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);

      if (subtract->rank() == 4)
      {
        auto nhwc_const = create_NHWC_from_NCHW(subtract);
        if (nhwc_const == nullptr)
          return false;
        node->x(nhwc_const);
      }
      node->y(pre_trans);
    }
    else if (const_x == nullptr && const_y != nullptr)
    {
      // case of pred_node - subtract
      pred_node = loco::must_cast<luci::CircleNode *>(node->x());
      subtract = const_y;

      if (!is_NCHW_with_const(node, pred_node, subtract))
        return false;

      auto pre_trans = create_pre_transpose(node);
      pre_trans->a(pred_node);

      if (subtract->rank() == 4)
      {
        auto nhwc_const = create_NHWC_from_NCHW(subtract);
        if (nhwc_const == nullptr)
          return false;
        node->y(nhwc_const);
      }

      node->x(pre_trans);
    }
    else if (const_x == nullptr && const_y == nullptr)
    {
      // Both inputs are not constant.
      // In this case, we cannot distinguish NCHW from NHWC,
      // so just insert Transpose Ops.
      // Only support for input rank 4.
      auto input_x = loco::must_cast<luci::CircleNode *>(node->x());
      if (input_x->rank() != 4)
        return false;
      auto input_y = loco::must_cast<luci::CircleNode *>(node->y());
      if (input_y->rank() != 4)
        return false;

      auto pre_trans_x = create_pre_transpose(node);
      pre_trans_x->a(input_x);
      node->x(pre_trans_x);

      auto pre_trans_y = create_pre_transpose(node);
      pre_trans_y->a(input_y);
      node->y(pre_trans_y);
    }

    // Do shape inference for this node again.
    node->shape_status(luci::ShapeStatus::UNDEFINED);

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

  // Annotate NHWC operators
  // NHWC operators are detected by pattern matching
  //
  // Pattern
  //    pre-Transose (or pre-Reshape) + [intermediate Ops] + post-Transpose (or post-Reshape)
  //
  // [intermediate Ops] are annotated as NHWC
  //
  // NOTE A single pre-Transpose/Reshape can have multiple post-Transpose/Reshape.
  // For example,
  // pre-Transpose --- [intermediate Ops] --- post-Transpose
  //                |
  //                +--[intermediate Ops] --- post-Transpose
  //
  // NOTE Intermediate Ops SHOULD NOT contain pre-Transpose/Reshape
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    if (has_data_format(node))
      continue;

    if (is_pre_transpose(node) || is_pre_reshape(node))
    {
      std::set<loco::Node *> intermediate;

      // Variable to check intermediate Ops contain pre-Transpose/Reshape
      bool has_pre = false;

      // Variable to check the pattern is closed with post-Transpose/Reshape
      bool is_closed = true;

      // For recursive call of lambda
      std::function<void(loco::Node *)> collect_intermediate;
      collect_intermediate = [&](loco::Node *n) {
        for (auto succ : loco::succs(n))
        {
          // Skip unnecessary traversal
          if (intermediate.find(succ) != intermediate.end())
            continue;

          // Exit condition
          if (is_post_transpose(succ) || is_post_reshape(succ))
            continue;

          if (is_pre_transpose(succ) || is_pre_reshape(succ))
          {
            has_pre = true;
            break;
          }

          if (is_output(succ))
          {
            is_closed = false;
            break;
          }

          intermediate.emplace(succ);

          collect_intermediate(succ);
        }
      };

      collect_intermediate(node);

      if (has_pre or not is_closed)
        continue;

      for (auto inter : intermediate)
      {
        if (not has_data_format(inter))
          set_data_format(inter, DataFormat::NHWC);
      }
    }
  }

  // Annotate NCHW operators
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    switch (circle_node->opcode())
    {
      // List of supported Ops
      case luci::CircleOpcode::CIRCLEINPUT:
        if (!_preserve_input && !has_data_format(node))
        {
          set_data_format(node, DataFormat::NCHW);
        }
        break;
      case luci::CircleOpcode::CIRCLEOUTPUT:
        if (!_preserve_output && !has_data_format(node))
        {
          set_data_format(node, DataFormat::NCHW);
        }
        break;
      // SOFTMAX, LOG_SOFTMAX are not converted, because
      // tflite/circle assumes the last channel is always axis
      case luci::CircleOpcode::ADD:
      case luci::CircleOpcode::CONCATENATION:
      case luci::CircleOpcode::ELU:
      case luci::CircleOpcode::GELU:
      case luci::CircleOpcode::LEAKY_RELU:
      case luci::CircleOpcode::LOGISTIC:
      case luci::CircleOpcode::MAXIMUM:
      case luci::CircleOpcode::MEAN:
      case luci::CircleOpcode::MINIMUM:
      case luci::CircleOpcode::MUL:
      case luci::CircleOpcode::NEG:
      case luci::CircleOpcode::PAD:
      case luci::CircleOpcode::PADV2:
      case luci::CircleOpcode::REDUCE_MAX:
      case luci::CircleOpcode::REDUCE_MIN:
      case luci::CircleOpcode::RELU:
      case luci::CircleOpcode::RELU6:
      case luci::CircleOpcode::RSQRT:
      case luci::CircleOpcode::SPLIT_V:
      case luci::CircleOpcode::SQUARED_DIFFERENCE:
      case luci::CircleOpcode::SUB:
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
    else if (has_dynamic_shape(node))
    {
      // This pass only works for static-shaped node
      INFO(l) << "Skip the node with a dynamic shape." << std::endl;
      continue;
    }
    else
    {
      ConvertNCHWToNHWC converter;
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      if (circle_node->rank() != 4)
      {
        // TODO replace the check above with the input rank check, and remove the condition below
        if (not dynamic_cast<luci::CircleMean *>(node) and
            not dynamic_cast<luci::CircleReduceMax *>(node) and
            not dynamic_cast<luci::CircleReduceMin *>(node))
          continue;
      }

      if (circle_node->accept(&converter))
      {
        set_data_format(node, DataFormat::NHWC);
        changed = true;
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
