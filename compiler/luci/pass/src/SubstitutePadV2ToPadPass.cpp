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

#include "luci/Pass/SubstitutePadV2ToPadPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <vector>

/**
 * @brief Convert PadV2 op in a certain condition to Pad
 * @details Condition to convert PadV2 to Pad is like below:
 *
 * Basic Condition)
 *
 *    C1) For all i, PadV2.input[i] >= 0
 *    C2) For all c, PadV2.constant_values[c] <= 0
 *    C3) PadV2 == MaxPool2D.value()
 *    C4) number of padded values at left < MaxPool2D.Filter.W
 *        number of padded values at right < MaxPool2D.Filter.W
 *        number of padded values at top < MaxPool2D.Filter.H
 *        number of padded values at bottom < MaxPool2D.Filter.H
 *
 *    Example graph is as follows:
 *
 *        %1 = CircleRelu                  # relu_output[i] >= 0
 *        %2 = CirclePadV2(%1, constant_values <= 0)
 *        %3 = CircleMaxPool2D(%2, ...)    # output will be chosen from relu_output
 *
 * In this case, it's OK to replace PadV2 with Pad, which uses 0 as padding constant.
 *
 * Optional Condition)
 *
 *    Terminology)
 *        - 'reshaping op' : op that does not change the value of tensor
 *           but changes position of tensor value, e.g., Transpose, Reshape, Slice, etc.
 *
 *    C5) Input of PadV2 could be 'reshaping op'. Example is as follow:
 *
 *        %1 = CircleRelu           # output[i] >= 0
 *        %2 = CircleTranspose(%1)  # reshaping op
 *        ...                       # more reshaping ops
 *        %n = CirclePadV2(%n-1, constant_values <= 0)
 *        %n+1 = CircleMaxPool2D(%n, ...)
 *
 *    C6) PadV2 could be an input of 'reshaping op'. Example is as follow:
 *
 *        %1 = CircleRelu
 *        %2 = CirclePadV2(%1, constant_values <= 0)
 *        %3 = CircleTranspose(%2)  # reshaping op
 *        ...                       # more reshaping ops
 *        %n = CircleMaxPool2D(%n-1, ...)
 *
 * Why is this pass required?
 *
 *        When PyTorch model is converted into Circle model, sometimes PadV2 is inserted with
 *        the following pattern:
 *
 *        %1 = Circle.Conv2D(..., activation = Relu)
 *        %2 = Circle.Transpose(%1, perm=[0,3,1,2])
 *        %3 = Circle.PadV2(%2, constant_values = -3.4028234663852886e+38)
 *        %4 = Circle.Transpose(%3, perm=[0,2,3,1])
 *        %5 = Circle.MaxPool2D(%4, filter=[3,3], padding="VALID")
 *
 *        Large negative padding constant of %3 caused problem when we quantized this model.
 *        So we need to convert the negative number to some number in reasonable range for
 *        quantization, e.g., zero.
 */
namespace
{

struct Paddings
{
  struct Pad
  {
    int32_t front;
    int32_t end;
  };
  /**
   * @brief Store paddings position information.
   * @details _padding_pos[k] stores Pad object at axis k
   *
   * @note  Paddings must be for rank 4 tensor
   */
  std::vector<Pad> _padding_pos;

  Paddings(luci::CircleConst *paddings)
  {
    assert(paddings->dtype() == loco::DataType::S32);
    assert(paddings->rank() == 2);
    assert(paddings->dim(1).value() == 2);
    assert(paddings->size<loco::DataType::S32>() == paddings->rank() * 4);

    for (uint32_t i = 0; i < paddings->dim(0).value(); i++)
    {
      Pad pad{.front = paddings->at<loco::DataType::S32>(i * 2),
              .end = paddings->at<loco::DataType::S32>(i * 2 + 1)};
      _padding_pos.emplace_back(pad);
    }

    assert(_padding_pos.size() == 4);
  }

  /**
   * @brief Check if this padding area is covered by filter
   *
   * @note This is to check condition C4).
   *       _padding_pos should store values according to NHWC.
   */
  bool smaller_than(int32_t filter_h, int32_t filter_w)
  {
    auto &pad_H = _padding_pos.at(1);
    auto &pad_W = _padding_pos.at(2);

    return (pad_H.front < filter_h) && (pad_H.end < filter_h) && (pad_W.front < filter_w) &&
           (pad_W.end < filter_w);
  }

  /**
   * @brief Track how paddings change after CircleTranspose
   * @details Consider the following graph,
   *
   *   %1 = Circle.Input
   *   %2 = Circle.PadV2(%1,
   *                     paddings=[[0, 0], [0, 0], [2, 3], [4, 5]],
   *                     padding_value = -100)
   *   %3 = Circle.Transpose(%2, perm[0, 2, 3, 1])
   *
   *   Output of %3 has padding constant value(-100) from %2 at position below:
   *
   *    - axis | front | end
   *     ------|-------|-----
   *       0   |   0   |   0
   *       1   |   2   |   3
   *       2   |   4   |   5
   *       3   |   0   |   0
   *
   *   This method keeps track of such change of padding position.
   */
  void apply(luci::CircleTranspose *transpose)
  {
    assert(transpose);
    luci::CircleConst *perm = loco::must_cast<luci::CircleConst *>(transpose->perm());

    std::vector<Pad> transposed_pos;
    transposed_pos.resize(4);

    for (uint32_t to = 0; to < 4; to++)
    {
      int32_t from = perm->at<loco::DataType::S32>(to);
      transposed_pos.at(to) = _padding_pos.at(from);
    }

    _padding_pos = transposed_pos;
  }
};

struct ReshapingNode
{
  /// @brief Check if node is 'reshaping op'
  static bool check(loco::Node *node)
  {
    if (dynamic_cast<luci::CircleTranspose *>(node))
      return true;
    // add more 'reshaping op'

    return false;
  }

  /// @brief Retuen reshaping op's input
  static loco::Node *input(loco::Node *node)
  {
    if (auto transpose = dynamic_cast<luci::CircleTranspose *>(node))
      return transpose->a();
    // add more 'reshaping op'

    throw std::runtime_error("Not yet supported reshaping op");
  }
};

/// @brief Get only successor node
loco::Node *get_only_succ(loco::Node *parent)
{
  assert(parent);

  auto successors = loco::succs(parent);
  if (successors.size() != 1)
    return nullptr;

  return *successors.begin();
}

// Check condition C1) and C5)
bool positive_or_zero(loco::Node *ifm)
{
  assert(ifm);

  if (ReshapingNode::check(ifm))
    return positive_or_zero(ReshapingNode::input(ifm));

  // Since Relu.output[i] >= 0
  if (dynamic_cast<luci::CircleRelu *>(ifm))
    return true;
  if (auto node = dynamic_cast<luci::CircleNodeMixin<luci::CircleNodeTrait::FusedActFunc> *>(ifm))
  {
    if (node->fusedActivationFunction() == luci::FusedActFunc::RELU)
      return true;
    // Add more FusedActFunc
  }
  // Add more ops of which output[i] >= 0

  return false;
}

template <loco::DataType DT> bool has_all_positive_values(luci::CircleConst *node)
{
  // Only numeric datatype is allowed
  static_assert(DT != loco::DataType::Unknown);
  static_assert(DT != loco::DataType::STRING);

  assert(node);

  auto size = node->size<DT>();
  for (decltype(size) t = 0; t < size; t++)
  {
    typename loco::DataTypeImpl<DT>::Type val = node->at<DT>(t);
    if (val <= 0)
      return false;
  }

  return true;
}

// To check condition C2)
bool has_all_positive_values(luci::CircleConst *node)
{
  assert(node);

  if (node->dtype() == loco::DataType::FLOAT32)
    return has_all_positive_values<loco::DataType::FLOAT32>(node);
  // Add more datatype

  throw std::runtime_error("Not yet supported datatype");
}

bool used_by_maxpool_only(luci::CircleNode *node, Paddings &paddings)
{
  auto successor = get_only_succ(node);

  // when successor is not only-succ
  if (successor == nullptr)
    return false;

  if (auto maxpool = dynamic_cast<luci::CircleMaxPool2D *>(successor))
  {
    // Let's check condition C4)
    return paddings.smaller_than(maxpool->filter()->h(), maxpool->filter()->w());
  }

  // Let's check condition C6)
  if (auto transpose = dynamic_cast<luci::CircleTranspose *>(successor))
  {
    auto appropriate = [](luci::CircleTranspose *transpose) {
      luci::CircleConst *perm = loco::must_cast<luci::CircleConst *>(transpose->perm());

      // For Transpose to be an input for MaxPool2D
      return (transpose->rank() == 4) && (perm && perm->dtype() == loco::DataType::S32) &&
             (perm->size<loco::DataType::S32>() == 4);
    };

    if (not appropriate(transpose))
      return false;

    paddings.apply(transpose);
    return used_by_maxpool_only(transpose, paddings);
  }
  // Support more 'reshaping op' later

  return false;
}

// Check condition C3), C4) and C6)
bool used_by_maxpool_only(luci::CirclePadV2 *pad_v2)
{
  // For PadV2 to be an input for MaxPool2D
  if (pad_v2->rank() != 4)
    return false;

  Paddings paddings(loco::must_cast<luci::CircleConst *>(pad_v2->paddings()));

  return used_by_maxpool_only(pad_v2, paddings);
}

loco::Node *build_pad_from(luci::CirclePadV2 *pad_v2)
{
  auto copy_shape = [](const luci::CircleNode *src, luci::CircleNode *dest) {
    auto rank = src->rank();
    dest->rank(rank);

    for (decltype(rank) axis = 0; axis < rank; axis++)
      dest->dim(axis) = src->dim(axis);
  };

  auto g = pad_v2->graph();

  auto pad = g->nodes()->create<luci::CirclePad>();
  {
    pad->name(pad_v2->name() + "/pad");
    luci::add_origin(pad, luci::get_origin(pad_v2));

    pad->dtype(pad_v2->dtype());
    copy_shape(pad_v2, pad);

    pad->input(pad_v2->input());
    pad->paddings(pad_v2->paddings());
  }

  return pad;
}

luci::CirclePadV2 *get_padv2(loco::Node *node)
{
  if (auto padv2 = dynamic_cast<luci::CirclePadV2 *>(node))
    return padv2;

  if (ReshapingNode::check(node))
    return get_padv2(ReshapingNode::input(node));

  return nullptr;
}

bool substitute_padv2_to_pad(luci::CircleMaxPool2D *maxp)
{
  // precondition
  assert(maxp);
  assert(maxp->value());

  auto pad_v2 = get_padv2(maxp->value());

  if (pad_v2 == nullptr)
    return false;

  assert(pad_v2->input());

  auto paddings = loco::must_cast<luci::CircleConst *>(pad_v2->paddings());
  auto constant_values = loco::must_cast<luci::CircleConst *>(pad_v2->constant_values());

  (void)paddings;
  assert(paddings);
  assert(paddings->dtype() == loco::DataType::S32);
  assert(constant_values);
  assert(constant_values->dtype() == pad_v2->dtype());

  if (not positive_or_zero(pad_v2->input()))
    return false;

  if (has_all_positive_values(constant_values))
    return false;

  if (not used_by_maxpool_only(pad_v2))
    return false;

  auto pad = build_pad_from(pad_v2);

  replace(pad_v2).with(pad);

  return true;
}

} // namespace

namespace luci
{

/**
 * Case 1) Basic case
 *
 * BEFORE
 *     [CircleRelu]
 *          |
 *          |       [CircleConst] [CircleConst]
 *          |             |              |
 *          -------+----------------------
 *                 |
 *            [CirclePadV2]
 *                 |
 *          [CircleMaxPool2D]
 *                 |
 *
 * AFTER
 *     [CircleRelu]
 *          |
 *          |           [CircleConst]    [CircleNode] [CircleConst]
 *          |             |      |            |              |
 *          -------+-------      -------------+--------------+
 *                 |                          |
 *            [CirclePad]                [CirclePadV2]
 *                 |
 *         [CircleMaxPool2D]
 *                 |
 *
 * Case 2) During conversion from a PyTorch model into a Circle model,
 * it is common that some 'Reshaping op', e.g., CircleTranspose,
 * are inserted in-between operations to swith NCHW into NHWC and vice versa.
 * This pass also needs to handle such situation.
 *
 * BEFORE
 *     [CircleRelu]
 *          |
 *          |       [CircleConst] [CircleConst]
 *          |             |              |
 *          -------+----------------------
 *                 |
 *          [CircleTranspose]
 *                 |
 *            [CirclePadV2]
 *                 |
 *          [CircleTranspose]
 *                 |
 *          [CircleMaxPool2D]
 *                 |
 *
 * AFTER
 *     [CircleRelu]
 *          |
 *          |           [CircleConst]    [CircleNode] [CircleConst]
 *          |             |      |            |              |
 *          -------+-------      -------------+--------------+
 *                 |                          |
 *          [CircleTranspose]           [CirclePadV2]
 *                 |
 *            [CirclePad]
 *                 |
 *          [CircleTranspose]
 *                 |
 *          [CircleMaxPool2D]
 *                 |
 */
bool SubstitutePadV2ToPadPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_node = dynamic_cast<luci::CircleMaxPool2D *>(node))
    {
      if (substitute_padv2_to_pad(circle_node))
      {
        changed = true;
      }
    }
  }
  return changed;
}

} // namespace luci
