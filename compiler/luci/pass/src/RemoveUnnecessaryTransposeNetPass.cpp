/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveUnnecessaryTransposeNetPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <vector>

namespace
{

class TaggedShapeAnalyzer final
{
public:
  /**
   * @brief check 'Transpose-Reshape-Transpose' can be replaced by one 'Reshape'.
   *
   * @example
   *  Let's explain how analyzer check Transpose-Reshape-Transpose pattern with an exact example.
   *
   *  Let's assume under pattern is given :
   *
   *      Input(1, 7, 7, 448)
   *            |
   *      Transpose(perm=(0, 3, 1, 2))
   *            |
   *      Resahape(shape=(1, 448, 49))
   *            |
   *      Transpose(perm=(0, 2, 1))
   *            |
   *      Output(1, 49, 448)
   *
   *  It simulates how each dimension of the tensor's shape are transformed/moved
   *  using a member variable named '_shape'.
   *  'tags' in _shape record the initial order of each dimension.
   *
   *   TIMELINE              |   _shape states :
   *                         |
   *   init_shape_with_tag   |    - value :   (1)   (7)   (7)   (448)
   *                         |    - tags  :   (-)   (0)   (1)   (2)
   *                         |
   *   analyze_transpose     |    - value :   (1)   (448)  (7)  (7)
   *                         |    - tags  :   (-)   (2)    (0)  (1)
   *                         |
   *   analyze_reshape       |    - value :   (1)   (448)   (49)
   *                         |    - tags  :   (-)   (2)     (0, 1)
   *                         |
   *   anaylze_transpose     |    - value  :   (1)   (49)    (448)
   *                         |    - tags   :   (-)   (0, 1)  (2)
   *
   *  After all simulation done, if tags are in same order as initial _shape,
   *  Transpose has no effect in final shape, which they can be removed as
   *  unnecessary Ops.
   */
  template <loco::DataType DType>
  bool can_remove_transposes(const luci::CircleTranspose *f_tr, const luci::CircleReshape *m_rs,
                             const luci::CircleTranspose *b_tr);

private:
  void init_shape_with_tag(const luci::CircleNode *in_tensor);
  template <loco::DataType PermType>
  bool analyze_transpose(const luci::CircleTranspose *transpose_node);
  template <loco::DataType ShapeType> bool analyze_reshape(const luci::CircleReshape *reshape_node);
  bool verify_tag() const;

  struct Dim final
  {
    uint32_t value;
    std::vector<uint8_t> tags;
  };
  using Shape = std::vector<Dim>;

  const uint8_t START_TAG = 0;
  Shape _shape;
};

/**
 * @brief initalize _shape with input tensor named in_tensor
 *
 * @note 'tags' are attached to non-1 valued dimension.
 */
void TaggedShapeAnalyzer::init_shape_with_tag(const luci::CircleNode *in_tensor)
{
  _shape.clear();
  uint8_t tag = START_TAG;

  for (uint32_t i = 0; i < in_tensor->rank(); i++)
  {
    TaggedShapeAnalyzer::Dim dim;
    {
      dim.value = in_tensor->dim(i).value();
      if (dim.value != 1)
        dim.tags.push_back(tag++);
    }
    _shape.push_back(dim);
  }
}

/**
 * @brief update _shape based on 'Transpose' permutation value
 *
 * @example Let's assume Transpose(perm=0, 3, 1, 2) is given to [before] _shape.
 *
 *  This function reordered the Dims' order based on permutaiton value.
 *
 *  [before]  _shape :
 *              - value :   (1)   (7)   (7)   (448)
 *              - tags  :   (-)   (0)   (1)   (2)
 *
 *  [after]  _shape :
 *              - value :   (1)   (448)  (7)  (7)
 *              - tags  :   (-)   (2)    (0)  (1)
 */
template <loco::DataType PermType>
bool TaggedShapeAnalyzer::analyze_transpose(const luci::CircleTranspose *transpose_node)
{
  const luci::CircleConst *perm_node = loco::must_cast<luci::CircleConst *>(transpose_node->perm());
  assert(perm_node->dtype() == PermType);

  TaggedShapeAnalyzer::Shape new_shape;
  const auto size = perm_node->size<PermType>();
  for (uint32_t i = 0; i < size; i++)
  {
    auto perm_idx = perm_node->at<PermType>(i);
    new_shape.push_back(_shape.at(perm_idx));
  }
  _shape = new_shape;
  return true;
}

/**
 * @brief update _shape based on 'Reshape' shape value
 *
 * @example Let's assume Reshape(shape=1, 448, 49) is given to [before] _shape.
 *
 * This function merge N Dims into one.
 * Whem merging, Dim.value is producted and Dim.tags are aggregated.
 *
 *  [before]  _shape :
 *              - value :   (1)   (448)  (7)   (7)
 *              - tags  :   (-)   (2)    (0)   (1)
 *
 *  [after]  _shape :
 *              - value :   (1)   (448)  (49)
 *              - tags  :   (-)   (2)    (0, 1)
 */
template <loco::DataType ReshapeType>
bool TaggedShapeAnalyzer::analyze_reshape(const luci::CircleReshape *reshape_node)
{
  const luci::CircleConst *shape_node = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  assert(shape_node->dtype() == ReshapeType);

  // At least one element must be in reshape's output-tensor.
  if (shape_node->size<ReshapeType>() <= 0)
    return false;

  // Create new_shape based on reshape_node/shape
  Shape new_shape;
  for (uint32_t i = 0; i < shape_node->size<ReshapeType>(); i++)
  {
    TaggedShapeAnalyzer::Dim dim;
    dim.value = shape_node->at<ReshapeType>(i);
    new_shape.push_back(dim);
  }

  // indexing for _shape [old_shape_start_idx, old_shape_end_idx)
  uint32_t old_shape_start_idx = 0;
  uint32_t old_shape_end_idx = 1;
  uint32_t old_shape_product = _shape[old_shape_start_idx].value;

  auto expand_range = [&]() -> bool {
    if (old_shape_end_idx >= _shape.size())
      return false;

    old_shape_product *= _shape[old_shape_end_idx].value;
    old_shape_end_idx++;
    return true;
  };

  auto move_to_next_range = [&]() -> bool {
    if (old_shape_end_idx > _shape.size())
      return false;

    old_shape_start_idx = old_shape_end_idx;
    old_shape_end_idx++;
    old_shape_product = _shape[old_shape_start_idx].value;
    return true;
  };

  // Add tags from '_shape' to the 'new_shape'
  uint32_t new_shape_idx = 0;
  while (new_shape_idx < new_shape.size())
  {
    Dim &target_dim = new_shape[new_shape_idx];

    // Ignore dim == 1
    if (target_dim.value == 1)
    {
      new_shape_idx++;
      continue;
    }

    while (old_shape_product < target_dim.value)
    {
      if (expand_range() == false)
        break;
    }

    if (old_shape_product != target_dim.value)
      return false;

    // old_shape_product == target_dim.value
    for (uint32_t idx = old_shape_start_idx; idx < old_shape_end_idx; idx++)
    {
      const auto &old_tags = _shape[idx].tags;
      target_dim.tags.insert(target_dim.tags.end(), old_tags.begin(), old_tags.end());
    }

    new_shape_idx++;
    move_to_next_range();
  }
  _shape = new_shape;
  return true;
}

bool TaggedShapeAnalyzer::verify_tag() const
{
  // check whether tags in _shape are incremental
  uint8_t tag = START_TAG;
  for (const auto &dim : _shape)
  {
    for (const auto &t : dim.tags)
    {
      if (t == tag)
        tag++;
      else
        return false;
    }
  }
  return true;
}

template <loco::DataType DType>
bool TaggedShapeAnalyzer::can_remove_transposes(const luci::CircleTranspose *f_tr,
                                                const luci::CircleReshape *m_rs,
                                                const luci::CircleTranspose *b_tr)
{
  assert(loco::must_cast<luci::CircleConst *>(f_tr->perm())->dtype() == DType);
  assert(loco::must_cast<luci::CircleConst *>(m_rs->shape())->dtype() == DType);
  assert(loco::must_cast<luci::CircleConst *>(b_tr->perm())->dtype() == DType);

  const luci::CircleNode *in_tensor = loco::must_cast<luci::CircleNode *>(f_tr->a());

  init_shape_with_tag(in_tensor);

  if (not analyze_transpose<DType>(f_tr))
    return false;

  if (not analyze_reshape<DType>(m_rs))
    return false;

  if (not analyze_transpose<DType>(b_tr))
    return false;

  if (not verify_tag())
    return false;

  return true;
}

/**
 * @brief create CircleReshape node that reshapes 'front_transpose input tensor shape' into
 * 'back_transposes output tensor shape'
 */
template <loco::DataType ShapeType>
luci::CircleReshape *create_reshape_node(loco::Graph *graph,
                                         const luci::CircleTranspose *front_transpose,
                                         const luci::CircleReshape *mid_reshape,
                                         const luci::CircleTranspose *back_transpose)
{
  std::string composed_name =
    front_transpose->name() + ";" + mid_reshape->name() + ";" + back_transpose->name();

  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> src_origin{luci::get_origin(front_transpose),
                                                                  luci::get_origin(mid_reshape),
                                                                  luci::get_origin(back_transpose)};
  auto const composed_origin = luci::composite_origin(src_origin);

  auto shape_node = graph->nodes()->create<luci::CircleConst>();
  {
    shape_node->dtype(ShapeType);
    shape_node->rank(1);
    shape_node->dim(0).set(back_transpose->rank());

    shape_node->size<ShapeType>(back_transpose->rank());
    for (uint32_t i = 0; i < back_transpose->rank(); i++)
    {
      shape_node->at<ShapeType>(i) = back_transpose->dim(i).value();
    }
    shape_node->shape_status(luci::ShapeStatus::VALID);
    shape_node->name(composed_name + "/shape");
    luci::add_origin(shape_node, composed_origin);
  }

  auto reshape_node = graph->nodes()->create<luci::CircleReshape>();
  {
    reshape_node->name(composed_name);
    reshape_node->tensor(front_transpose->a());
    reshape_node->shape(shape_node);
    luci::add_origin(reshape_node, composed_origin);
  }
  return reshape_node;
}

bool remove_unnecessary_transpose(luci::CircleTranspose *node)
{
  // find 'front_transpose - mid_reshape - back_transpose' pattern
  const auto back_transpose = node;
  const auto mid_reshape = dynamic_cast<luci::CircleReshape *>(back_transpose->a());
  {
    if (mid_reshape == nullptr)
      return false;
  }
  const auto front_transpose = dynamic_cast<luci::CircleTranspose *>(mid_reshape->tensor());
  {
    if (not front_transpose)
      return false;

    const auto perm = dynamic_cast<luci::CircleConst *>(front_transpose->perm());
    if (perm->dtype() != loco::DataType::S32)
      return false;
  }

  // check perm, shape node and its' datatype
  const auto back_perm = dynamic_cast<luci::CircleConst *>(back_transpose->perm());
  {
    if (back_perm == nullptr)
      return false;

    if (back_perm->dtype() != loco::DataType::S32)
      return false;
  }
  const auto shape = dynamic_cast<luci::CircleConst *>(mid_reshape->shape());
  {
    if (shape == nullptr)
      return false;

    if (shape->dtype() != loco::DataType::S32)
      return false;
  }
  const auto front_perm = dynamic_cast<luci::CircleConst *>(front_transpose->perm());
  {
    if (front_perm == nullptr)
      return false;

    if (front_perm->dtype() != loco::DataType::S32)
      return false;
  }

  // for now, handle only rank reduction equal (not expansion) cases
  const auto output_rank = back_transpose->rank();
  const auto input_rank = front_transpose->rank();
  if (input_rank < output_rank)
    return false;

  // analyze pattern to check this pass is applicable
  TaggedShapeAnalyzer analyzer;
  if (not analyzer.can_remove_transposes<loco::DataType::S32>(front_transpose, mid_reshape,
                                                              back_transpose))
    return false;

  // repalce with new_node
  luci::CircleReshape *new_node = create_reshape_node<loco::DataType::S32>(
    node->graph(), front_transpose, mid_reshape, back_transpose);

  replace(node).with(new_node);

  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 * Current pass only targets below cases:
 *  - in.rank() >= out.rank()
 *  - 'Reshape' used to reduce N dimension into one (e.g. A x B x C => A x BC)
 *
 *
 *     [CircleNode]      [CircleConst]
 *    (in)               (perm)
 *         \              /
 *       [CircleTranspose]    [CircleConst]
 *            \               (shape)
 *             \             /
 *             [CircleReshape]      [CircleConst]
 *               \                  (perm)
 *                \                /
 *                 [CircleTranspose]
 *                  \
 *                   \
 *                    [CircleNode]
 *                    (out)
 *
 * AFTER
 *
 *    [CircleNode]        [CircleConst]
 *     (in)                (shape)
 *        \               /
 *        [CircleReshape]
 *         (new)
 *          \
 *          [CircleNode]
 *          (out)
 *
 */

bool RemoveUnnecessaryTransposeNetPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto transpose_node = dynamic_cast<luci::CircleTranspose *>(node))
    {
      if (remove_unnecessary_transpose(transpose_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
