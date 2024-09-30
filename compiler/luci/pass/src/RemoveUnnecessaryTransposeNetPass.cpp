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

struct TagDim final
{
  int32_t value;
  std::vector<uint8_t> tags;
};

using TagShape = std::vector<TagDim>;

int32_t flatsize(const TagShape &shape)
{
  int32_t size = 1;
  for (const auto &dim : shape)
  {
    if (dim.value >= 1)
      size *= dim.value;
  }
  return size;
}

/**
 * @brief if 'dst' has -1 valued dim, replace -1 with inferenced value
 *
 * @return  ture, if successfully replace -1 value
 *          false, otherwise
 */
bool inference_incomplete_shape(const TagShape &src, TagShape &dst)
{
  std::vector<size_t> incomplete_indexes;
  for (size_t i = 0; i < dst.size(); i++)
  {
    if (dst[i].value == -1)
      incomplete_indexes.push_back(i);
  }

  if (incomplete_indexes.size() == 0)
    return true;
  else if (incomplete_indexes.size() == 1)
  {
    if (flatsize(src) % flatsize(dst) == 0)
      dst[incomplete_indexes[0]].value = flatsize(src) / flatsize(dst);
    else
      return false;
  }
  else // incomplete_indexes.size() >= 2
    return false;

  return true;
}

template <loco::DataType DTYPE> std::vector<int> extract_const(const luci::CircleConst *const_node)
{
  std::vector<int> values;
  auto size = const_node->size<DTYPE>();
  for (auto i = 0u; i < size; ++i)
  {
    auto v = const_node->at<DTYPE>(i);
    values.push_back(v);
  }
  return values;
};

std::vector<int> extract_shape(const luci::CircleNode *node)
{
  std::vector<int> shape;
  auto rank = node->rank();
  for (auto i = 0u; i < rank; ++i)
  {
    shape.push_back(node->dim(i).value());
  }
  return shape;
};

class TaggedShapeAnalyzer final
{
public:
  template <loco::DataType DType>
  bool can_remove_transposes(const luci::CircleTranspose *f_tr, const luci::CircleReshape *m_rs,
                             const luci::CircleTranspose *b_tr);

private:
  void init_shape_with_tag(const std::vector<int> &in_shape);

  void analyze_transpose(const std::vector<int> &perm);

  template <loco::DataType ShapeType> bool analyze_reshape(const luci::CircleReshape *);

  bool verify_tag() const;

  const uint8_t START_TAG = 0;
  TagShape _shape;
};

/**
 * @brief initalize _shape based on 'in_shape'
 *
 * @note 'tags' are attached to non-1 valued dimension.
 */
void TaggedShapeAnalyzer::init_shape_with_tag(const std::vector<int> &in_shape)
{
  _shape.clear();
  uint8_t tag = START_TAG;

  for (auto i : in_shape)
  {
    TagDim dim;
    dim.value = i;

    if (dim.value != 1)
      dim.tags.push_back(tag++);

    _shape.push_back(dim);
  }
}

/**
 * @brief update _shape based on 'perm'
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
void TaggedShapeAnalyzer::analyze_transpose(const std::vector<int> &perm)
{
  TagShape new_shape;
  for (auto i : perm)
  {
    new_shape.push_back(_shape.at(i));
  }
  _shape = new_shape;
}

/**
 * @brief update _shape based on 'Reshape' shape value
 *
 * @return False, if it determined that removing transposes is impossible
 *
 * @example Let's assume Reshape(shape=1, 448, 49) is given to [before] _shape.
 *
 *  [before]  _shape :
 *              - value :   (1)   (448)  (7)   (7)
 *              - tags  :   (-)   (2)    (0)   (1)
 *
 *  [after]  _shape :
 *              - value :   (1)   (448)  (49)
 *              - tags  :   (-)   (2)    (0, 1)
 */
// TODO: Update analyze_reshape get std::vector<int>&
template <loco::DataType ReshapeType>
bool TaggedShapeAnalyzer::analyze_reshape(const luci::CircleReshape *reshape_node)
{
  const luci::CircleConst *shape_node = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  assert(shape_node->dtype() == ReshapeType);

  // At least one element must be in reshape's output-tensor.
  if (shape_node->size<ReshapeType>() <= 0)
    return false;

  // Create new_shape based on reshape_node/shape
  TagShape new_shape;
  for (uint32_t i = 0; i < shape_node->size<ReshapeType>(); i++)
  {
    TagDim dim;
    dim.value = shape_node->at<ReshapeType>(i);

    new_shape.push_back(dim);
  }

  // inference new_shape dim with -1 value
  if (inference_incomplete_shape(_shape, new_shape) == false)
    return false;

  // indexing for _shape [old_shape_start_idx, old_shape_end_idx)
  uint32_t old_shape_start_idx = 0;
  uint32_t old_shape_end_idx = 1;
  auto old_shape_product = _shape[old_shape_start_idx].value;

  auto expand_range = [&]() -> bool {
    if (old_shape_end_idx >= _shape.size())
      return false;

    old_shape_product *= _shape[old_shape_end_idx].value;
    old_shape_end_idx++;
    return true;
  };

  auto move_to_next_range = [&]() -> bool {
    if (old_shape_end_idx >= _shape.size())
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
    TagDim &target_dim = new_shape[new_shape_idx];

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

    assert(old_shape_product == target_dim.value);
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
bool TaggedShapeAnalyzer::can_remove_transposes(const luci::CircleTranspose *f_tr,
                                                const luci::CircleReshape *m_rs,
                                                const luci::CircleTranspose *b_tr)
{
  assert(loco::must_cast<luci::CircleConst *>(f_tr->perm())->dtype() == DType);
  assert(loco::must_cast<luci::CircleConst *>(m_rs->shape())->dtype() == DType);
  assert(loco::must_cast<luci::CircleConst *>(b_tr->perm())->dtype() == DType);

  const luci::CircleNode *in_tensor = loco::must_cast<luci::CircleNode *>(f_tr->a());

  auto in_shape = extract_shape(in_tensor);
  auto front_perm = extract_const<DType>(f_tr);
  auto back_perm = extract_const<DType>(b_tr);

  init_shape_with_tag(in_shape);

  analyze_transpose(front_perm);

  // TODO : update analyze_reshape gets std::vector<int> like analyze_transpose
  if (not analyze_reshape(m_rs))
    return false;

  analyze_transpose(back_perm);

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
  }

  // check perm and shape are CircleConst node and its' datatype is S32
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
