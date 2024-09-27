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

#define RET_FALSE_UNLESS(condition) \
  if (not(condition))               \
    return false;

struct Dim final
{
  Dim(int32_t v) : value(v) {}

  int32_t value;
  std::vector<uint8_t> tags;
};

using Shape = std::vector<Dim>;

template <loco::DataType DTYPE> class TaggedShapeAnalyzer final
{
public:
  TaggedShapeAnalyzer(const luci::CircleNode *in_node, const luci::CircleTranspose *front_transpose,
                      const luci::CircleReshape *mid_reshape,
                      const luci::CircleTranspose *back_transpose)
    : _in_node(in_node), _front_transpose(front_transpose), _mid_reshape(mid_reshape),
      _back_transpose(back_transpose)
  {
  }

  bool init();
  bool can_remove_transposes();

private:
  void reset_tag(const std::vector<int> &in_shape);
  void analyze_transpose(const std::vector<int> &perm);
  bool analyze_reshape(const std::vector<int> &new_shape);
  bool verify_tag() const;

private:
  const luci::CircleNode *_in_node;
  const luci::CircleTranspose *_front_transpose;
  const luci::CircleReshape *_mid_reshape;
  const luci::CircleTranspose *_back_transpose;

  std::vector<int> _in_shape;
  std::vector<int> _front_perm;
  std::vector<int> _mid_shape;
  std::vector<int> _back_perm;

  Shape _shape;
};

/**
 * @brief initalize _shape based on 'in_shape'
 *
 * @note 'tags' are attached to non-1 valued dimension.
 */
template <loco::DataType DTYPE>
void TaggedShapeAnalyzer<DTYPE>::reset_tag(const std::vector<int> &in_shape)
{
  _shape.clear();
  uint8_t tag = 0u;

  for (auto i : in_shape)
  {
    Dim dim(i);
    if (dim.value != 1)
      dim.tags.push_back(tag++);

    _shape.push_back(dim);
  }
}

/**
 * @brief update _shape based on given perm value
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
template <loco::DataType DTYPE>
void TaggedShapeAnalyzer<DTYPE>::analyze_transpose(const std::vector<int> &perm)
{
  Shape new_shape;
  for (auto i : perm)
  {
    new_shape.push_back(_shape.at(i));
  }
  _shape = new_shape;
}

/**
 * @brief update _shape based on givne new_shape
 *
 * @return False, if it fails to update _shape
 *
 * @note It only supports cases where N dims are combined into 1. (e.g. (A, B, C) -> (A,BxC))
 *       Otherwise, It returns false
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
template <loco::DataType DTYPE>
bool TaggedShapeAnalyzer<DTYPE>::analyze_reshape(const std::vector<int> &new_shape)
{
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

  // Create new_tagged_shape based on new_shape
  Shape new_tagged_shape;

  uint32_t new_shape_idx = 0;
  while (new_shape_idx < new_shape.size())
  {
    int target_dim = new_shape[new_shape_idx];

    // Ignore dim == 1
    if (target_dim == 1)
    {
      new_tagged_shape.push_back(Dim(1));
      new_shape_idx++;
      continue;
    }

    while (old_shape_product < target_dim)
    {
      if (expand_range() == false)
        break;
    }

    if (old_shape_product != target_dim)
      return false;

    assert(old_shape_product == target_dim);

    Dim dim(target_dim);
    for (uint32_t idx = old_shape_start_idx; idx < old_shape_end_idx; idx++)
    {
      const auto &old_tags = _shape[idx].tags;
      dim.tags.insert(dim.tags.end(), old_tags.begin(), old_tags.end());
    }
    new_tagged_shape.push_back(dim);

    new_shape_idx++;
    move_to_next_range();
  }
  _shape = new_tagged_shape;
  return true;
}

template <loco::DataType DTYPE> bool TaggedShapeAnalyzer<DTYPE>::verify_tag() const
{
  // check whether tags in _shape are incremental
  uint8_t tag = 0u;
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
 * @brief initalize _in_shape, _front_perm, _mid_shape and _back_perm
 *
 * @return false, if failed to initalize _in_shape _front_perm, _mid_shape and _back_perm
 */
template <loco::DataType DTYPE> bool TaggedShapeAnalyzer<DTYPE>::init()
{
  auto extract_const = [](const luci::CircleConst *const_node) -> std::vector<int> {
    std::vector<int> values;
    auto size = const_node->size<DTYPE>();
    for (auto i = 0u; i < size; ++i)
    {
      auto v = const_node->at<DTYPE>(i);
      values.push_back(v);
    }
    return values;
  };

  auto extract_shape = [](const luci::CircleNode *node) -> std::vector<int> {
    std::vector<int> shape;
    auto rank = node->rank();
    for (auto i = 0u; i < rank; ++i)
    {
      shape.push_back(node->dim(i).value());
    }
    return shape;
  };

  const auto front_perm = dynamic_cast<luci::CircleConst *>(_front_transpose->perm());
  {
    RET_FALSE_UNLESS(front_perm != nullptr);
    RET_FALSE_UNLESS(front_perm->dtype() == DTYPE);
  }

  const auto back_perm = dynamic_cast<luci::CircleConst *>(_back_transpose->perm());
  {
    RET_FALSE_UNLESS(back_perm != nullptr);
    RET_FALSE_UNLESS(back_perm->dtype() == DTYPE);
  }

  _in_shape = extract_shape(_in_node);
  _front_perm = extract_const(front_perm);
  _mid_shape = extract_shape(_mid_reshape);
  _back_perm = extract_const(back_perm);

  auto all_known = [](const std::vector<int> &v) -> bool {
    for (auto i : v)
      if (i < 0)
        return false;
    return true;
  };

  RET_FALSE_UNLESS(all_known(_in_shape));
  RET_FALSE_UNLESS(all_known(_front_perm));
  RET_FALSE_UNLESS(all_known(_mid_shape));
  RET_FALSE_UNLESS(all_known(_back_perm));

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
template <loco::DataType DTYPE> bool TaggedShapeAnalyzer<DTYPE>::can_remove_transposes()
{
  reset_tag(_in_shape);

  analyze_transpose(_front_perm);

  if (not analyze_reshape(_mid_shape))
    return false;

  analyze_transpose(_back_perm);

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
  RET_FALSE_UNLESS(mid_reshape != nullptr);

  const auto front_transpose = dynamic_cast<luci::CircleTranspose *>(mid_reshape->tensor());
  RET_FALSE_UNLESS(front_transpose != nullptr);

  const auto in_node = loco::must_cast<luci::CircleNode *>(front_transpose->a());

  // for now, handle only rank reduction equal (not expansion) cases
  const auto output_rank = back_transpose->rank();
  const auto input_rank = front_transpose->rank();
  if (input_rank < output_rank)
    return false;

  // analyze pattern to check this pass is applicable
  TaggedShapeAnalyzer<loco::DataType::S32> analyzer(in_node, front_transpose, mid_reshape,
                                                    back_transpose);
  RET_FALSE_UNLESS(analyzer.init());
  RET_FALSE_UNLESS(analyzer.can_remove_transposes());

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
