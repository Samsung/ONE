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

class Shape
{
  friend class SECursor;

private:
  struct ShapeElement
  {
    uint32_t value;
    std::vector<char> tags;
  };

public:
  void add_elem(uint32_t v)
  {
    ShapeElement se;
    se.value = v;
    elements.push_back(ShapeElement(se));
  }

  void add_elem_with_tag(uint32_t v, char tag)
  {
    ShapeElement se;
    se.value = v;
    se.tags.push_back(tag);
    elements.push_back(ShapeElement(se));
  }

  void add_elem(ShapeElement e) { elements.push_back(e); }

  std::vector<ShapeElement> elements;
};

class SECursor
{
private:
  uint32_t _start_idx; // [start_idx, _end_idx) points Shape.elements range
  uint32_t _end_idx;
  uint32_t _product; // product of [se[_start_idx], se[_end_idx -1]]

  std::vector<Shape::ShapeElement> &_se;

public:
  SECursor(Shape &s) : _start_idx(0), _end_idx(1), _se(s.elements)
  {
    _product = s.elements.at(0).value;
  }

  void move_next_range()
  {
    _start_idx = _end_idx;
    _end_idx = _end_idx + 1;

    if (_start_idx >= _se.size())
      return;

    _product = _se.at(_start_idx).value;
  }

  void expand_range()
  {
    if (_end_idx >= _se.size())
      return;

    _product *= _se.at(_end_idx).value;
    _end_idx++;
  }

  // get tags all tags from pointed elements
  std::vector<char> get_tags()
  {
    std::vector<char> tags;
    for (uint32_t i = _start_idx; i < _end_idx; i++)
    {
      const auto &indexed_tag = _se.at(i).tags;
      tags.insert(std::end(tags), std::begin(indexed_tag), std::end(indexed_tag));
    }
    return tags;
  }

  // append tags at the end of each pointed elements' tag
  void append_tags(std::vector<char> tags)
  {
    for (uint32_t i = _start_idx; i < _end_idx; i++)
    {
      auto &indexed_se = _se.at(i);

      if (indexed_se.value == 1)
        continue;

      indexed_se.tags.insert(std::end(indexed_se.tags), std::begin(tags), std::end(tags));
    }
  }

  void append_incremental_tags(const char tag)
  {
    char incremental_tag = tag;
    for (uint32_t i = _start_idx; i < _end_idx; i++)
    {
      auto &indexed_se = _se.at(i);

      if (indexed_se.value == 1)
        continue;

      indexed_se.tags.push_back(incremental_tag++);
    }
  }

  bool is_end() const { return _se.size() <= _start_idx; }

  int size() const { return _end_idx - _start_idx; }

  uint32_t product() const { return _product; }
};

class PatternSimulator
{
public:
  PatternSimulator(const luci::CircleTranspose *f_tr, const luci::CircleReshape *m_rs,
                   const luci::CircleTranspose *b_tr)
    : _front_transpose(f_tr), _mid_reshape(m_rs), _back_transpose(b_tr), _main_tag('A'),
      _sub_tag('a')
  {
    assert(_main_tag < _sub_tag);
  }

  void simulate();

  bool areTransposesUnnecessary() const;

  luci::CircleReshape *createAlternative(loco::Graph *graph);

private:
  void init_shape(const luci::CircleNode *in_tensor);
  void simulate_transpose(const luci::CircleTranspose *transpose_node);
  void simulate_reshape(const luci::CircleReshape *reshape_node);

  const luci::CircleTranspose *_front_transpose;
  const luci::CircleReshape *_mid_reshape;
  const luci::CircleTranspose *_back_transpose;

  // 'tag' is used to track where each elements come from.
  // If there are indentical tags between elements, `_sub_tag` are  used to distinguish.
  const char _main_tag;
  const char _sub_tag;
  Shape _shape;
};

void PatternSimulator::init_shape(const luci::CircleNode *in_tensor)
{
  char tag = _main_tag;

  for (uint32_t i = 0; i < in_tensor->rank(); i++)
  {
    uint32_t value = in_tensor->dim(i).value();

    if (value == 1)
    {
      // 'tag' is used to track in reshape
      // If value == 1, tracking is meaningless.
      _shape.add_elem(value);
    }
    else
      _shape.add_elem_with_tag(value, tag++);
  }
}

void PatternSimulator::simulate_transpose(const luci::CircleTranspose *transpose_node)
{
  const luci::CircleConst *perm_node = dynamic_cast<luci::CircleConst *>(transpose_node->perm());

  assert(perm_node->dtype() == loco::DataType::S32);

  const auto size = perm_node->size<loco::DataType::S32>();
  assert(_shape.elements.size() == size); // permution doesn't change rank

  Shape new_shape;
  for (uint32_t i = 0; i < size; i++)
  {
    auto perm_idx = perm_node->at<loco::DataType::S32>(i);
    new_shape.add_elem(_shape.elements.at(perm_idx));
  }
  _shape = new_shape;
}

void PatternSimulator::simulate_reshape(const luci::CircleReshape *reshape_node)
{
  const luci::CircleConst *shape_node = dynamic_cast<luci::CircleConst *>(reshape_node->shape());
  assert(shape_node != nullptr); // TODO handle shape_node is nullptr
  assert(shape_node->dtype() == loco::DataType::S32);

  Shape new_shape;
  for (uint32_t i = 0; i < shape_node->size<loco::DataType::S32>(); i++)
  {
    new_shape.add_elem(shape_node->at<loco::DataType::S32>(i));
  }

  SECursor old_cursor(_shape);
  SECursor new_cursor(new_shape);

  // while travesing, append tags from previous shape
  while (!old_cursor.is_end() && !new_cursor.is_end())
  {
    if (old_cursor.product() == new_cursor.product())
    {
      auto tags = old_cursor.get_tags();
      new_cursor.append_tags(tags);

      if (new_cursor.size() > 1)
        new_cursor.append_incremental_tags(_sub_tag);

      old_cursor.move_next_range();
      new_cursor.move_next_range();
    }
    else if (old_cursor.product() > new_cursor.product())
    {
      new_cursor.expand_range();
    }
    else
    {
      old_cursor.expand_range();
    }
  }

  _shape = new_shape;
}

void PatternSimulator::simulate()
{
  const luci::CircleNode *in_tensor = dynamic_cast<luci::CircleNode *>(_front_transpose->a());
  init_shape(in_tensor);
  simulate_transpose(_front_transpose);
  simulate_reshape(_mid_reshape);
  simulate_transpose(_back_transpose);
}

bool PatternSimulator::areTransposesUnnecessary() const
{
  auto is_incremental_tags = [&](const std::vector<char> &tags) -> bool {
    char current_tag = _main_tag;
    for (auto t : tags)
    {
      if (current_tag <= t)
        current_tag = t;
      else
        return false;
    }
    return true;
  };

  auto current_tags = std::vector<char>(1, _main_tag);

  for (uint32_t i = 0; i < _shape.elements.size(); i++)
  {
    const auto &next_tags = _shape.elements.at(i).tags;

    if (next_tags.size() == 0)
      continue;

    if (is_incremental_tags(next_tags) == false)
      return false;

    std::string current_tag_str = std::string(std::begin(current_tags), std::end(current_tags));
    std::string next_tag_str = std::string(std::begin(next_tags), std::end(next_tags));

    if (current_tag_str <= next_tag_str)
      current_tags = next_tags;
    else
      return false;
  }
  return true;
}

luci::CircleReshape *PatternSimulator::createAlternative(loco::Graph *graph)
{
  std::string pattern_name =
    _front_transpose->name() + "/" + _mid_reshape->name() + "/" + _back_transpose->name();

  auto shape_node = graph->nodes()->create<luci::CircleConst>();
  {
    shape_node->dtype(loco::DataType::S32);
    shape_node->rank(1);
    shape_node->dim(0).set(_back_transpose->rank());
    shape_node->size<loco::DataType::S32>(_back_transpose->rank());
    for (uint32_t i = 0; i < _back_transpose->rank(); i++)
    {
      shape_node->at<loco::DataType::S32>(i) = _back_transpose->dim(i).value();
    }
    shape_node->shape_status(luci::ShapeStatus::VALID);
    shape_node->name(pattern_name + "/shape");
  }

  auto reshape_node = graph->nodes()->create<luci::CircleReshape>();
  {
    reshape_node->name(pattern_name);
    reshape_node->tensor(_front_transpose->a());
    reshape_node->shape(shape_node);
    luci::add_origin(reshape_node, luci::get_origin(_mid_reshape));
  }

  return reshape_node;
}

bool remove_no_effect_reshape(luci::CircleTranspose *node)
{
  // find 'front_transpose - mid_reshape - back_transpose' pattern
  const auto back_transpose = node;
  const auto mid_reshape = dynamic_cast<luci::CircleReshape *>(back_transpose->a());
  {
    if (not mid_reshape)
      return false;

    const auto shape_node = dynamic_cast<luci::CircleConst *>(mid_reshape->shape());
    if (shape_node == nullptr)
      return false;
  }
  const auto front_transpose = dynamic_cast<luci::CircleTranspose *>(mid_reshape->tensor());
  {
    if (not front_transpose)
      return false;
  }

  // simulate
  PatternSimulator sim(front_transpose, mid_reshape, back_transpose);

  sim.simulate();

  if (sim.areTransposesUnnecessary() == false)
    return false;

  // replace
  auto new_reshape_node = sim.createAlternative(node->graph());
  replace(node).with(new_reshape_node);
  return true;
}

} // namespace

namespace luci
{

bool RemoveUnnecessaryTransposeNetPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto transpose_node = dynamic_cast<luci::CircleTranspose *>(node))
    {
      if (remove_no_effect_reshape(transpose_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
