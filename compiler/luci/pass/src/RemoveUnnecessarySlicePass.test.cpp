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
#include "luci/Pass/RemoveUnnecessarySlicePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void create_remove_unnecessary_slice(loco::Graph *g,
                                     const std::initializer_list<uint32_t> input_shape, bool remove)
{
  assert(g);

  // Input Create.
  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->shape_status(luci::ShapeStatus::VALID);
  input->rank(input_shape.size());
  input->shape(input_shape);

  // Begin Create.
  auto begin = g->nodes()->create<luci::CircleConst>();
  begin->dtype(loco::DataType::S32);
  begin->size<loco::DataType::S32>(input_shape.size());
  begin->rank(1);
  begin->dim(0).set(input_shape.size());
  for (int i = 0; i < input_shape.size(); ++i)
  {
    begin->at<loco::DataType::S32>(i) = remove ? 0 : 1;
  }

  // Size Create.
  auto size = g->nodes()->create<luci::CircleConst>();
  size->dtype(loco::DataType::S32);
  size->size<loco::DataType::S32>(input_shape.size());
  size->rank(1);
  size->dim(0).set(input_shape.size());
  for (int i = 0; i < input_shape.size(); ++i)
  {
    size->at<loco::DataType::S32>(i) = -1;
  }

  // Slice Node create.
  auto slice = g->nodes()->create<luci::CircleSlice>();
  slice->dtype(loco::DataType::S32);
  slice->input(input);
  slice->begin(begin);
  slice->size(size);

  // Output Connect.
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(slice);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
}

} // namespace

TEST(RemoveUnnecessarySlicePass, name)
{
  luci::RemoveUnnecessarySlicePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessarySlicePass, remove_no_effect_slice)
{
  auto graph = loco::make_graph();
  create_remove_unnecessary_slice(graph.get(), {2, 4, 2, 3}, true);
  luci::CircleSlice *slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto slice = dynamic_cast<luci::CircleSlice *>(node);
    if (not slice)
      continue;
    slice_node = slice;
    break;
  }
  ASSERT_NE(nullptr, slice_node);
  luci::RemoveUnnecessarySlicePass pass;
  while (pass.run(graph.get()))
    ;
  slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto slice = dynamic_cast<luci::CircleSlice *>(node);
    if (not slice)
      continue;
    slice_node = slice;
    break;
  }
  ASSERT_EQ(nullptr, slice_node);
}

TEST(RemoveUnnecessarySlicePass, remove_no_effect_slice_NEG)
{
  auto graph = loco::make_graph();
  create_remove_unnecessary_slice(graph.get(), {2, 4, 2, 3}, false);
  luci::CircleSlice *slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto slice = dynamic_cast<luci::CircleSlice *>(node);
    if (not slice)
      continue;
    slice_node = slice;
    break;
  }
  ASSERT_NE(nullptr, slice_node);
  luci::RemoveUnnecessarySlicePass pass;
  while (pass.run(graph.get()))
    ;
  slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto slice = dynamic_cast<luci::CircleSlice *>(node);
    if (not slice)
      continue;
    slice_node = slice;
    break;
  }
  ASSERT_NE(nullptr, slice_node);
}
