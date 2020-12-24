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
#include "luci/Pass/RemoveUnnecessaryStridedSlicePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void create_remove_unnecessary_strided_slice(loco::Graph *g,
                                     const std::initializer_list<uint32_t> input_shape, bool remove)
{
  assert(g);

  // Input create
  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->shape_status(luci::ShapeStatus::VALID);
  input->rank(input_shape.size());
  input->shape(input_shape);

  // Begin create
  auto begin = g->nodes()->create<luci::CircleConst>();
  begin->dtype(loco::DataType::S32);
  begin->size<loco::DataType::S32>(input_shape.size());
  begin->rank(1);
  begin->dim(0).set(input_shape.size());
  for (int i = 0; i < input_shape.size(); ++i)
  {
    begin->at<loco::DataType::S32>(i) = remove ? 0 : 1;
  }

  // Strides create
  auto strides = g->nodes()->create<luci::CircleConst>();
  strides->dtype(loco::DataType::S32);
  strides->size<loco::DataType::S32>(input_shape.size());
  strides->rank(1);
  strides->dim(0).set(input_shape.size());
  for (int i = 0; i < input_shape.size(); ++i)
  {
    strides->at<loco::DataType::S32>(i) = -1;
  }

  // StridedSlice Node create
  auto strided_slice = g->nodes()->create<luci::CircleStridedSlice>();
  strided_slice->dtype(loco::DataType::S32);
  strided_slice->input(input);
  strided_slice->begin(begin);
  strided_slice->strides(strides);

  // Output connect
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(strided_slice);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
}

} // namespace

TEST(RemoveUnnecessaryStridedSlicePass, remove_unnecessary_strided_slice)
{
  auto graph = loco::make_graph();
  create_remove_unnecessary_strided_slice(graph.get(), {2, 4, 2, 3}, true);
  luci::CircleStridedSlice *strided_slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto strided_slice = dynamic_cast<luci::CircleStridedSlice *>(node);
    if (not strided_slice)
      continue;
    strided_slice_node = strided_slice;
    break;
  }
  ASSERT_NE(nullptr, strided_slice_node);
  luci::RemoveUnnecessaryStridedSlicePass pass;
  while (pass.run(graph.get()))
    ;
  strided_slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto strided_slice = dynamic_cast<luci::CircleStridedSlice *>(node);
    if (not strided_slice)
      continue;
    strided_slice_node = strided_slice;
    break;
  }
  ASSERT_EQ(nullptr, strided_slice_node);
}

TEST(RemoveUnnecessarySlicePass, remove_unnecessary_strided_slice_NEG)
{
  auto graph = loco::make_graph();
  create_remove_unnecessary_strided_slice(graph.get(), {2, 4, 2, 3}, false);
  luci::CircleStridedSlice *strided_slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto strided_slice = dynamic_cast<luci::CircleStridedSlice *>(node);
    if (not strided_slice)
      continue;
    strided_slice_node = strided_slice;
    break;
  }
  ASSERT_NE(nullptr, strided_slice_node);
  luci::RemoveUnnecessaryStridedSlicePass pass;
  while (pass.run(graph.get()))
    ;
  strided_slice_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto strided_slice = dynamic_cast<luci::CircleStridedSlice *>(node);
    if (not strided_slice)
      continue;
    strided_slice_node = strided_slice;
    break;
  }
  ASSERT_NE(nullptr, strided_slice_node);
}
