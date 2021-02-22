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

#include "moco/Pass/Passes/ConstantFoldStridedSlice.h"
#include "TestHelper.h"

#include <moco/IR/TFNodes.h>
#include <loco.h>

#include <memory>

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

moco::TFConst *const_vector_init(loco::Graph *graph, std::vector<int32_t> values)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();
  auto dim = values.size();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(1);
  const_node->dim(0).set(dim);

  const_node->size<loco::DataType::S32>(dim);
  for (int32_t i = 0; i < dim; ++i)
    const_node->at<loco::DataType::S32>(i) = values[i];

  return const_node;
}

moco::TFConst *const_matrix(loco::Graph *graph, int32_t dimh, int32_t dimw)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(2);
  const_node->dim(0).set(dimh);
  const_node->dim(1).set(dimw);

  auto elements = dimh * dimw;
  const_node->size<loco::DataType::S32>(elements);
  for (int32_t i = 0; i < elements; ++i)
    const_node->at<loco::DataType::S32>(i) = i;

  return const_node;
}

} // namespace

TEST(ConstantFoldStridedSlice, basic_matrix55_11)
{
  loco::Graph graph;

  auto sslice_node = graph.nodes()->create<moco::TFStridedSlice>();
  {
    auto const_input = const_matrix(&graph, 5, 5);
    sslice_node->input(const_input);

    auto const_begin = const_vector_init(&graph, {1, 1});
    sslice_node->begin(const_begin);
    auto const_end = const_vector_init(&graph, {2, 4});
    sslice_node->end(const_end);
    auto const_strides = const_vector_init(&graph, {1, 1});
    sslice_node->strides(const_strides);

    sslice_node->shrink_axis_mask(1);
  }
  setup_output_node(&graph, sslice_node);

  auto pass = std::make_unique<moco::ConstantFoldStridedSlice>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto ssnode = find_first_node_bytype<moco::TFStridedSlice>(&graph);
  ASSERT_EQ(ssnode, nullptr);

  auto ssconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(ssconst, nullptr);
  ASSERT_EQ(ssconst->size<loco::DataType::S32>(), 3);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(0), 6);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(1), 7);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(2), 8);
}

TEST(ConstantFoldStridedSlice, basic_vector4_0)
{
  loco::Graph graph;

  auto sslice_node = graph.nodes()->create<moco::TFStridedSlice>();
  {
    auto const_input = const_vector_init(&graph, {1, 5, 5, 64});
    sslice_node->input(const_input);

    auto const_begin = const_vector_init(&graph, {0});
    sslice_node->begin(const_begin);
    auto const_end = const_vector_init(&graph, {1});
    sslice_node->end(const_end);
    auto const_strides = const_vector_init(&graph, {1});
    sslice_node->strides(const_strides);

    sslice_node->shrink_axis_mask(1);
  }
  setup_output_node(&graph, sslice_node);

  auto pass = std::make_unique<moco::ConstantFoldStridedSlice>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto ssnode = find_first_node_bytype<moco::TFStridedSlice>(&graph);
  ASSERT_EQ(ssnode, nullptr);

  auto ssconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(ssconst, nullptr);
  ASSERT_EQ(ssconst->size<loco::DataType::S32>(), 1);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(0), 1);
}

TEST(ConstantFoldStridedSlice, basic_vector4_1)
{
  loco::Graph graph;

  auto sslice_node = graph.nodes()->create<moco::TFStridedSlice>();
  {
    auto const_input = const_vector_init(&graph, {1, 5, 5, 64});
    sslice_node->input(const_input);

    auto const_begin = const_vector_init(&graph, {1});
    sslice_node->begin(const_begin);
    auto const_end = const_vector_init(&graph, {2});
    sslice_node->end(const_end);
    auto const_strides = const_vector_init(&graph, {1});
    sslice_node->strides(const_strides);

    sslice_node->shrink_axis_mask(1);
  }
  setup_output_node(&graph, sslice_node);

  auto pass = std::make_unique<moco::ConstantFoldStridedSlice>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto ssnode = find_first_node_bytype<moco::TFStridedSlice>(&graph);
  ASSERT_EQ(ssnode, nullptr);

  auto ssconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(ssconst, nullptr);
  ASSERT_EQ(ssconst->size<loco::DataType::S32>(), 1);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(0), 5);
}

TEST(ConstantFoldStridedSlice, basic_vector4_2)
{
  loco::Graph graph;

  auto sslice_node = graph.nodes()->create<moco::TFStridedSlice>();
  {
    auto const_input = const_vector_init(&graph, {1, 5, 5, 64});
    sslice_node->input(const_input);

    auto const_begin = const_vector_init(&graph, {2});
    sslice_node->begin(const_begin);
    auto const_end = const_vector_init(&graph, {3});
    sslice_node->end(const_end);
    auto const_strides = const_vector_init(&graph, {1});
    sslice_node->strides(const_strides);

    sslice_node->shrink_axis_mask(1);
  }
  setup_output_node(&graph, sslice_node);

  auto pass = std::make_unique<moco::ConstantFoldStridedSlice>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto ssnode = find_first_node_bytype<moco::TFStridedSlice>(&graph);
  ASSERT_EQ(ssnode, nullptr);

  auto ssconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(ssconst, nullptr);
  ASSERT_EQ(ssconst->size<loco::DataType::S32>(), 1);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(0), 5);
}

namespace
{

/**
 * @note tfconst_at() implementation should be same as that of inside
 *       ConstantFoldStridedSlice.cpp for valid testing
 */
int32_t tfconst_at(const moco::TFConst *tfconst, const std::vector<uint32_t> &pos)
{
  uint32_t rank = tfconst->rank();
  assert(rank == pos.size());

  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    uint32_t dim = tfconst->dim(r).value();
    element = element * dim + pos.at(r);
  }
  return tfconst->at<loco::DataType::S32>(element);
}

} // namespace

TEST(ConstantFoldStridedSlice, tfconst_at)
{
  loco::Graph graph;

  auto const_node = graph.nodes()->create<moco::TFConst>();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(3);
  const_node->dim(0).set(2);
  const_node->dim(1).set(3);
  const_node->dim(2).set(4);

  auto elements = 2 * 3 * 4;
  const_node->size<loco::DataType::S32>(elements);
  for (int32_t i = 0; i < elements; ++i)
    const_node->at<loco::DataType::S32>(i) = i;
  /*
    [
      [ 0, 1, 2, 3] <- [0,0,0]
      [ 4, 5, 6, 7] <- [0,1,0] [0,1,1] [0,1,2]
      [ 8, 9,10,11]
    ]
    [
      [12,13,14,15]
      [16,17,18,19] <- [1,1,0] [1,1,1]
      [20,21,22,23] <- [1,2,0] [1,2,1] [1,2,2] [1,2,3]
    ]
  */

  ASSERT_EQ(tfconst_at(const_node, {0, 0, 0}), 0);
  ASSERT_EQ(tfconst_at(const_node, {1, 1, 1}), 17);
  ASSERT_EQ(tfconst_at(const_node, {0, 1, 2}), 6);
  ASSERT_EQ(tfconst_at(const_node, {1, 2, 3}), 23);
}
