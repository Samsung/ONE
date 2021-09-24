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

#include "luci/Pass/FuseAddWithFullyConnectedPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

// TODO Reduce duplicate codes in ResolveCustomOpMatMulPass.cpp
template <typename T>
luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<T> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape.at(i);
    size *= shape.at(i);
  }
  node->shape_status(luci::ShapeStatus::VALID);

#define INIT_VALUES(DT)                          \
  {                                              \
    node->size<DT>(size);                        \
    for (uint32_t i = 0; i < values.size(); ++i) \
      node->at<DT>(i) = values[i];               \
  }

  switch (dtype)
  {
    case loco::DataType::U8:
      INIT_VALUES(loco::DataType::U8);
      break;
    case loco::DataType::S16:
      INIT_VALUES(loco::DataType::S16);
      break;
    case loco::DataType::S32:
      INIT_VALUES(loco::DataType::S32);
      break;
    case loco::DataType::FLOAT32:
      INIT_VALUES(loco::DataType::FLOAT32)
      break;
    default:
      INTERNAL_EXN("create_const_node called with unsupported type");
      break;
  }
  return node;
}

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *         [FC]
 *           |
 *     [Add w/ Relu]
 *
 *  AFTER
 *
 *      [FC w/ Relu] (bias updated)
 *
 */
class FCAddGraphlet
{
public:
  FCAddGraphlet() = default;

  void init(loco::Graph *g)
  {
    std::vector<float> weights_val(16 * 4);
    _fc_f = create_const_node(g, loco::DataType::FLOAT32, {16, 4}, weights_val);

    std::vector<float> bias_val(16);
    _fc_b = create_const_node(g, loco::DataType::FLOAT32, {1, 16}, bias_val);

    _fc = g->nodes()->create<luci::CircleFullyConnected>();
    _fc->weights(_fc_f);
    _fc->bias(_fc_b);
    _fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc->dtype(loco::DataType::FLOAT32);
    _fc->shape({1, 16});
    _fc->name("fc");

    std::vector<float> addition_val;
    for (uint32_t i = 0; i < 16; i++)
      addition_val.push_back(static_cast<float>(i));
    _add_c = create_const_node(g, loco::DataType::FLOAT32, {1, 16}, addition_val);

    _add = g->nodes()->create<luci::CircleAdd>();
    _add->x(_fc);
    _add->y(_add_c);
    _add->fusedActivationFunction(luci::FusedActFunc::RELU);
    _add->dtype(loco::DataType::FLOAT32);
    _add->shape({1, 16});
    _add->name("add");
  }

public:
  luci::CircleFullyConnected *fc() { return _fc; }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_fc_f = nullptr;
  luci::CircleConst *_fc_b = nullptr;
  luci::CircleConst *_add_c = nullptr;
};

class FuseAddWithFCTestGraph : public TestIOGraph, public FCAddGraphlet
{
public:
  FuseAddWithFCTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 4}, {1, 16});
    FCAddGraphlet::init(g());

    _fc->input(input());

    output()->from(_add);
  }
};

class FuseAddWithFullyConnectedPassTest : public ::testing::Test
{
public:
  FuseAddWithFCTestGraph g;
  luci::FuseAddWithFullyConnectedPass pass;
};

} // namespace

TEST_F(FuseAddWithFullyConnectedPassTest, simple_test)
{
  g.init();

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto fc = dynamic_cast<luci::CircleFullyConnected *>(g.output()->from());
  EXPECT_NE(nullptr, fc);

  auto bias = loco::must_cast<luci::CircleConst *>(g.fc()->bias());
  for (uint32_t i = 0; i < bias->size<loco::DataType::FLOAT32>(); i++)
  {
    EXPECT_EQ(i, bias->at<loco::DataType::FLOAT32>(i));
  }
}
