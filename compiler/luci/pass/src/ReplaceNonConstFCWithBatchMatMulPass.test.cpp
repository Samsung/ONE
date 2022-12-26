/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ReplaceNonConstFCWithBatchMatMulPass.h"

#include <luci/test/TestIOGraph.h>
#include <luci/IR/CircleNodes.h>

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
 *   [IFM1] [IFM2] [BIAS]
 *        \   |   /
 *          [FC]
 *            |
 *          [Res]
 *
 *  AFTER
 *   [IFM1] [IFM2]
 *        \   |
 *      [BatchMatMul] [BIAS]
 *              \      /
 *               [Add]
 *                 |
 *               [Res]
 *
 */
struct FCGraphlet
{
public:
  FCGraphlet() = default;
  virtual ~FCGraphlet() = default;

  void init(loco::Graph *g, const ShapeU32 r_shape, const float bv)
  {
    _tr_y = g->nodes()->create<luci::CircleTranspose>();
    _tr_y->a(_y);
    std::vector<int32_t> tr_val = {1, 0};
    _tr_y->perm(create_const_node(g, loco::DataType::S32, {2}, tr_val));

    _fc = g->nodes()->create<luci::CircleFullyConnected>();
    _fc->input(_x);
    _fc->weights(_tr_y);
    _fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc->dtype(loco::DataType::FLOAT32);
    _fc->shape(r_shape);
    auto l = _fc->dim(_fc->rank() - 1).value();
    std::vector<float> bias_val(l, bv);
    _fc->bias(create_const_node(g, loco::DataType::FLOAT32, {l}, bias_val));
    _fc->name("fc");
  }

public:
  luci::CircleFullyConnected *fc() { return _fc; }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleTranspose *_tr_y = nullptr;
  luci::CircleInput *_x = nullptr;
  luci::CircleInput *_y = nullptr;
};

struct FCGraph : public TestIsGraphlet<2>, public TestOGraphlet, public FCGraphlet
{
  FCGraph() = default;
  virtual ~FCGraph() = default;
  void init(const ShapeU32 x_shape, const ShapeU32 y_shape, const ShapeU32 r_shape, const float bv)
  {
    TestIsGraphlet<2>::init(g(), {x_shape, y_shape});
    TestOGraphlet::init(g(), r_shape);
    _x = input(0);
    _y = input(1);
    FCGraphlet::init(g(), r_shape, bv);
    output()->from(_fc);
  }
};

class ReplaceNonConstFCWithBatchMatMulPassTest : public ::testing::Test
{
public:
  FCGraph g;
  luci::ReplaceNonConstFCWithBatchMatMulPass pass;
};

} // namespace

TEST_F(ReplaceNonConstFCWithBatchMatMulPassTest, simple_test)
{
  g.init({2, 3}, {2, 3}, {2, 2}, 0.0f);

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto res = dynamic_cast<luci::CircleReshape *>(g.output()->from());
  EXPECT_NE(nullptr, res);
}

TEST_F(ReplaceNonConstFCWithBatchMatMulPassTest, nonzero_bias_test)
{
  g.init({2, 3}, {2, 3}, {2, 2}, 1.0f);

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto mm = dynamic_cast<luci::CircleAdd *>(g.output()->from());
  EXPECT_NE(nullptr, mm);
}

TEST_F(ReplaceNonConstFCWithBatchMatMulPassTest, wrong_op_NEG)
{
  loco::Graph g;

  auto inp = g.nodes()->create<luci::CircleInput>();
  auto relu = g.nodes()->create<luci::CircleRelu>();
  relu->features(inp);

  luci::ReplaceNonConstFCWithBatchMatMulPass pass;
  auto changed = pass.run(&g);

  EXPECT_EQ(false, changed);
}
