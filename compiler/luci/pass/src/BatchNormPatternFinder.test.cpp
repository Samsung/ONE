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

#include "BatchNormPatternFinder.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

#include <initializer_list>

namespace luci
{
namespace test
{

using uilist = std::initializer_list<uint32_t>;
using ilist = std::initializer_list<int32_t>;

/**
 * @brief Graph with Input and Output
 */
class TestIOGraph
{
public:
  TestIOGraph() = default;

public:
  void init(const uilist shape_in, const uilist shape_out)
  {
    _graph_input = _g.inputs()->create();
    _graph_output = _g.outputs()->create();

    _input = _g.nodes()->create<luci::CircleInput>();
    _input->shape(shape_in);
    _input->shape_status(luci::ShapeStatus::VALID);
    _input->name("input");

    _output = _g.nodes()->create<luci::CircleOutput>();
    _output->shape(shape_out);
    _output->shape_status(luci::ShapeStatus::VALID);
    _output->name("output");

    _input->index(_graph_input->index());
    _output->index(_graph_output->index());

    auto input_shape = std::make_unique<loco::TensorShape>();
    set(input_shape.get(), shape_in);
    _graph_input->shape(std::move(input_shape));

    auto output_shape = std::make_unique<loco::TensorShape>();
    set(output_shape.get(), shape_out);
    _graph_output->shape(std::move(output_shape));
  }

protected:
  void set(loco::TensorShape *shape, const uilist &values)
  {
    uint32_t r = 0;
    shape->rank(values.size());
    for (auto v : values)
      shape->dim(r++).set(v);
  }

public:
  loco::Graph *g(void) { return &_g; }
  luci::CircleOutput *output(void) { return _output; }

protected:
  loco::Graph _g;
  loco::GraphInput *_graph_input = nullptr;
  loco::GraphOutput *_graph_output = nullptr;
  luci::CircleInput *_input = nullptr;
  luci::CircleOutput *_output = nullptr;
};

/**
 * @brief Graphlet with Add and Const as beta from BatchNorm
 */
class AddBetaGraphlet
{
public:
  AddBetaGraphlet() = default;

  void init(loco::Graph *g, const uilist shape, luci::FusedActFunc actf)
  {
    _add = g->nodes()->create<luci::CircleAdd>();
    _add_beta = g->nodes()->create<luci::CircleConst>();

    _add->dtype(loco::DataType::FLOAT32);
    _add_beta->dtype(loco::DataType::FLOAT32);

    _add->fusedActivationFunction(actf);

    assert(shape.size() > 0);
    auto last_it = std::prev(shape.end(), 1);
    auto channel_size = *last_it;

    _add->shape(shape);
    _add_beta->shape({channel_size});
    _add_beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
      _add_beta->at<loco::DataType::FLOAT32>(i) = i;

    _add->name("add");
    _add_beta->name("add_beta");
  }

public:
  luci::CircleAdd *add() { return _add; }

protected:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_add_beta = nullptr;
};

/**
 * @brief Graphlet with Mul and Const as gamma from BatchNorm
 */
class MulGammaGraphlet
{
public:
  MulGammaGraphlet() = default;

  void init(loco::Graph *g, const uilist shape, luci::FusedActFunc actf)
  {
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_gamma = g->nodes()->create<luci::CircleConst>();

    _mul->dtype(loco::DataType::FLOAT32);
    _mul_gamma->dtype(loco::DataType::FLOAT32);

    _mul->fusedActivationFunction(actf);

    assert(shape.size() > 0);
    auto last_it = std::prev(shape.end(), 1);
    auto channel_size = *last_it;

    _mul->shape(shape);
    _mul_gamma->shape({channel_size});
    _mul_gamma->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
      _mul_gamma->at<loco::DataType::FLOAT32>(i) = i;

    _mul->name("mul");
    _mul_gamma->name("mul_gamma");
  }

public:
  luci::CircleMul *mul(void) { return _mul; }

protected:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_gamma = nullptr;
};

/**
 * @brief Graph of Mul-Add pattern from BatchNorm
 */
class MulAddGraph : public TestIOGraph, public AddBetaGraphlet, public MulGammaGraphlet
{
public:
  MulAddGraph() = default;

  void init(const uilist shape_in, const uilist shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    MulGammaGraphlet::init(g(), shape_in, luci::FusedActFunc::NONE);
    AddBetaGraphlet::init(g(), shape_out, luci::FusedActFunc::RELU);

    // connect network
    _mul->x(_input);
    _mul->y(_mul_gamma);
    _add->x(_mul);
    _add->y(_add_beta);
    _output->from(_add);
  }
};

/**
 * @brief Graph of Add with Const
 */
class AddGraph : public TestIOGraph, public AddBetaGraphlet
{
public:
  AddGraph() = default;

  void init(const uilist shape_in, const uilist shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    AddBetaGraphlet::init(g(), shape_in, luci::FusedActFunc::RELU);

    // connect network
    _add->x(_input);
    _add->y(_add_beta);
    _output->from(_add);
  }
};

} // namespace test
} // namespace luci

TEST(BatchNormPatternFinderTest, is_batchnorm_add)
{
  luci::test::MulAddGraph mag;

  mag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;

  auto res = luci::is_batchnorm_add(mag.add(), mul, beta);
  ASSERT_TRUE(res);
  ASSERT_NE(nullptr, mul);
  ASSERT_NE(nullptr, beta);
}

TEST(BatchNormPatternFinderTest, is_batchnorm_add2)
{
  luci::test::MulAddGraph mag;

  mag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  auto res = luci::is_batchnorm_add(mag.add());
  ASSERT_TRUE(res);
}

TEST(BatchNormPatternFinderTest, is_batchnorm_add_NEG)
{
  luci::test::AddGraph ag;

  ag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;

  auto res = luci::is_batchnorm_add(ag.add(), mul, beta);
  ASSERT_FALSE(res);
}

TEST(BatchNormPatternFinderTest, is_batchnorm_mul)
{
  luci::test::MulAddGraph mag;

  mag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  luci::CircleNode *pred = nullptr;
  luci::CircleConst *gamma = nullptr;

  auto res = luci::is_batchnorm_mul(mag.mul(), pred, gamma);
  ASSERT_TRUE(res);
  ASSERT_NE(nullptr, pred);
  ASSERT_NE(nullptr, gamma);
}
