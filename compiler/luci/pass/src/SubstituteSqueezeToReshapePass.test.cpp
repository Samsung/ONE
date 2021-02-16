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
#include "luci/Pass/SubstituteSqueezeToReshapePass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

using uilist = std::initializer_list<uint32_t>;
using ilist = std::initializer_list<int32_t>;

class PassTestGraph
{
public:
  PassTestGraph() = default;

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

class SubstituteSqueezeToReshapeGraph : public PassTestGraph
{
public:
  SubstituteSqueezeToReshapeGraph() = default;

public:
  void init(const uilist shape_in, const uilist shape_out, const ilist squeeze_dims)
  {
    PassTestGraph::init(shape_in, shape_out);

    _squeeze = _g.nodes()->create<luci::CircleSqueeze>();
    _squeeze->input(_input);
    _squeeze->squeeze_dims(squeeze_dims);
    _squeeze->name("squeeze");

    _output->from(_squeeze);
  }

protected:
  luci::CircleSqueeze *_squeeze = nullptr;
};

class SubstituteSqueezeToReshapeTest : public ::testing::Test
{
public:
  SubstituteSqueezeToReshapeTest() = default;

  void run_pass(void)
  {
    while (_shapeinf.run(_graph.g()) || _pass.run(_graph.g()))
      ;
  }

protected:
  SubstituteSqueezeToReshapeGraph _graph;
  luci::SubstituteSqueezeToReshapePass _pass;
  luci::CircleShapeInferencePass _shapeinf;
};

} // namespace

TEST(SubstituteSqueezeToReshapePassTest, name)
{
  luci::SubstituteSqueezeToReshapePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(SubstituteSqueezeToReshapeTest, simple_with_squeeze_dims)
{
  _graph.init({1, 16, 1, 1}, {1, 16}, {2, 3});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto squeeze = dynamic_cast<luci::CircleSqueeze *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, squeeze);
  auto reshape_shape = loco::must_cast<luci::CircleConst *>(reshape->shape());
  ASSERT_EQ(2, reshape_shape->size<loco::DataType::S32>());
  ASSERT_EQ(1, reshape_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(16, reshape_shape->at<loco::DataType::S32>(1));
}

TEST_F(SubstituteSqueezeToReshapeTest, simple_without_squeeze_dims)
{
  _graph.init({1, 16, 1, 1}, {16}, {});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto squeeze = dynamic_cast<luci::CircleSqueeze *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, squeeze);
  auto reshape_shape = loco::must_cast<luci::CircleConst *>(reshape->shape());
  ASSERT_EQ(1, reshape_shape->size<loco::DataType::S32>());
  ASSERT_EQ(16, reshape_shape->at<loco::DataType::S32>(0));
}

TEST_F(SubstituteSqueezeToReshapeTest, input_with_0_dims)
{
  _graph.init({1, 16, 0, 1}, {16, 0}, {});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto squeeze = dynamic_cast<luci::CircleSqueeze *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, squeeze);
  auto reshape_shape = loco::must_cast<luci::CircleConst *>(reshape->shape());
  ASSERT_EQ(2, reshape_shape->size<loco::DataType::S32>());
  ASSERT_EQ(16, reshape_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(0, reshape_shape->at<loco::DataType::S32>(1));
}

TEST_F(SubstituteSqueezeToReshapeTest, nothing_to_squeeze)
{
  _graph.init({2, 16, 16, 3}, {2, 16, 16, 3}, {});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto squeeze = dynamic_cast<luci::CircleSqueeze *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, squeeze);
}

TEST_F(SubstituteSqueezeToReshapeTest, all_to_squeeze)
{
  _graph.init({1, 1}, {}, {});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto squeeze = dynamic_cast<luci::CircleSqueeze *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, squeeze);
}

TEST_F(SubstituteSqueezeToReshapeTest, wrong_squeeze_dims_NEG)
{
  _graph.init({1, 16, 1, 1}, {1, 16, 1, 1}, {1});

  // shape inference will throw for invalid squeeze_dims
  EXPECT_THROW(run_pass(), std::exception);
}
