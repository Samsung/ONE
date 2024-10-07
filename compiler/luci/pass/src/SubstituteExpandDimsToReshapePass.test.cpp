/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Pass/SubstituteExpandDimsToReshapePass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

using uilist = std::initializer_list<uint32_t>;

class PassTestGraph
{
public:
  PassTestGraph() = default;

public:
  void init(const uilist shape_in, const uilist shape_out, const int val)
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

    _const = _g.nodes()->create<luci::CircleConst>();
    _const->dtype(loco::DataType::S32);
    _const->size<loco::DataType::S32>(1);
    _const->at<loco::DataType::S32>(0) = val;
    _const->name("const");

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
  luci::CircleConst *_const = nullptr;
};

class SubstituteExpandDimsToReshapeGraph : public PassTestGraph
{
public:
  SubstituteExpandDimsToReshapeGraph() = default;

public:
  void init(const uilist shape_in, const uilist shape_out, const int axis)
  {
    PassTestGraph::init(shape_in, shape_out, axis);

    _expand_dims = _g.nodes()->create<luci::CircleExpandDims>();
    _expand_dims->input(_input);
    _expand_dims->axis(_const);
    _expand_dims->name("expand_dims");

    _output->from(_expand_dims);
  }

protected:
  luci::CircleExpandDims *_expand_dims = nullptr;
};

class SubstituteExpandDimsToReshapeTest : public ::testing::Test
{
public:
  SubstituteExpandDimsToReshapeTest() = default;

  void run_pass(void)
  {
    while (_shapeinf.run(_graph.g()) || _pass.run(_graph.g()))
      ;
  }

protected:
  SubstituteExpandDimsToReshapeGraph _graph;
  luci::SubstituteExpandDimsToReshapePass _pass;
  luci::CircleShapeInferencePass _shapeinf;
};

} // namespace

TEST(SubstituteExpandDimsToReshapePassTest, name)
{
  luci::SubstituteExpandDimsToReshapePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(SubstituteExpandDimsToReshapeTest, simple_with_expand_dims_1)
{
  _graph.init({2, 16}, {2, 1, 16}, 1);

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto expand_dims = dynamic_cast<luci::CircleExpandDims *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, expand_dims);
  auto reshape_shape = loco::must_cast<luci::CircleConst *>(reshape->shape());
  ASSERT_EQ(3, reshape_shape->size<loco::DataType::S32>());
  ASSERT_EQ(2, reshape_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, reshape_shape->at<loco::DataType::S32>(1));
  ASSERT_EQ(16, reshape_shape->at<loco::DataType::S32>(2));
}

TEST_F(SubstituteExpandDimsToReshapeTest, simple_with_expand_dims_M1)
{
  _graph.init({2, 3, 4}, {2, 3, 4, 1}, -1);

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto expand_dims = dynamic_cast<luci::CircleExpandDims *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, expand_dims);
  auto reshape_shape = loco::must_cast<luci::CircleConst *>(reshape->shape());
  ASSERT_EQ(4, reshape_shape->size<loco::DataType::S32>());
  ASSERT_EQ(2, reshape_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(3, reshape_shape->at<loco::DataType::S32>(1));
  ASSERT_EQ(4, reshape_shape->at<loco::DataType::S32>(2));
  ASSERT_EQ(1, reshape_shape->at<loco::DataType::S32>(3));
}

TEST_F(SubstituteExpandDimsToReshapeTest, simple_with_expand_dims_2)
{
  _graph.init({16, 3, 1}, {16, 3, 1, 1}, 2);

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto expand_dims = dynamic_cast<luci::CircleExpandDims *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, expand_dims);
  auto reshape_shape = loco::must_cast<luci::CircleConst *>(reshape->shape());
  ASSERT_EQ(4, reshape_shape->size<loco::DataType::S32>());
  ASSERT_EQ(16, reshape_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(3, reshape_shape->at<loco::DataType::S32>(1));
  ASSERT_EQ(1, reshape_shape->at<loco::DataType::S32>(2));
  ASSERT_EQ(1, reshape_shape->at<loco::DataType::S32>(3));
}

TEST_F(SubstituteExpandDimsToReshapeTest, nothing_to_expand_dims)
{
  _graph.init({2, 16, 16, 3}, {2, 16, 16, 3}, {});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto expand_dims = dynamic_cast<luci::CircleExpandDims *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, expand_dims);
}
