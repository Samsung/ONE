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
#include "luci/Pass/ForwardReshapeToUnaryOpPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

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

    _output = _g.nodes()->create<luci::CircleOutput>();
    _output->shape(shape_out);
    _output->shape_status(luci::ShapeStatus::VALID);

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

class ForwardReshapeToNegGraph : public PassTestGraph
{
public:
  ForwardReshapeToNegGraph() = default;

public:
  void init(const uilist shape_in, const uilist shape_out)
  {
    PassTestGraph::init(shape_in, shape_out);

    std::vector<uint32_t> shape_out_v = shape_out;

    _reshape_shape = _g.nodes()->create<luci::CircleConst>();
    _reshape = _g.nodes()->create<luci::CircleReshape>();
    _neg = _g.nodes()->create<luci::CircleNeg>();

    _reshape_shape->dtype(loco::DataType::S32);
    _reshape_shape->rank(1);
    _reshape_shape->dim(0).set(shape_out_v.size());
    _reshape_shape->shape_status(luci::ShapeStatus::VALID);
    // values
    const auto size = shape_out_v.size();
    _reshape_shape->size<loco::DataType::S32>(size);
    for (uint32_t i = 0; i < size; i++)
      _reshape_shape->at<loco::DataType::S32>(i) = shape_out_v[i];

    _reshape->tensor(_input);
    _reshape->shape(_reshape_shape);
    _neg->x(_reshape);
    _output->from(_neg);
  }

protected:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleNeg *_neg = nullptr;
  luci::CircleConst *_reshape_shape = nullptr;
};

class ForwardReshapeToNegGraphTest : public ::testing::Test
{
public:
  ForwardReshapeToNegGraphTest() = default;

  void run_pass(void)
  {
    while (_pass.run(_graph.g()))
      ;
  }

protected:
  ForwardReshapeToNegGraph _graph;
  luci::ForwardReshapeToUnaryOpPass _pass;
};

} // namespace

TEST_F(ForwardReshapeToNegGraphTest, simple_forward)
{
  _graph.init({2, 2, 2}, {2, 4});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto neg = dynamic_cast<luci::CircleNeg *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, neg);
  neg = dynamic_cast<luci::CircleNeg *>(reshape->tensor());
  ASSERT_NE(nullptr, neg);
}
