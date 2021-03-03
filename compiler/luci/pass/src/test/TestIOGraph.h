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

#ifndef __LUCI_PASS_TEST_IO_GRAPH_H__
#define __LUCI_PASS_TEST_IO_GRAPH_H__

#include "TestShape.h"

#include <luci/IR/CircleNodes.h>

namespace luci
{
namespace test
{

/**
 * @brief Graphlet with Inputs and loco::Graph for multiple inputs
 * @note  Every Graph will have Input(s) and Output(s)
 *        We put loco::Graph only in IsGraphlet not to declare separate
 *        class for loco::Graph
 */
template <unsigned N> class TestIsGraphlet
{
public:
  TestIsGraphlet() = default;

public:
  virtual void init(loco::Graph *g, const ShapeU32 shape_in)
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _graph_inputs[n] = g->inputs()->create();

      _inputs[n] = g->nodes()->create<luci::CircleInput>();
      _inputs[n]->shape(shape_in);
      _inputs[n]->shape_status(luci::ShapeStatus::VALID);
      _inputs[n]->dtype(loco::DataType::FLOAT32);
      _inputs[n]->name("input_" + std::to_string(n));

      _inputs[n]->index(_graph_inputs[n]->index());

      auto input_shape = std::make_unique<loco::TensorShape>();
      set_shape_vector(input_shape.get(), shape_in);
      _graph_inputs[n]->shape(std::move(input_shape));
      _graph_inputs[n]->dtype(loco::DataType::FLOAT32);
    }
  }

public:
  loco::Graph *g(void) { return &_g; }
  luci::CircleInput *input(int idx) { return _inputs[idx]; }

protected:
  loco::Graph _g;
  std::array<loco::GraphInput *, N> _graph_inputs{};
  std::array<luci::CircleInput *, N> _inputs{};
};

/**
 * @brief Graphlet with one Input
 */
class TestIGraphlet : public TestIsGraphlet<1>
{
public:
  luci::CircleInput *input() { return _inputs[0]; }
};

/**
 * @brief Graphlet with Outputs for multiple outputs
 */
template <unsigned N> class TestOsGraphlet
{
public:
  TestOsGraphlet() = default;

public:
  virtual void init(loco::Graph *g, const ShapeU32 shape_out)
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _graph_outputs[n] = g->outputs()->create();

      _outputs[n] = g->nodes()->create<luci::CircleOutput>();
      _outputs[n]->shape(shape_out);
      _outputs[n]->shape_status(luci::ShapeStatus::VALID);
      _outputs[n]->dtype(loco::DataType::FLOAT32);
      _outputs[n]->name("output_" + std::to_string(n));

      _outputs[n]->index(_graph_outputs[n]->index());

      auto output_shape = std::make_unique<loco::TensorShape>();
      set_shape_vector(output_shape.get(), shape_out);
      _graph_outputs[n]->shape(std::move(output_shape));
      _graph_outputs[n]->dtype(loco::DataType::FLOAT32);
    }
  }

public:
  luci::CircleOutput *output(int idx) { return _outputs[idx]; }

protected:
  std::array<loco::GraphOutput *, N> _graph_outputs{};
  std::array<luci::CircleOutput *, N> _outputs{};
};

/**
 * @brief Graphlet with one Output
 */
class TestOGraphlet : public TestOsGraphlet<1>
{
public:
  luci::CircleOutput *output() { return _outputs[0]; }
};

/**
 * @brief Graph with Input and Output
 */
class TestIOGraph : public TestIGraphlet, public TestOGraphlet
{
public:
  TestIOGraph() = default;

public:
  virtual void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIsGraphlet<1>::init(g(), shape_in);
    TestOsGraphlet<1>::init(g(), shape_out);
  }
};

} // namespace test
} // namespace luci

#endif // __LUCI_PASS_TEST_IO_GRAPH_H__
