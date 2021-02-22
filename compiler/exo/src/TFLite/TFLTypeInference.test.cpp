/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "TFLTypeInference.h"
#include "Pass/TypeInferencePass.h"

#include <loco/IR/PermutingCodec.h>

#include <gtest/gtest.h>

namespace
{

class Sequential
{
public:
  loco::Pull *addPullLayer(const loco::DataType &dtype = loco::DataType::FLOAT32)
  {
    loco::Pull *pull = _graph.nodes()->create<loco::Pull>();

    auto graph_input = _graph.inputs()->create();
    graph_input->name("graph_input");
    loco::link(graph_input, pull);

    pull->dtype(dtype);
    setSampleShape(pull);

    return last(pull);
  }

  loco::ReLU *addReLULayer(void)
  {
    loco::ReLU *relu = _graph.nodes()->create<loco::ReLU>();

    relu->input(_last);

    return last(relu);
  }

  loco::Push *addPushLayer(void)
  {
    loco::Push *push = _graph.nodes()->create<loco::Push>();

    auto graph_output = _graph.outputs()->create();
    graph_output->name("graph_output");
    loco::link(graph_output, push);

    push->from(_last);

    return last(push);
  }

  loco::Graph *graph() { return &_graph; }

private:
  template <typename T> uint32_t setSampleShape(T *op)
  {
    const uint32_t n = 1;
    const uint32_t h = 100;
    const uint32_t w = 100;
    const uint32_t c = 3;
    op->rank(4);
    op->dim(0).set(n);
    op->dim(1).set(c);
    op->dim(2).set(h);
    op->dim(3).set(w);
    return n * h * w * c;
  }

  template <typename T> T *last(T *node)
  {
    _last = node;
    return node;
  }

private:
  loco::Graph _graph;
  loco::Node *_last{nullptr};
};

struct TypeInferenceTest : public Sequential, public ::testing::Test
{
  virtual ~TypeInferenceTest() = default;
};

} // namespace

// TypeInference SHOULD PROPAGATE type information properly
TEST_F(TypeInferenceTest, Regression_0000)
{
  auto pull = addPullLayer(loco::DataType::S8);
  auto relu = addReLULayer();
  auto push = addPushLayer();

  using namespace exo;

  TypeInferencePass type_inf_pass;
  type_inf_pass.run(graph());

  ASSERT_EQ(tflite::TensorType_INT8, TypeInference::get(relu));
  ASSERT_EQ(tflite::TensorType_INT8, TypeInference::get(push));
}
