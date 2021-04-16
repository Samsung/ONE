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

#include "ConnectNode.h"

#include "ConnectNode.test.h"

#include <luci/Service/CircleNodeClone.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class NodeGraphlet : public NodeGraphletT<luci::CircleUnidirectionalSequenceLSTM>
{
public:
  NodeGraphlet() = default;

public:
  void init(loco::Graph *g) override
  {
    NodeGraphletT<luci::CircleUnidirectionalSequenceLSTM>::init(g);

    _node->fusedActivationFunction(luci::FusedActFunc::RELU);
  }
};

class TestNodeGraph : public TestIsOGraph<24>, public NodeGraphlet
{
public:
  TestNodeGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIsOGraph<24>::init({shape, shape, shape, shape, shape, shape, shape, shape,
                            shape, shape, shape, shape, shape, shape, shape, shape,
                            shape, shape, shape, shape, shape, shape, shape, shape},
                           shape);
    NodeGraphlet::init(g());

    node()->input(input(0));

    node()->input_to_input_weights(input(1));
    node()->input_to_forget_weights(input(2));
    node()->input_to_cell_weights(input(3));
    node()->input_to_output_weights(input(4));

    node()->recurrent_to_input_weights(input(5));
    node()->recurrent_to_forget_weights(input(6));
    node()->recurrent_to_cell_weights(input(7));
    node()->recurrent_to_output_weights(input(8));

    node()->cell_to_input_weights(input(9));
    node()->cell_to_forget_weights(input(10));
    node()->cell_to_output_weights(input(11));

    node()->input_gate_bias(input(12));
    node()->forget_gate_bias(input(13));
    node()->cell_gate_bias(input(14));
    node()->output_gate_bias(input(15));

    node()->projection_weights(input(16));
    node()->projection_bias(input(17));

    node()->activation_state(input(18));
    node()->cell_state(input(19));

    node()->input_layer_norm_coefficients(input(20));
    node()->forget_layer_norm_coefficients(input(21));
    node()->cell_layer_norm_coefficients(input(22));
    node()->output_layer_norm_coefficients(input(23));

    output()->from(node());
  }
};

} // namespace

TEST(ConnectNodeTest, connect_UnidirectionalSequenceLSTM)
{
  TestNodeGraph tng;
  tng.init({2, 3});

  ConnectionTestHelper cth;
  cth.prepare_inputs(&tng);

  auto *node = tng.node();
  ASSERT_NO_THROW(loco::must_cast<luci::CircleUnidirectionalSequenceLSTM *>(node));

  auto *clone = luci::clone_node(node, cth.graph_clone());
  cth.clone_connect(node, clone);

  ASSERT_EQ(24, clone->arity());
  for (uint32_t i = 0; i < 24; ++i)
    ASSERT_EQ(cth.inputs(i), clone->arg(i));
}
