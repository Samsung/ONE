/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/DecomposeSoftmaxPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Softmax graph
 *
 *        [CircleInput]
 *              |
 *              |
 *       [CircleSoftMax]
 *              |
 *              |
 *        [CircleOutput]
 */
template <loco::DataType T> struct SoftmaxGraph
{
  loco::Graph _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleSoftmax *_softmax = nullptr;
  luci::CircleOutput *_output = nullptr;

  SoftmaxGraph()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3;

    // graph input and output
    auto graph_input = _g.inputs()->create();
    auto graph_output = _g.outputs()->create();

    // CircleInput
    _input = _g.nodes()->create<luci::CircleInput>();
    _input->index(graph_input->index());
    _input->shape({N, H, W, C});
    _input->dtype(T);
    _input->name("input");

    // CircleSoftmax
    _softmax = _g.nodes()->create<luci::CircleSoftmax>();
    _softmax->logits(_input);
    _softmax->shape({N, H, W, C});
    _softmax->dtype(T);
    _softmax->name("softmax");
    _softmax->beta(0.5f);

    // CircleOutput
    _output = _g.nodes()->create<luci::CircleOutput>();
    _output->index(graph_output->index());
    _output->from(_softmax);
    _output->shape({N, H, W, C});
    _output->dtype(T);
    _output->name("output");
  }
};

} // namespace

TEST(DecomposeSoftmaxPass, simple_test)
{
  /**
   *  tests:
   *    1) decomposition pass has nonnull name
   *    2) decomposition runs successfully for float32 softmax graph
   *    3) resulting graph has the following structure:
   *
   *      [CircleNode]   [CircleConst(=-1)]
   *          |    \       /           |
   *          |     \     /            |
   *          | [CircleReduceMax]      |
   *          |    /                   |
   *          |   /                    |
   *          |  /                     |
   *        [Sub]                      |
   *          |                        |
   *          | [CircleConst(=0.5)]    |
   *          |   /                    |
   *          |  /                     |
   *        [Mul]                      |
   *          |                        |
   *        [Exp]                      |
   *          | \                      |
   *          |  \                     |
   *          |  [CircleSum]-----------+
   *          |  /
   *          | /
   *        [Div]
   *          |
   *          |
   *      [CircleNode]
   */
  luci::DecomposeSoftmaxPass pass;
  SoftmaxGraph<loco::DataType::FLOAT32> softmax_g;

  auto const name = pass.name();
  ASSERT_NE(nullptr, name);

  auto ret = pass.run(&softmax_g._g);
  EXPECT_TRUE(ret);

  auto div = dynamic_cast<luci::CircleDiv *>(softmax_g._output->from());
  EXPECT_NE(nullptr, div);

  auto exp = dynamic_cast<luci::CircleExp *>(div->x());
  EXPECT_NE(nullptr, exp);

  auto sum = dynamic_cast<luci::CircleSum *>(div->y());
  EXPECT_NE(nullptr, sum);

  auto exp_1 = dynamic_cast<luci::CircleExp *>(sum->input());
  EXPECT_EQ(exp, exp_1);

  auto indices = dynamic_cast<luci::CircleConst *>(sum->reduction_indices());
  EXPECT_NE(nullptr, indices);
  EXPECT_EQ(indices->dtype(), loco::DataType::S32);
  EXPECT_EQ(indices->size<loco::DataType::S32>(), 1);
  EXPECT_EQ(indices->scalar<loco::DataType::S32>(), -1);

  auto mul = dynamic_cast<luci::CircleMul *>(exp->x());
  EXPECT_NE(nullptr, mul);

  auto sub = dynamic_cast<luci::CircleSub *>(mul->x());
  EXPECT_NE(nullptr, sub);

  auto beta = dynamic_cast<luci::CircleConst *>(mul->y());
  EXPECT_NE(nullptr, beta);
  EXPECT_EQ(beta->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(beta->size<loco::DataType::FLOAT32>(), 1);
  EXPECT_FLOAT_EQ(beta->scalar<loco::DataType::FLOAT32>(), 0.5f);

  auto input = dynamic_cast<luci::CircleInput *>(sub->x());
  EXPECT_NE(nullptr, input);

  auto max = dynamic_cast<luci::CircleReduceMax *>(sub->y());
  EXPECT_NE(nullptr, max);

  auto indices_1 = dynamic_cast<luci::CircleConst *>(max->reduction_indices());
  EXPECT_NE(nullptr, indices_1);
  EXPECT_EQ(indices, indices_1);

  auto input_1 = dynamic_cast<luci::CircleInput *>(max->input());
  EXPECT_NE(nullptr, input_1);
  EXPECT_EQ(input, input_1);
}

TEST(DecomposeSoftmaxPass, wrong_condition_NEG)
{
  luci::DecomposeSoftmaxPass pass;
  SoftmaxGraph<loco::DataType::S32> softmax_g;

  auto ret = pass.run(&softmax_g._g);
  EXPECT_FALSE(ret);

  auto softmax = dynamic_cast<luci::CircleSoftmax *>(softmax_g._output->from());
  EXPECT_NE(nullptr, softmax);
}
