/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <logo/Phase.h>

#include "luci/Pass/ConvertNCHWToNHWCPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph with a single Op (example: Add).
 *
 *  BEFORE
 *  - All Ops including Input/Output are NCHW.
 *
 *             [Input] [beta]
 *                |  /
 *              [Add]
 *                |
 *             [Output]
 *
 *  AFTER
 *  - All Ops including Input/Output are NHWC.
 *
 *             [Input]
 *                |
 *         [Transpose]
 *                |
 *        [Transpose] [beta]
 *                |  /
 *              [Add]
 *                |
 *         [Transpose]
 *                |
 *         [Transpose]
 *                |
 *             [Output]
 */
class SimpleGraph
{
public:
  SimpleGraph() = default;

public:
  void init()
  {
    input = g.nodes()->create<luci::CircleInput>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    graph_input->dtype(loco::DataType::FLOAT32);
    input->dtype(loco::DataType::FLOAT32);
    output->dtype(loco::DataType::FLOAT32);
    graph_output->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    graph_input->shape({1, channel_size, 4, 4});
    input->shape({1, channel_size, 4, 4});
    output->shape({1, channel_size, 4, 4});
    graph_output->shape({1, channel_size, 4, 4});

    auto graph_body = insertGraphBody(input);
    output->from(graph_body);
  }

  virtual ~SimpleGraph() = default;

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

class AddGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    add = g.nodes()->create<luci::CircleAdd>();
    beta = g.nodes()->create<luci::CircleConst>();

    add->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    add->shape({1, channel_size, 4, 4});
    beta->shape({1, channel_size, 1, 1});

    beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }

    add->x(input);
    add->y(beta);

    return add;
  }

public:
  luci::CircleAdd *add = nullptr;
  luci::CircleConst *beta = nullptr;
};

class MulGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    mul = g.nodes()->create<luci::CircleMul>();
    multiplier = g.nodes()->create<luci::CircleConst>();

    mul->dtype(loco::DataType::FLOAT32);
    multiplier->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    mul->shape({1, channel_size, 4, 4});
    multiplier->shape({1, channel_size, 1, 1});

    multiplier->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      multiplier->at<loco::DataType::FLOAT32>(i) = i;
    }

    mul->x(input);
    mul->y(multiplier);

    return mul;
  }

public:
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *multiplier = nullptr;
};

class PadGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    pad = g.nodes()->create<luci::CirclePad>();
    paddings = g.nodes()->create<luci::CircleConst>();

    pad->dtype(loco::DataType::FLOAT32);
    paddings->dtype(loco::DataType::S32);

    uint32_t channel_size = 16;
    pad->shape({1, channel_size, 4, 4});
    paddings->shape({4, 2});

    // paddings data (NCHW)
    // [[0,0], [0,0], [1,1], [2,2]]
    paddings->size<loco::DataType::S32>(8);
    for (uint32_t dim = 0; dim < 4; dim++)
    {
      for (uint32_t i = 0; i < 2; i++)
      {
        int32_t data = 0;

        if (dim == 2)
          data = 1;
        else if (dim == 3)
          data = 2;

        paddings->at<loco::DataType::S32>(dim * 2 + i) = data;
      }
    }

    pad->input(input);
    pad->paddings(paddings);

    return pad;
  }

public:
  luci::CirclePad *pad = nullptr;
  luci::CircleConst *paddings = nullptr;
};

class Relu6Graph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    relu6 = g.nodes()->create<luci::CircleRelu6>();
    relu6->features(input);

    return relu6;
  }

public:
  luci::CircleRelu6 *relu6 = nullptr;
};

void check_pre_trans(loco::Node *node)
{
  auto pre_trans = dynamic_cast<luci::CircleTranspose *>(node);
  EXPECT_NE(nullptr, pre_trans);
  auto pre_trans_perm = dynamic_cast<luci::CircleConst *>(pre_trans->perm());
  EXPECT_NE(nullptr, pre_trans_perm);
  EXPECT_EQ(1, pre_trans_perm->rank());
  EXPECT_EQ(4, pre_trans_perm->dim(0).value());
  EXPECT_EQ(loco::DataType::S32, pre_trans_perm->dtype());
  EXPECT_EQ(0, pre_trans_perm->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, pre_trans_perm->at<loco::DataType::S32>(1));
  EXPECT_EQ(3, pre_trans_perm->at<loco::DataType::S32>(2));
  EXPECT_EQ(1, pre_trans_perm->at<loco::DataType::S32>(3));
}

void check_post_trans(loco::Node *node)
{
  auto post_trans = dynamic_cast<luci::CircleTranspose *>(node);
  EXPECT_NE(nullptr, post_trans);
  auto post_trans_perm = dynamic_cast<luci::CircleConst *>(post_trans->perm());
  EXPECT_NE(nullptr, post_trans_perm);
  EXPECT_EQ(1, post_trans_perm->rank());
  EXPECT_EQ(4, post_trans_perm->dim(0).value());
  EXPECT_EQ(loco::DataType::S32, post_trans_perm->dtype());
  EXPECT_EQ(0, post_trans_perm->at<loco::DataType::S32>(0));
  EXPECT_EQ(3, post_trans_perm->at<loco::DataType::S32>(1));
  EXPECT_EQ(1, post_trans_perm->at<loco::DataType::S32>(2));
  EXPECT_EQ(2, post_trans_perm->at<loco::DataType::S32>(3));
}

void run_phase(loco::Graph *g, bool preserve_input, bool preserve_output)
{
  logo::Phase phase;

  // Default passes.
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());

  // Pass to test
  phase.emplace_back(
    std::make_unique<luci::ConvertNCHWToNHWCPass>(preserve_input, preserve_output));

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.run(phase);
}

} // namespace

TEST(ConvertNCHWToNHWC, Add)
{
  AddGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.add->x());

  auto add_succs = loco::succs(g.add);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  uint32_t channel_size = 16;
  auto new_beta = dynamic_cast<luci::CircleConst *>(g.add->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(1, new_beta->dim(1).value());
  EXPECT_EQ(1, new_beta->dim(2).value());
  EXPECT_EQ(channel_size, new_beta->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Mul)
{
  MulGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.mul->x());

  auto mul_succs = loco::succs(g.mul);
  EXPECT_EQ(1, mul_succs.size());
  check_post_trans(*mul_succs.begin());

  uint32_t channel_size = 16;
  auto new_multiplier = dynamic_cast<luci::CircleConst *>(g.mul->y());
  EXPECT_NE(nullptr, new_multiplier);
  EXPECT_EQ(4, new_multiplier->rank());
  EXPECT_EQ(1, new_multiplier->dim(0).value());
  EXPECT_EQ(1, new_multiplier->dim(1).value());
  EXPECT_EQ(1, new_multiplier->dim(2).value());
  EXPECT_EQ(channel_size, new_multiplier->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Pad)
{
  PadGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.pad->input());

  auto pad_succs = loco::succs(g.pad);
  EXPECT_EQ(1, pad_succs.size());
  check_post_trans(*pad_succs.begin());

  auto new_paddings = dynamic_cast<luci::CircleConst *>(g.pad->paddings());
  EXPECT_NE(nullptr, new_paddings);
  EXPECT_EQ(2, new_paddings->rank());
  EXPECT_EQ(4, new_paddings->dim(0).value());
  EXPECT_EQ(2, new_paddings->dim(1).value());
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(0));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(1));
  EXPECT_EQ(1, new_paddings->at<loco::DataType::S32>(2));
  EXPECT_EQ(1, new_paddings->at<loco::DataType::S32>(3));
  EXPECT_EQ(2, new_paddings->at<loco::DataType::S32>(4));
  EXPECT_EQ(2, new_paddings->at<loco::DataType::S32>(5));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(6));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(7));

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Unknown_Shape_NEG)
{
  AddGraph g;
  g.init();

  // Unknown shape
  g.input->dim(0).unset();
  g.add->dim(0).unset();
  g.output->dim(0).unset();

  luci::ConvertNCHWToNHWCPass pass(false, false);
  EXPECT_EQ(false, pass.run(&g.g));
}

TEST(ConvertNCHWToNHWC, Preserve_Input_Output)
{
  // Preserve input
  {
    AddGraph g;
    g.init();

    run_phase(&g.g, true, false);

    // Check input shape
    EXPECT_EQ(1, g.input->dim(0).value());
    EXPECT_EQ(16, g.input->dim(1).value());
    EXPECT_EQ(4, g.input->dim(2).value());
    EXPECT_EQ(4, g.input->dim(3).value());

    // Check output shape
    EXPECT_EQ(1, g.output->dim(0).value());
    EXPECT_EQ(4, g.output->dim(1).value());
    EXPECT_EQ(4, g.output->dim(2).value());
    EXPECT_EQ(16, g.output->dim(3).value());
  }

  // Preserve output
  {
    AddGraph g;
    g.init();

    run_phase(&g.g, false, true);

    // Check input shape
    EXPECT_EQ(1, g.input->dim(0).value());
    EXPECT_EQ(4, g.input->dim(1).value());
    EXPECT_EQ(4, g.input->dim(2).value());
    EXPECT_EQ(16, g.input->dim(3).value());

    // Check output shape
    EXPECT_EQ(1, g.output->dim(0).value());
    EXPECT_EQ(16, g.output->dim(1).value());
    EXPECT_EQ(4, g.output->dim(2).value());
    EXPECT_EQ(4, g.output->dim(3).value());
  }

  // Preserve both input and output
  {
    AddGraph g;
    g.init();

    run_phase(&g.g, true, true);

    // Check input shape
    EXPECT_EQ(1, g.input->dim(0).value());
    EXPECT_EQ(16, g.input->dim(1).value());
    EXPECT_EQ(4, g.input->dim(2).value());
    EXPECT_EQ(4, g.input->dim(3).value());

    // Check output shape
    EXPECT_EQ(1, g.output->dim(0).value());
    EXPECT_EQ(16, g.output->dim(1).value());
    EXPECT_EQ(4, g.output->dim(2).value());
    EXPECT_EQ(4, g.output->dim(3).value());
  }
}

TEST(ConvertNCHWToNHWC, Relu6)
{
  Relu6Graph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.relu6->features());

  auto relu6_succs = loco::succs(g.relu6);
  EXPECT_EQ(1, relu6_succs.size());
  check_post_trans(*relu6_succs.begin());

  // Check relu6 shape
  EXPECT_EQ(1, g.relu6->dim(0).value());
  EXPECT_EQ(4, g.relu6->dim(1).value());
  EXPECT_EQ(4, g.relu6->dim(2).value());
  EXPECT_EQ(16, g.relu6->dim(3).value());
}
